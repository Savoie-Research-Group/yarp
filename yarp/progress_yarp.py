# progress_yarp.py
import sys
import json
import pickle
from pathlib import Path

# In a real environment, you'd import these from your yarp modules
from yarp.util.input import InputParser
from yarp.reaction.external.job_manager import get_job_manager
from yarp.reaction.external.calculator import get_calculator

def load_state(work_dir: Path):
    with open(work_dir / "STATUS.json", "r") as f:
        status_tracker = json.load(f)
    with open(work_dir / "YARP_RXNS.pkl", "rb") as f:
        reactions = pickle.load(f)
    return status_tracker, reactions

def save_state(work_dir: Path, status_tracker: dict, reactions: dict, failed_rxns: dict):
    with open(work_dir / "STATUS.json", "w") as f:
        json.dump(status_tracker, f, indent=4)
    with open(work_dir / "YARP_RXNS.pkl", "wb") as f:
        pickle.dump(reactions, f)
        
    if failed_rxns:
        # Append or create a separate pickle file for failed runs
        fail_file = work_dir / "failed_rxns.pkl"
        existing_fails = {}
        if fail_file.exists():
            with open(fail_file, "rb") as f:
                existing_fails = pickle.load(f)
        existing_fails.update(failed_rxns)
        with open(fail_file, "wb") as f:
            pickle.dump(existing_fails, f)

def progress_yarp(work_dir_str: str):
    work_dir = Path(work_dir_str).resolve()
    status_tracker, reactions = load_state(work_dir)
    
    # We need the task definitions to know dependencies. 
    # We reconstruct the InputParser from the saved config.
    inp = InputParser(status_tracker["input_config"])
    pipeline_tasks = inp.pipeline_tasks
    
    # Initialize the correct job manager based on user input
    job_manager = get_job_manager(inp.scheduler)
    container_runner = getattr(inp, 'container', 'docker')
    failed_rxns = {}

    print(f"Processing YARP progress in {work_dir}...")

    for rxn_hash, rxn_state in status_tracker["reactions"].items():
        rxn_obj = reactions.get(rxn_hash)
        if not rxn_obj:
            continue
            
        tasks_status = rxn_state["tasks"]

        # --- PHASE 1: Check currently running jobs ---
        for task_id, meta in tasks_status.items():
            if meta["status"] == "submitted":
                if not job_manager.is_running(meta["job_id"]):
                    # Job finished! Let's check if it succeeded.
                    calc = get_calculator(pipeline_tasks[task_id], rxn_obj, container_runner=container_runner)
                    calc.set_scratch_dir(Path(meta["scratch_dir"]))
                    
                    if calc.check_output():
                        try:
                            calc.scrape_data()
                            calc.cleanup()
                            meta["status"] = "terminated_normally"
                            print(f"[{rxn_hash}] Task '{task_id}' completed successfully.")
                        except Exception as e:
                            meta["status"] = "finished_with_error"
                            meta["error_log"] = f"Scraping failed: {str(e)}"
                            failed_rxns[rxn_hash] = rxn_obj
                            print(f"[{rxn_hash}] Task '{task_id}' failed during data scraping.")
                    else:
                        meta["status"] = "finished_with_error"
                        meta["error_log"] = "External calculation failed or expected output missing."
                        failed_rxns[rxn_hash] = rxn_obj
                        print(f"[{rxn_hash}] Task '{task_id}' finished with errors.")

        # --- PHASE 2: Update Pending -> Ready based on DAG ---
        for task_id, meta in tasks_status.items():
            if meta["status"] == "pending":
                task_def = pipeline_tasks[task_id]
                
                # Check if all execution dependencies are met
                dependencies_met = True
                for dep_id in task_def.depends_on:
                    if tasks_status[dep_id]["status"] != "terminated_normally":
                        dependencies_met = False
                        break
                        
                if dependencies_met:
                    meta["status"] = "ready"
                    print(f"[{rxn_hash}] Task '{task_id}' prerequisites met. Marked as READY.")

        # --- PHASE 3: Submit Ready Jobs ---
        for task_id, meta in tasks_status.items():
            if meta["status"] == "ready":
                task_def = pipeline_tasks[task_id]
                calc = get_calculator(task_def, rxn_obj, container_runner=container_runner)
                
                scratch_path = work_dir / "SCRATCH" / f"{rxn_hash}_{task_id}"
                calc.set_scratch_dir(scratch_path)
                
                # Pre-flight Data Check
                if not calc.has_prerequisites():
                    meta["status"] = "finished_with_error"
                    meta["error_log"] = "Pre-flight check failed: Missing required data in reaction object."
                    failed_rxns[rxn_hash] = rxn_obj
                    print(f"[{rxn_hash}] Task '{task_id}' aborted: Pre-flight check failed.")
                    continue
                
                # Generate and Submit
                print(f"[{rxn_hash}] Submitting task '{task_id}'...")
                calc.generate_input()
                script_path = calc.write_submission_script()
                
                job_id = job_manager.submit(script_path)
                
                if job_id:
                    meta["status"] = "submitted"
                    meta["job_id"] = job_id
                    meta["scratch_dir"] = str(scratch_path)
                else:
                    meta["status"] = "finished_with_error"
                    meta["error_log"] = "Job manager failed to submit job."
                    failed_rxns[rxn_hash] = rxn_obj

    # Write all updates back to disk
    save_state(work_dir, status_tracker, reactions, failed_rxns)
    print("YARP progress tracking complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python progress_yarp.py /path/to/working/dir")
        sys.exit(1)
    progress_yarp(sys.argv[1])