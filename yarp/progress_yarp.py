# progress_yarp.py
import sys
import json
import pickle
from pathlib import Path

from yarp.util.input import InputParser
from yarp.reaction.external.job_manager import get_job_manager
from yarp.reaction.external.calc_factory import get_calculator

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

def progress_yarp(work_dir: Path):
    print(f"Processing YARP progress in {work_dir}...")

    # 1. Load the state
    status_tracker, reactions = load_state(work_dir)

    # 2. Parse the configuration
    config = InputParser(status_tracker["input_config"])

    # Notice these are direct attributes of `config` based on your InputParser!
    scheduler = config.job_manager.scheduler
    container_runner = config.job_manager.container
    max_active_jobs = config.job_manager.max_active_jobs

    job_manager = get_job_manager(scheduler)

    print(f"Jobs will be run using '{scheduler}' scheduler and '{container_runner}' containers.")
    print(f"Max active jobs allowed: {max_active_jobs}")

    # =================================================================
    # PASS 1: Check Status of Submitted Jobs & Tally Active Jobs
    # =================================================================
    active_jobs = 0
    failed_rxns = {}

    # =================================================================
    # PASS 1A: Check Status of Submitted GLOBAL Jobs
    # =================================================================
    for g_task_id, g_meta in status_tracker.get("global_tasks", {}).items():
        if g_meta["status"] == "submitted":
            task_def = config.global_tasks[g_task_id]

            calc = get_calculator(task_def, reactions, config.job_manager)

            if g_meta["scratch_dir"]:
                calc.set_scratch_dir(Path(g_meta["scratch_dir"]))

            if not job_manager.is_running(g_meta["job_id"]):
                if calc.check_output():
                    print(f" * Global Task '{g_task_id}' finished successfully!")
                    g_meta["status"] = "terminated_normally"
                    calc.scrape_data()
                    calc.cleanup()
                else:
                    print(f" * Global Task '{g_task_id}' failed.")
                    g_meta["status"] = "finished_with_error"
            else:
                active_jobs += 1

    # =================================================================
    # PASS 1B: Check Status of Submitted PIPELINE Jobs
    # =================================================================
    for rxn_hash, rxn_meta in status_tracker["reactions"].items():
        rxn_obj = reactions.get(rxn_hash)
        if not rxn_obj: continue

        for task_id, meta in rxn_meta["tasks"].items():
            if meta["status"] == "submitted":
                task_def = config.pipeline_tasks[task_id]
                calc = get_calculator(task_def, rxn_obj, config.job_manager)

                if meta["scratch_dir"]:
                    calc.set_scratch_dir(Path(meta["scratch_dir"]))

                if job_manager.is_running(meta["job_id"]):
                    # Job is still going, add it to our tally!
                    print(f"   * [{rxn_hash}] Task '{task_id}' is still running. Come back later...")
                    active_jobs += 1
                else:
                    # Job finished! Time to process it.
                    print(f"   * [{rxn_hash}] Task '{task_id}' finished running. Checking output...")
                    if calc.check_output():
                        if calc.scrape_data():
                            calc.cleanup()
                            meta["status"] = "terminated_normally"
                            print(f"   * [{rxn_hash}] Task '{task_id}' completed successfully.")
                        else:
                            meta["status"] = "finished_with_error"
                            meta["error_log"] = "Data scraping failed."
                            failed_rxns[rxn_hash] = rxn_obj
                            print(f"   * [{rxn_hash}] Task '{task_id}' failed during data scraping.")
                    else:
                        meta["status"] = "finished_with_error"
                        meta["error_log"] = "Output validation failed."
                        failed_rxns[rxn_hash] = rxn_obj
                        print(f"   * [{rxn_hash}] Task '{task_id}' failed output validation.")
 
    # =================================================================
    # PASS 2: Update Pending Tasks to Ready
    # =================================================================
    for rxn_hash, rxn_data in status_tracker["reactions"].items():
        tasks_status = rxn_data["tasks"]
        for task_id, meta in tasks_status.items():
            if meta["status"] == "pending":
                task_def = config.pipeline_tasks[task_id]
                
                deps_met = True
                for dep_id in task_def.depends_on:
                    # Check if the dependency is a global task
                    if dep_id in status_tracker.get("global_tasks", {}):
                        if status_tracker["global_tasks"][dep_id]["status"] != "terminated_normally":
                            deps_met = False
                            break
                    # Otherwise, check the local pipeline tasks
                    else:
                        if tasks_status[dep_id]["status"] != "terminated_normally":
                            deps_met = False
                            break
                
                if deps_met:
                    # --- NEW: Evaluate Pre-Process Filters ---
                    stage_filter = config.stage_filters.get(task_def.parent_stage)

                    # Only evaluate at the entry points of the pipeline
                    if stage_filter and task_def.task_type in ["reactant_conformer", "product_conformer"]:
                        rxn_obj = reactions[rxn_hash]

                        # Dynamically grab the dictionary from the reaction object (e.g. rxn.barrier)
                        prop_dict = getattr(rxn_obj, stage_filter.property, None)

                        if prop_dict is not None and stage_filter.source in prop_dict:
                            val = prop_dict[stage_filter.source]

                            # If it exceeds the threshold, kill the run cleanly
                            if val > stage_filter.threshold:
                                meta["status"] = "filtered_out"
                                meta["error_log"] = f"Filtered out: {stage_filter.property} ({val:.2f}) > threshold ({stage_filter.threshold})"

                                # Prevent printing the error twice (once for reactant, once for product)
                                if rxn_hash not in failed_rxns:
                                    failed_rxns[rxn_hash] = rxn_obj
                                    print(f"   * [{rxn_hash}] Filtered out! {stage_filter.property} ({val:.2f}) > {stage_filter.threshold}")

                                continue # Skip setting this task to 'ready'
                        else:
                            # If EGAT finished but data is missing entirely, fail it out
                            meta["status"] = "finished_with_error"
                            meta["error_log"] = f"Missing filter property: {stage_filter.property} from {stage_filter.source}"

                            if rxn_hash not in failed_rxns:
                                failed_rxns[rxn_hash] = rxn_obj
                                print(f"   * [{rxn_hash}] Pipeline aborted: Missing ML property '{stage_filter.property}' from '{stage_filter.source}'.")

                            continue # Skip setting this task to 'ready'

                    meta["status"] = "ready"
                    print(f"   * [{rxn_hash}] Task '{task_id}' prerequisites met. Marked as READY.")

    # =================================================================
    # PASS 3: Submit New Jobs (Respecting the Limit)
    # =================================================================
    print(f"Current active jobs: {active_jobs} / {max_active_jobs}")

    # =================================================================
    # PASS 3A: Submit New GLOBAL Jobs
    # =================================================================
    for g_task_id, g_meta in status_tracker.get("global_tasks", {}).items():
        # Global limit check
        if active_jobs >= max_active_jobs:
            print(f"Max active jobs ({max_active_jobs}) reached. Holding off on further submissions.")
            break

        if g_meta["status"] == "ready":
                
            task_def = config.global_tasks[g_task_id]
            calc = get_calculator(task_def, reactions, config.job_manager)
            
            scratch_path = work_dir / "SCRATCH" / f"GLOBAL_{g_task_id}"
            calc.set_scratch_dir(scratch_path)

            if not calc.has_prerequisites():
                g_meta["status"] = "finished_with_error"
                print(f" * Global Task '{g_task_id}' aborted: Pre-flight check failed.")
                continue

            print(f" * Submitting global task '{g_task_id}'...")
            calc.generate_input()
            script_path = calc.write_submission_script()

            job_id = job_manager.submit(script_path)

            if job_id:
                g_meta["status"] = "submitted"
                g_meta["job_id"] = job_id
                g_meta["scratch_dir"] = str(scratch_path)
                active_jobs += 1
            else:
                g_meta["status"] = "finished_with_error"


    # =================================================================
    # PASS 3B: Submit New PIPELINE Jobs
    # =================================================================
    for rxn_hash, rxn_meta in status_tracker["reactions"].items():
        # Global limit check
        if active_jobs >= max_active_jobs:
            print(f"Max active jobs ({max_active_jobs}) reached. Holding off on further submissions.")
            break

        rxn_obj = reactions.get(rxn_hash)
        if not rxn_obj: continue

        for task_id, meta in rxn_meta["tasks"].items():
            # Inner limit check (in case we hit the limit mid-reaction)
            if active_jobs >= max_active_jobs:
                break 

            if meta["status"] == "ready":
                task_def = config.pipeline_tasks[task_id]
                calc = get_calculator(task_def, rxn_obj, config.job_manager)

                scratch_path = work_dir / "SCRATCH" / f"{rxn_hash}_{task_id}"
                calc.set_scratch_dir(scratch_path)

                # Pre-flight Data Check
                if not calc.has_prerequisites():
                    meta["status"] = "finished_with_error"
                    meta["error_log"] = "Pre-flight check failed: Missing required data in reaction object."
                    failed_rxns[rxn_hash] = rxn_obj
                    print(f"   * [{rxn_hash}] Task '{task_id}' aborted: Pre-flight check failed.")
                    continue

                # Generate and Submit
                print(f"   * [{rxn_hash}] Submitting task '{task_id}'...")
                calc.generate_input()
                script_path = calc.write_submission_script()

                job_id = job_manager.submit(script_path)

                if job_id:
                    meta["status"] = "submitted"
                    meta["job_id"] = job_id
                    meta["scratch_dir"] = str(scratch_path)
                    active_jobs += 1 # Increment our tally!
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
    absolute_work_dir = Path(sys.argv[1]).resolve()
    progress_yarp(absolute_work_dir)