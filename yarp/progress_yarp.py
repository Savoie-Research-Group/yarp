"""
Directed acyclic graph workflow manager for YARP jobs
"""
import argparse
import json
import pickle
from pathlib import Path

from yarp.util.input import InputParser
from yarp.reaction.external.job_manager import get_job_manager
from yarp.reaction.external.calc_factory import get_calculator

def load_state(work_dir: Path):
    """
    Load in status tracker and reaction objects from working directory
    """
    # Grab all JSONs, but explicitly ignore the failure log!
    json_files = [f for f in work_dir.glob("*.json") if f.name != "failed_status.json"]

    if not json_files:
        raise FileNotFoundError(f"No valid STATUS JSON file found in {work_dir}")

    json_path = json_files[0]

    # glob() returns full paths, so we can just open json_path directly
    with open(json_path, "r") as f:
        status_tracker = json.load(f)

    # ERM: I don't love this, as it's fragile, but it's the only way I can see to
    # allow for pytest to write the STATUS.json file to a temp dir during testing...

    rxn_file = status_tracker.get('reaction_output_file')
    with open(work_dir / rxn_file, "rb") as f:
        reactions = pickle.load(f)

    return status_tracker, reactions

def save_state(work_dir: Path, status_tracker: dict, reactions: dict, failed_rxns: dict):
    """
    Save tasks status and reactions to the working directory.
    If any reactions have failed in the pipeline, they are removed from
    the status tracker and output reaction files and collected elsewhere.
    """
    # ====================================================================
    # 1. PROCESS FAILURES (Extract Data & Save to Failed Logs)
    # ====================================================================
    if failed_rxns:
        # A. Update the Pickle file for programmatic recovery
        fail_file = work_dir / "failed_rxns.pkl"
        existing_fails = {}
        if fail_file.exists():
            with open(fail_file, "rb") as f:
                existing_fails = pickle.load(f)
        existing_fails.update(failed_rxns)
        with open(fail_file, "wb") as f:
            pickle.dump(existing_fails, f)

        # B. Create/Update the human-readable JSON log
        fail_json_file = work_dir / "failed_status.json"
        existing_fail_status = {}
        if fail_json_file.exists():
            with open(fail_json_file, "r") as f:
                existing_fail_status = json.load(f)

        # Extract the human-readable error reasons from the status_tracker
        for rxn_hash in failed_rxns.keys():
            if rxn_hash not in existing_fail_status:
                existing_fail_status[rxn_hash] = {}

            rxn_tasks = status_tracker.get("reactions", {}).get(rxn_hash, {}).get("tasks", {})
            for task_id, task_meta in rxn_tasks.items():
                # Catch both standard errors and programmatic filters (like IRC)
                if task_meta.get("status") in ["finished_with_error", "filtered_out"]:
                    existing_fail_status[rxn_hash][task_id] = {
                        "error_log": task_meta.get("error_log", "Unknown error"),
                        "scratch_dir": task_meta.get("scratch_dir", "N/A")
                    }

        with open(fail_json_file, "w") as f:
            json.dump(existing_fail_status, f, indent=4)

        # C. CLEANUP: Now that errors are extracted, delete from the active pool!
        for failed_hash in failed_rxns.keys():
            if failed_hash in reactions:
                del reactions[failed_hash]
            if "reactions" in status_tracker and failed_hash in status_tracker["reactions"]:
                del status_tracker["reactions"][failed_hash]

    # ====================================================================
    # 2. SAVE ACTIVE STATE (Dump cleanly purged boards to disk)
    # ====================================================================
    status_file = status_tracker.get('status_output_file')
    with open(work_dir / status_file, "w") as f:
        json.dump(status_tracker, f, indent=4)

    rxn_file = status_tracker.get('reaction_output_file')
    with open(work_dir / rxn_file, "wb") as f:
        pickle.dump(reactions, f)

def progress_yarp(work_dir: Path):
    """
    Primary logic flow for managing
    """
    print(f"Processing YARP progress in {work_dir}...")

    # Load the state
    status_tracker, reactions = load_state(work_dir)

    # Parse the configuration derived from OG user input file
    config = InputParser(status_tracker["input_config"])

    # Set up job manager
    scheduler = config.job_manager.scheduler
    container_runner = config.job_manager.container
    max_active_jobs = config.job_manager.max_active_jobs

    job_manager = get_job_manager(scheduler, config.job_manager)

    print(f"Jobs will be run using '{scheduler}' scheduler and '{container_runner}' containers.")
    print(f"Max active jobs allowed: {max_active_jobs}")

    # =================================================================
    # PASS 0.1: Synchronize Conformers Across Identical Species
    # =================================================================
    print("Synchronizing conformer data across identical chemical species...")
    species_conformer_pool = {}

    # 0.1.A Pool all conformers from all reactions using the unique hash
    for rxn_obj in reactions.values():
        for species in [rxn_obj.reactant, rxn_obj.product]:
            if not species: continue
            sp_hash = species.hash 
            if sp_hash not in species_conformer_pool:
                species_conformer_pool[sp_hash] = {}
            species_conformer_pool[sp_hash].update(species.conformers)

    # 0.1.B Distribute the enriched pools back to all reactions
    for rxn_obj in reactions.values():
        for species in [rxn_obj.reactant, rxn_obj.product]:
            if not species: continue
            species.conformers.update(species_conformer_pool[species.hash])

    # =================================================================
    # PASS 0.2: Fast-Forward Previously Characterized Reactions
    # =================================================================
    print("Performing pre-flight checks to skip already characterized reactions...")

    # --- PRE-COMPUTATION: Build the Reverse Dependency Graph ---
    # Find out which downstream tasks require a given upstream task
    required_by = {tid: [] for tid in config.pipeline_tasks.keys()}
    for task_id, task_def in config.pipeline_tasks.items():
        for dep_id in task_def.depends_on:
            # dep_id (upstream) is required by task_id (downstream)
            if dep_id in required_by: # Ensure it's a pipeline task, not global
                required_by[dep_id].append(task_id)

    # 0.2.A. Fast-Forward Pipeline Tasks (Local)
    for rxn_hash, rxn_meta in status_tracker["reactions"].items():
        rxn_obj = reactions.get(rxn_hash)
        if not rxn_obj: continue

        for task_id, meta in rxn_meta["tasks"].items():
            if meta["status"] in ["ready", "pending"]:
                task_def = config.pipeline_tasks[task_id]
                task_type = getattr(task_def, 'task_type', '')

                # Determine target species for the task (reactant or product)
                target_species = rxn_obj.reactant if "reactant" in task_type else rxn_obj.product

                already_done = False
                desired_key = ""

                # Make sure to pull out the right level of theory for GSM input block                     
                if task_type == "gsm":
                    lot = getattr(task_def.config, 'gsm_lot', None)
                    software = getattr(task_def.config, 'software', None)
                else:
                    lot = getattr(task_def.config, 'lot', None)
                    software = getattr(task_def.config, 'software', None)
                
                # Generate the desired data key
                desired_key = f"{lot}_{software}"
                
                # Check if reactant/product conformers have been generated
                if task_type in ["reactant_conformer", "product_conformer"]:
                    conf_gen_keys = [key for key in target_species.conformers.keys() if "conf_gen" in key]
                    if target_species and any(desired_key in key for key in conf_gen_keys):
                        already_done = True

                # Check if transition state initial guess conformers have been generated
                elif task_type == "gsm":
                    ts_guess_keys = [key for key in rxn_obj.ts_geom.keys() if "ts_guess" in key]
                    if any(desired_key in key for key in ts_guess_keys):
                        already_done = True

                # Check if reactant/product conformers have been optimized
                elif task_type in ["reactant_optimization", "product_optimization"]:
                    rpopt_keys = [key for key in target_species.conformers.keys() if "rpopt" in key]
                    if target_species and any(desired_key in key for key in rpopt_keys):
                        already_done = True

                # Check if transition state conformers have been optimized
                elif task_type == "transition_state_optimization":
                    tsopt_keys = [key for key in rxn_obj.ts_geom.keys() if "tsopt" in key]
                    if any(desired_key in key for key in tsopt_keys):
                        already_done = True

                # Check if IRC validation has been performed
                elif task_type == "irc_validation":
                    val_keys = [key for key in rxn_obj.ts_geom.keys() if "validated_ts" in key]
                    val_check = any(desired_key in key for key in val_keys)
                    fbar_check = any(desired_key in key for key in rxn_obj.barrier.keys())
                    rbar_check = any(desired_key in key for key in rxn_obj.reverse_barrier.keys())
                    if val_check and fbar_check and rbar_check:
                        already_done = True

                # Execute the fast-forward!
                if already_done:
                    meta["status"] = "terminated_normally"
                    print(f"   * [{rxn_hash}] \tTask '{task_id}' ({desired_key}) already characterized/synced. Fast-forwarding...")

    # 0.2.B. Fast-Forward Global Tasks
    for g_task_id, g_meta in status_tracker.get("global_tasks", {}).items():
        if g_meta["status"] in ["ready", "pending"]:
            task_def = config.global_tasks[g_task_id]
            model = getattr(task_def.config, 'model', None)

            if model:
                all_completed = True
                for rxn_hash in status_tracker["reactions"].keys():
                    rxn_obj = reactions.get(rxn_hash)
                    if not rxn_obj or not hasattr(rxn_obj, 'barrier') or model not in rxn_obj.barrier:
                        all_completed = False
                        break

                if all_completed and len(status_tracker["reactions"]) > 0:
                     g_meta["status"] = "terminated_normally"
                     print(f" * Global Task '{g_task_id}' (Model: {model}) already completed for all reactions. Fast-forwarding...")

    # =================================================================
    # PASS 1: Check Status of Submitted Jobs & Tally Active Jobs
    # =================================================================
    active_jobs = 0
    failed_rxns = {}

    # Registry to prevent redundant submissions for identical species
    active_species_tasks = set()

    # Pre-populate registry with jobs that are ALREADY running from previous loops
    for rxn_hash, rxn_meta in status_tracker["reactions"].items():
        rxn_obj = reactions.get(rxn_hash)
        for task_id, meta in rxn_meta["tasks"].items():
            if meta["status"] in ["submitted", "running"]:
                task_def = config.pipeline_tasks[task_id]
                task_type = getattr(task_def, 'task_type', '')

                # Only track single-species tasks (skip reaction path tasks)
                if task_type in ["reactant_conformer", "product_conformer", "reactant_optimization", "product_optimization"]:
                    is_reactant = "reactant" in task_type
                    species = rxn_obj.reactant if is_reactant else rxn_obj.product
                    if species:
                        active_species_tasks.add((species.hash, task_id))

    # =================================================================
    # PASS 1.1: Check Status of Submitted GLOBAL Jobs
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
    # PASS 1.2: Check Status of Submitted PIPELINE Jobs
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
                    print(f"   * [{rxn_hash}] \tTask '{task_id}' is still running. Come back later...")
                    active_jobs += 1
                else:
                    # Job finished! Time to process it.
                    print(f"   * [{rxn_hash}] \tTask '{task_id}' finished running. Checking output...")
                    if calc.check_output():
                        if calc.scrape_data():
                            calc.cleanup()
                            meta["status"] = "terminated_normally"
                            print(f"   * [{rxn_hash}] \tTask '{task_id}' completed successfully.")

                            # --- Evaluate IRC Outcome ---
                            if task_def.task_type == "irc_validation":
                                lot = getattr(task_def.config, 'lot', 'unknown')
                                software = getattr(task_def.config, 'software', 'unknown')
                                outcome_key = f"{lot}_{software}"

                                # Access the saved label safely
                                outcome = getattr(rxn_obj, 'outcome_label', {}).get(outcome_key)

                                if outcome not in ["intended", "inverse_intended"]:
                                    meta["status"] = "filtered_out"
                                    meta["error_log"] = f"IRC validation failed: Outcome was '{outcome}'."

                                    if rxn_hash not in failed_rxns:
                                        failed_rxns[rxn_hash] = rxn_obj

                                    print(f"   * [{rxn_hash}] \tReaction failed IRC validation (Outcome: '{outcome}'). Routing to failed_rxns.")
                        else:
                            meta["status"] = "finished_with_error"
                            meta["error_log"] = "Data scraping failed."
                            failed_rxns[rxn_hash] = rxn_obj
                            print(f"   * [{rxn_hash}] \tTask '{task_id}' failed during data scraping.")
                    else:
                        meta["status"] = "finished_with_error"
                        meta["error_log"] = "Output validation failed."
                        failed_rxns[rxn_hash] = rxn_obj
                        print(f"   * [{rxn_hash}] \tTask '{task_id}' failed output validation.")
 
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
                    # --- Evaluate Pre-Process Filters ---
                    stage_filter = config.stage_filters.get(task_def.parent_stage)

                    # Only evaluate at the entry points of the pipeline
                    if stage_filter and task_def.task_type in ["reactant_conformer", "product_conformer"]:
                        rxn_obj = reactions[rxn_hash]

                        # Dynamically grab the dictionary from the reaction object (e.g. rxn.barrier)
                        prop_dict = getattr(rxn_obj, stage_filter.type, None)

                        if prop_dict is not None and stage_filter.source in prop_dict:
                            val = prop_dict[stage_filter.source]

                            # If it exceeds the threshold, kill the run cleanly
                            if val > stage_filter.threshold:
                                meta["status"] = "filtered_out"
                                meta["error_log"] = f"Filtered out: {stage_filter.type} ({val:.2f}) > threshold ({stage_filter.threshold})"

                                # Prevent printing the error twice (once for reactant, once for product)
                                if rxn_hash not in failed_rxns:
                                    failed_rxns[rxn_hash] = rxn_obj
                                    print(f"   * [{rxn_hash}] \tFiltered out! {stage_filter.type} ({val:.2f}) > {stage_filter.threshold}")

                                continue # Skip setting this task to 'ready'
                        else:
                            # If EGAT finished but data is missing entirely, fail it out
                            meta["status"] = "finished_with_error"
                            meta["error_log"] = f"Missing filter property: {stage_filter.type} from {stage_filter.source}"

                            if rxn_hash not in failed_rxns:
                                failed_rxns[rxn_hash] = rxn_obj
                                print(f"   * [{rxn_hash}] \tPipeline aborted: Missing ML property '{stage_filter.type}' from '{stage_filter.source}'.")

                            continue # Skip setting this task to 'ready'

                    meta["status"] = "ready"
                    print(f"   * [{rxn_hash}] \tTask '{task_id}' prerequisites met. Marked as READY.")

    # =================================================================
    # PASS 3: Submit New Jobs (Respecting the Limit)
    # =================================================================
    print(f"Current active jobs: {active_jobs} / {max_active_jobs}")

    # Check for idle state: no active jobs and no remaining pending/ready tasks
    pending_or_ready = False
    for rxn_data in status_tracker["reactions"].values():
        if any(meta["status"] in ["pending", "ready"] for meta in rxn_data["tasks"].values()):
            pending_or_ready = True
            break

    if not pending_or_ready:
        for g_meta in status_tracker.get("global_tasks", {}).values():
            if g_meta["status"] in ["pending", "ready"]:
                pending_or_ready = True
                break

    if active_jobs == 0 and not pending_or_ready:
        print("No active or pending tasks remain; YARP has reached a quiescent state.")

    # =================================================================
    # PASS 3.1: Submit New GLOBAL Jobs
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

            job_id = job_manager.submit(script_path, task_def.config)

            if job_id:
                g_meta["status"] = "submitted"
                g_meta["job_id"] = job_id
                g_meta["scratch_dir"] = str(scratch_path)
                active_jobs += 1
            else:
                g_meta["status"] = "finished_with_error"


    # =================================================================
    # PASS 3.2: Submit New PIPELINE Jobs
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
                task_type = getattr(task_def, 'task_type', '')

                # --- Redundancy Blocker ---
                if task_type in ["reactant_conformer", "product_conformer", "reactant_optimization", "product_optimization"]:
                    is_reactant = "reactant" in task_type
                    species = rxn_obj.reactant if is_reactant else rxn_obj.product

                    if species:
                        registry_key = (species.hash, task_id)

                        if registry_key in active_species_tasks:
                            # Silently skip submission! Another identical species is doing the work.
                            # It remains "ready" and will be fast-forwarded in PASS 0 on a future loop.
                            print(f"   [DEBUG] ---> Blocked! Identical species calculation already active.")
                            continue
                        else:
                            # We are the "leader"! Claim this species/task combo.
                            active_species_tasks.add(registry_key)

                calc = get_calculator(task_def, rxn_obj, config.job_manager)

                scratch_path = work_dir / "SCRATCH" / f"{rxn_hash}_{task_id}"
                calc.set_scratch_dir(scratch_path)

                # Pre-flight Data Check
                if not calc.has_prerequisites():
                    meta["status"] = "finished_with_error"
                    meta["error_log"] = "Pre-flight check failed: Missing required data in reaction object."
                    failed_rxns[rxn_hash] = rxn_obj
                    print(f"   * [{rxn_hash}] \tTask '{task_id}' aborted: Pre-flight check failed.")
                    continue

                # Generate and Submit
                print(f"   * [{rxn_hash}] \tSubmitting task '{task_id}'...")
                calc.generate_input()
                script_path = calc.write_submission_script()

                job_id = job_manager.submit(script_path, task_def.config)

                if job_id:
                    meta["status"] = "submitted"
                    meta["job_id"] = job_id
                    meta["scratch_dir"] = str(scratch_path)
                    active_jobs += 1 # Increment our tally!
                else:
                    meta["status"] = "finished_with_error"
                    meta["error_log"] = "Job manager failed to submit job."
                    failed_rxns[rxn_hash] = rxn_obj

    # =================================================================
    # Congrats! You made it through!
    # =================================================================

    # Write all updates back to disk
    save_state(work_dir, status_tracker, reactions, failed_rxns)
    print("YARP progress tracking complete.")
    print(f"======================\n")

def main():
    print("Launching YARP job submitter: yarp-progress")
    print(f"======================")
    parser = argparse.ArgumentParser(description="Initialize YARP with a YAML config.")
    parser.add_argument("work_dir", type=str, help="Path to the YARP working directory")
    args = parser.parse_args()

    absolute_work_dir = Path(args.work_dir).resolve()
    progress_yarp(absolute_work_dir)

if __name__ == "__main__":
    main()