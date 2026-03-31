import sys
import yaml
import pickle
import json
from pathlib import Path
from yarp.util.input import InputParser
from yarp.reaction.generate_rxns import generate_rxns

def initialize_from_dict(file_dict):
    """
    Core logic separated from file I/O for testing purposes.
    Generates the reactions and the status tracker dictionary.
    """
    inp = InputParser(file_dict)

    if inp.job_manager.container == "apptainer":
        sif_path = Path(inp.job_manager.sif_location)
        if not sif_path.exists() or not sif_path.is_dir():
            raise FileNotFoundError(f"Fatal: The specified Apptainer SIF location does not exist or is not a directory: {sif_path}")
    
    print("Let's generate some initial reaction objects...")
    raw_reactions = generate_rxns(inp)
    # FORCE KEYS TO STRINGS to ensure consistency between Pickle and JSON
    reactions = {str(k): v for k, v in raw_reactions.items()}
    
    # Initialize the STATUS dictionary
    status_tracker = {
        "pipeline_tasks": list(inp.pipeline_tasks.keys()),
        "global_tasks_list": list(inp.global_tasks.keys()),
        "global_tasks": {},
        "input_config": file_dict,
        "reactions": {}
    }

    # Process global tasks
    for task_id, task_def in inp.global_tasks.items():
        status_tracker["global_tasks"][task_id] = {
            "status": "ready" if not task_def.depends_on else "pending", 
            "job_id": None,          
            "scratch_dir": None      
        }

    # Process pipeline tasks
    for rxn_hash in reactions.keys():
        tasks_status = {}
        for task_id, task_def in inp.pipeline_tasks.items():
            tasks_status[task_id] = {
                # 'pending' if it waits for other tasks in THIS pipeline run
                # 'ready' if it has no dependencies (or dependencies are already met via external data)
                "status": "ready" if not task_def.depends_on else "pending", 
                "job_id": None,          
                "scratch_dir": None      
            }
            
        status_tracker["reactions"][rxn_hash] = {
            "tasks": tasks_status
        }
        
    return reactions, status_tracker

def save_state(work_dir, reactions, status_tracker):
    """Writes the generated state to disk."""
    with open(work_dir / "YARP_RXNS.pkl", "wb") as f:
        pickle.dump(reactions, f)
        
    with open(work_dir / "STATUS.json", "w") as f:
        json.dump(status_tracker, f, indent=4)
        
    print(f"Initialized {len(reactions)} reactions and saved to disk!")

def main(yaml_file):
    with open(yaml_file, 'r') as f:
        file_dict = yaml.safe_load(f)

    # Figure out a way to print the current version/commit hash
    print("First off, here's the input file you provided:")
    print("=====================================")
    print(yaml.dump(file_dict))
    print("=====================================")
        
    work_dir = Path.cwd()
    reactions, status_tracker = initialize_from_dict(file_dict)
    save_state(work_dir, reactions, status_tracker)

    print("If you want to characterize these reactions, you'll need to go run 'progress_yarp' next.")
    print("See ya, bye!")

if __name__ == "__main__":
    print(f"""Welcome to
               __   __ _    ____  ____  
               \ \ / // \  |  _ \|  _ \ 
                \ V // _ \ | |_) | |_) |
                 | |/ ___ \|  _ <|  __/ 
                 |_/_/   \_\_| \_\_|
                        // Yet Another Reaction Program
    """)

    main(sys.argv[1])