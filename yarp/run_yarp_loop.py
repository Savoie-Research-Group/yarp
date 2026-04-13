import time
import subprocess
import argparse
from datetime import datetime, timedelta
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', '-w', type=str, required=True, help='Path to YARP working directory')
    parser.add_argument('--interval', '-i', type=int, default=5, help='Minutes between runs')
    parser.add_argument('--duration', '-d', type=int, default=60, help='Total runtime in minutes')
    args = parser.parse_args()

    end_time = datetime.now() + timedelta(minutes=args.duration)

    # Check if the directory exists; throw a runtime error if it doesn't
    work_dir = Path(args.working_dir).resolve()
    if not work_dir.is_dir():
        raise RuntimeError(f"Error: The working directory '{work_dir}' does not exist. Please set it up before running this script.")

    # Resolve path to progress_yarp.py
    script_location = Path(__file__).parent.resolve()
    target_script = script_location / "progress_yarp.py"

    print(f"Starting YARP loop. Running progress_yarp.py every {args.interval} mins until {end_time}")

    # Background periodic execution loop
    execute_counter = 0
    while datetime.now() < end_time:
        execute_counter += 1
        print(f"[{datetime.now()}] Executing progress_yarp.py (Run {execute_counter})...")

        # Open the output file, and route stdout AND stderr to it
        output_file = work_dir / f"prog{execute_counter}.out"
        with open(output_file, "w") as out_f:
            subprocess.run(
                ["python", str(target_script), str(work_dir)], 
                stdout=out_f, 
                stderr=subprocess.STDOUT 
            )

        # Calculate next run time and sleep
        time.sleep(args.interval * 60)

    print("YARP total runtime reached. Shutting down.")

if __name__ == "__main__":
    main()