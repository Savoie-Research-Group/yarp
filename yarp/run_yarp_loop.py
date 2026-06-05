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

    # Background periodic execution loop
    execute_counter = 0

    # Open the output file, and record initial message
    output_file = work_dir / f"yarp_loop.out"
    with open(output_file, "a") as out_f:
        out_f.write(f"Starting YARP loop. Running progress_yarp.py every {args.interval} mins until {end_time}\n")

    # Route stdout AND stderr from yarp-progress to output file
    ACTIVE = True
    while ACTIVE:
        execute_counter += 1
        with open(output_file, "a") as out_f:
            out_f.write(f"[{datetime.now()}] Executing yarp_progress (Run {execute_counter})...\n---*****---\n")
            subprocess.run(
                ["python", str(target_script), str(work_dir)], 
                stdout=out_f,
                stderr=subprocess.STDOUT 
            )

        # Stop if the log contains a shutdown marker
        stop_marker = "No active or pending tasks remain; YARP has reached a quiescent state."
        if stop_marker in output_file.read_text():
            ACTIVE = False
            print("Shutdown marker found in output file; exiting loop.", flush=True)
            break

        # Exit on time limit too
        if datetime.now() >= end_time:
            ACTIVE = False
            print("YARP total runtime reached. Shutting down.", flush=True)
            break

        # Calculate next run time and sleep
        time.sleep(args.interval * 60)


if __name__ == "__main__":
    main()