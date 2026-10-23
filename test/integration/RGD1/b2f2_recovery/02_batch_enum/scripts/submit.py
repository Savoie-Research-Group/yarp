"""Step 02b: run the shards of a run, as a cluster job array or locally.

Cluster (default): reads inputs/cluster_submit_template.sh, fills in its
{{...}} placeholders, writes RUN_DIR/job.sh and prints the command that
submits it. It does not submit anything itself.

Local (--local): runs the shards one after another with this python, logging
each to RUN_DIR/logs/. The pilot takes a minute or two.

Both modes skip shards that already have a DONE marker, so running them again
only redoes unfinished shards.

Step 03 uses this script too, with --driver pointing at its own run_shard.py.

Usage (from 02_batch_enum, in the yarp env):
    python scripts/submit.py RUN_DIR [--template FILE] [--python PATH] [--job-name NAME] [--driver FILE]
    python scripts/submit.py RUN_DIR --local
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RUN_SHARD = HERE / "run_shard.py"
TEMPLATE = HERE.parent / "inputs" / "cluster_submit_template.sh"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="a folder written by make_run.py")
    ap.add_argument("--local", action="store_true", help="run the shards here instead of writing job.sh")
    ap.add_argument("--template", type=Path, default=TEMPLATE)
    ap.add_argument("--python", default=sys.executable, help="python of the yarp env on the cluster")
    ap.add_argument("--job-name", default=None, help="default: rgd1_<run dir name>")
    ap.add_argument("--driver", type=Path, default=RUN_SHARD, help="the shard script to run (default: step 02's)")
    a = ap.parse_args()
    run_dir = a.run_dir.resolve()
    driver = a.driver.resolve()
    shards = sorted(pd.read_csv(run_dir / "rows.csv").shard.unique())
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    if a.local:
        for s in shards:
            t0 = time.time()
            with open(log_dir / f"local_shard_{s:04d}.log", "w") as log:
                subprocess.run([sys.executable, str(driver), "--rows", str(run_dir / "rows.csv"),
                                "--out", str(run_dir / "out"), "--shard", str(s)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f"shard {s}/{shards[-1]} finished in {time.time() - t0:.0f} s")
        print(f"All shards done; next: python scripts/merge.py {a.run_dir}")
        return

    values = {"JOB_NAME": a.job_name or f"rgd1_{run_dir.name}", "N_SHARDS": shards[-1],
              "LOG_DIR": log_dir, "PYTHON": a.python, "RUN_SHARD": driver, "RUN_DIR": run_dir}
    text = a.template.read_text()
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", str(v))
    left = sorted(set(re.findall(r"\{\{\w+\}\}", text)))
    if left:
        sys.exit(f"{a.template} has placeholders submit.py does not fill: {left}")
    job = run_dir / "job.sh"
    job.write_text(text)
    print(f"Wrote {job} from {a.template.name} ({shards[-1]} array tasks).")
    print(f"Check it, then submit:\n    qsub {job}")


if __name__ == "__main__":
    main()
