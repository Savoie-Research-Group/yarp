from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Iterable

from .config import OSGConfig
from .condor import write_submit_file, split_manifest
from .state import OSGTask, event, task_from_dict, upsert_task

WORKER_SCRIPT = "run_egat_osg.sh"


def safe_task_name(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).replace(".", "_")


def write_worker_script(path: Path) -> Path:
    text = r'''#!/bin/bash
# Runs EGAT inside the HTCondor-selected OSG container.

set -uo pipefail

TASK_ID="${1:-unknown}"
ATTEMPT="${2:-0}"
WORKDIR="${_CONDOR_SCRATCH_DIR:-$PWD}"
START_EPOCH="$(date +%s)"

cd "$WORKDIR"

touch forward.log forward.err reverse.log reverse.err forward_out.csv reverse_out.csv

echo "hostname=$(hostname)"
echo "workdir=$PWD"
echo "task_id=$TASK_ID"
echo "attempt=$ATTEMPT"
echo "_CONDOR_SCRATCH_DIR=${_CONDOR_SCRATCH_DIR:-}"
echo "OSG_SITE_NAME=${OSG_SITE_NAME:-}"
echo "GLIDEIN_Site=${GLIDEIN_Site:-}"

write_result() {
    status="$1"
    exit_code="$2"
    category="$3"
    message="$4"
    end_epoch="$(date +%s)"
    runtime="$((end_epoch - START_EPOCH))"
    present_forward=false
    present_reverse=false
    [ -s forward_out.csv ] && present_forward=true
    [ -s reverse_out.csv ] && present_reverse=true
    cat > task_result.json <<JSON
{
  "task_id": "$TASK_ID",
  "reaction_id": null,
  "task_type": "egat_ml_predict",
  "attempt": $ATTEMPT,
  "exit_code": $exit_code,
  "status": "$status",
  "hostname": "$(hostname)",
  "start_time_epoch": $START_EPOCH,
  "end_time_epoch": $end_epoch,
  "runtime_seconds": $runtime,
  "expected_outputs": ["forward_out.csv", "reverse_out.csv"],
  "present_outputs": {
    "forward_out.csv": $present_forward,
    "reverse_out.csv": $present_reverse
  },
  "error_category": "$category",
  "error_message": "$message"
}
JSON
}

if [ ! -s egat_command.txt ]; then
    echo "Missing egat_command.txt" >&2
    write_result "failed" 64 "configuration" "missing_egat_command"
    exit 64
fi

EGAT_COMMAND="$(head -n 1 egat_command.txt)"
if [ -z "$EGAT_COMMAND" ]; then
    echo "EGAT command is empty" >&2
    write_result "failed" 64 "configuration" "empty_egat_command"
    exit 64
fi

if [ ! -f forward_in.csv ] || [ ! -f reverse_in.csv ]; then
    echo "Missing EGAT input CSV" >&2
    write_result "failed" 65 "configuration" "missing_input_csv"
    exit 65
fi

echo "Running forward EGAT command"
bash -lc "$EGAT_COMMAND --input forward_in.csv --output forward_out.csv" > forward.log 2> forward.err
FWD_EXIT=$?

echo "Running reverse EGAT command"
bash -lc "$EGAT_COMMAND --input reverse_in.csv --output reverse_out.csv" > reverse.log 2> reverse.err
REV_EXIT=$?

if [ "$FWD_EXIT" -ne 0 ] || [ "$REV_EXIT" -ne 0 ]; then
    write_result "failed" 20 "chemistry" "egat_command_failed"
    exit 20
fi

if [ ! -s forward_out.csv ] || [ ! -s reverse_out.csv ]; then
    write_result "failed" 21 "missing_output" "missing_egat_output_csv"
    exit 21
fi

write_result "success" 0 "" ""
exit 0
'''
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def _copy_static_worker(workflow_dir: Path, task_dir: Path) -> Path:
    source = workflow_dir / WORKER_SCRIPT
    destination = task_dir / WORKER_SCRIPT
    if source.exists():
        shutil.copy2(source, destination)
        destination.chmod(0o755)
        return destination
    return write_worker_script(destination)


def prepare_egat_task(
    *,
    work_dir: Path,
    state: dict,
    yarp_task_id: str,
    task_def,
    reactions,
    job_config,
    config: OSGConfig,
    workflow_dir: Path,
) -> OSGTask:
    from yarp.reaction.external.calc_factory import get_calculator

    if not config.egat_command:
        raise ValueError("Missing direct EGAT command. Set initialize.job_manager.osg.commands.egat or YARP_OSG_EGAT_COMMAND.")

    task_id = f"global.{yarp_task_id}"
    existing = state.get("tasks", {}).get(task_id)
    if existing and existing.get("status") in {"prepared", "submitted", "idle", "running", "held", "harvested"}:
        return task_from_dict(existing)

    attempt = int(existing.get("attempt", 0)) + 1 if existing else 1
    task_dir = work_dir / config.state_dir_name / "tasks" / "global" / safe_task_name(yarp_task_id) / f"attempt_{attempt}"
    task_dir.mkdir(parents=True, exist_ok=True)

    calc = get_calculator(task_def, reactions, job_config)
    calc.set_scratch_dir(task_dir)
    calc.generate_input()

    _copy_static_worker(workflow_dir, task_dir)
    (task_dir / "egat_command.txt").write_text(config.egat_command.strip() + "\n", encoding="utf-8")

    task = OSGTask(
        task_id=task_id,
        yarp_task_id=yarp_task_id,
        status="prepared",
        attempt=attempt,
        task_dir=str(task_dir),
    )
    upsert_task(state, task)
    event(state, "prepared", task.task_id, task_dir=str(task_dir))
    return task


def write_egat_manifest(state_root: Path, tasks: Iterable[OSGTask]) -> Path:
    manifest = state_root / "egat_jobs.tsv"
    rows = ["task_id task_dir attempt"]
    for task in tasks:
        if task.status == "prepared" and task.task_dir:
            rows.append(f"{task.task_id} {task.task_dir} {task.attempt}")
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return manifest


def prepare_submit_artifacts(
    *,
    state_root: Path,
    config: OSGConfig,
    workflow_dir: Path,
    tasks: Iterable[OSGTask],
) -> tuple[Path, list[Path]]:
    state_root.mkdir(parents=True, exist_ok=True)
    log_dir = state_root / "logs"
    submit_path = state_root / "batch_egat.submit"
    write_submit_file(
        submit_path,
        config=config,
        worker_script_name=WORKER_SCRIPT,
        log_dir=log_dir,
        resources=config.egat_resources,
    )
    manifest = write_egat_manifest(state_root, tasks)
    batches = split_manifest(manifest, config.submit_batch_size, state_root)
    return submit_path, batches


def read_task_result(task_dir: Path) -> dict:
    result_path = task_dir / "task_result.json"
    if not result_path.exists():
        return {
            "status": "failed",
            "exit_code": None,
            "error_category": "missing_output",
            "error_message": "task_result.json missing",
        }
    return json.loads(result_path.read_text(encoding="utf-8"))
