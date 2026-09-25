#!/usr/bin/env python3
"""Run yarp-progress in initialized SMILES batch work directories."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from .yarp_batch_common import (
    discover_initialized_workdirs,
    load_batch_config,
    read_json,
    resolve_config_path,
    status_is_quiescent,
    write_json,
)


RETRY_FILE = "yarp_batch_retries.json"


def repair_failed(work_dir: Path, status_name: str, max_retries: int) -> tuple[int, int]:
    if max_retries <= 0:
        return 0, 0

    status_path = work_dir / status_name
    if not status_path.exists():
        return 0, 0

    status = read_json(status_path)
    retries_path = work_dir / RETRY_FILE
    retries = read_json(retries_path) if retries_path.exists() else {}

    from yarp.util.input import InputParser

    parser = InputParser(status["input_config"])
    restored = 0
    exhausted = 0

    for task_id, task_status in status.get("global_tasks", {}).items():
        if task_status.get("status") != "finished_with_error":
            continue
        retry_key = f"GLOBAL::{task_id}"
        attempts = int(retries.get(retry_key, 0))
        if attempts >= max_retries:
            exhausted += 1
            continue
        task_def = parser.global_tasks[task_id]
        retries[retry_key] = attempts + 1
        task_status.clear()
        task_status.update(
            {
                "status": "ready" if not task_def.depends_on else "pending",
                "job_id": None,
                "scratch_dir": None,
                "retry_attempt": retries[retry_key],
            }
        )
        restored += 1

    failed_rxns_path = work_dir / "failed_rxns.pkl"
    if not failed_rxns_path.exists():
        write_json(status_path, status)
        write_json(retries_path, retries)
        return restored, exhausted

    rxn_file = status.get("reaction_output_file")
    if not rxn_file:
        write_json(status_path, status)
        write_json(retries_path, retries)
        return restored, exhausted

    active_rxns_path = work_dir / rxn_file
    with open(active_rxns_path, "rb") as handle:
        active_rxns = pickle.load(handle)
    with open(failed_rxns_path, "rb") as handle:
        failed_rxns = pickle.load(handle)

    failed_status_path = work_dir / "failed_status.json"
    failed_status = read_json(failed_status_path) if failed_status_path.exists() else {}
    remaining_failed_rxns = {}
    remaining_failed_status = {}

    for rxn_hash, rxn_obj in failed_rxns.items():
        attempts = int(retries.get(rxn_hash, 0))
        if attempts >= max_retries:
            exhausted += 1
            remaining_failed_rxns[rxn_hash] = rxn_obj
            if rxn_hash in failed_status:
                remaining_failed_status[rxn_hash] = failed_status[rxn_hash]
            continue

        retries[rxn_hash] = attempts + 1
        active_rxns[rxn_hash] = rxn_obj
        status.setdefault("reactions", {})[rxn_hash] = {
            "tasks": {
                task_id: {
                    "status": "ready" if not task_def.depends_on else "pending",
                    "job_id": None,
                    "scratch_dir": None,
                    "retry_attempt": retries[rxn_hash],
                }
                for task_id, task_def in parser.pipeline_tasks.items()
            }
        }
        restored += 1

    with open(active_rxns_path, "wb") as handle:
        pickle.dump(active_rxns, handle)
    write_json(status_path, status)
    write_json(retries_path, retries)

    if remaining_failed_rxns:
        with open(failed_rxns_path, "wb") as handle:
            pickle.dump(remaining_failed_rxns, handle)
        write_json(failed_status_path, remaining_failed_status)
    else:
        failed_rxns_path.unlink()
        if failed_status_path.exists():
            failed_status_path.unlink()

    return restored, exhausted


def run_yarp_progress(work_dir: Path, progress_log_name: str, yarp_progress: str, status_name: str) -> tuple[bool, int]:
    with open(work_dir / progress_log_name, "a") as log:
        log.write(f"\n=== {datetime.now()} yarp-progress in {work_dir} ===\n")
        log.flush()
        result = subprocess.run(
            [yarp_progress, "."],
            cwd=work_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(f"=== exit code: {result.returncode} ===\n")

    done = result.returncode == 0 and status_is_quiescent(work_dir / status_name)
    return done, result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="High-throughput yarp-progress over SMILES batch directories.")
    parser.add_argument(
        "batch_config",
        nargs="?",
        type=Path,
        default=Path("batch.yaml"),
        help="Batch YAML used by yarp-init-batch.",
    )
    parser.add_argument("--root", type=Path, help="Override root directory to scan for STATUS files.")
    parser.add_argument("--batch", type=str, help="Only scan one batch directory, e.g. batch_001.")
    parser.add_argument("--interval", "-i", type=int, default=5, help="Minutes between cycles.")
    parser.add_argument("--duration", "-d", type=int, default=60, help="Total runtime in minutes.")
    parser.add_argument("--repair-failures", action="store_true", help="Restore failed reactions before progressing.")
    parser.add_argument("--max-retries", type=int, default=0, help="Retries per failed reaction/global task.")
    parser.add_argument("--include-quiescent", action="store_true", help="Keep checking completed directories.")
    parser.add_argument("--limit", type=int, help="Limit number of work directories per cycle.")
    args = parser.parse_args()

    config, config_dir = load_batch_config(args.batch_config)
    root = resolve_config_path(args.root or config["output_dir"], config_dir)
    if args.batch:
        root = root / args.batch

    status_name = config["status_name"]
    progress_log_name = config["progress_log_name"]
    batch_log_name = config["batch_log_name"]
    yarp_progress = config["yarp_progress"]

    end_time = datetime.now() + timedelta(minutes=args.duration)
    root.mkdir(parents=True, exist_ok=True)
    batch_log_path = root / batch_log_name
    completed: set[Path] = set()
    failures = 0
    cycle = 0

    with open(batch_log_path, "a") as batch_log:
        batch_log.write(
            f"\nStarting yarp-progress-batch at {datetime.now()}; "
            f"root={root}; interval={args.interval}; end_time={end_time}\n"
        )

    while datetime.now() < end_time:
        cycle += 1
        work_dirs = discover_initialized_workdirs(root, status_name)
        if args.limit:
            work_dirs = work_dirs[: args.limit]

        active_dirs = []
        for work_dir in work_dirs:
            if not args.include_quiescent and work_dir in completed:
                continue

            if args.repair_failures:
                try:
                    restored, exhausted = repair_failed(work_dir, status_name, args.max_retries)
                    if restored or exhausted:
                        with open(batch_log_path, "a") as batch_log:
                            batch_log.write(
                                f"  [repair] {work_dir}: restored={restored} exhausted={exhausted}\n"
                            )
                except Exception as exc:
                    failures += 1
                    with open(batch_log_path, "a") as batch_log:
                        batch_log.write(f"  [repair-fail] {work_dir}: {exc}\n")

            if not args.include_quiescent and status_is_quiescent(work_dir / status_name):
                completed.add(work_dir)
                continue
            active_dirs.append(work_dir)

        with open(batch_log_path, "a") as batch_log:
            batch_log.write(
                f"[{datetime.now()}] cycle={cycle} discovered={len(work_dirs)} "
                f"active={len(active_dirs)} completed={len(completed)}\n"
            )

        if not active_dirs:
            with open(batch_log_path, "a") as batch_log:
                batch_log.write("All initialized work directories are quiescent; exiting.\n")
            break

        for work_dir in active_dirs:
            done, exit_code = run_yarp_progress(work_dir, progress_log_name, yarp_progress, status_name)
            if exit_code != 0:
                failures += 1
            if done:
                completed.add(work_dir)
            with open(batch_log_path, "a") as batch_log:
                state = "done" if done else "active"
                batch_log.write(f"  [{state}:{exit_code}] {work_dir}\n")

        if datetime.now() >= end_time:
            break
        time.sleep(args.interval * 60)

    with open(batch_log_path, "a") as batch_log:
        batch_log.write(f"Stopped yarp-progress-batch at {datetime.now()}; failures={failures}\n")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
