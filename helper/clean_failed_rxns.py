#!/usr/bin/env python3
"""
Remove scratch directories for failed reactions after manual inspection.

Usage:
    python clean_failed_rxns.py <work_dir>            # delete all failed scratch dirs
    python clean_failed_rxns.py <work_dir> --dry-run  # preview without deleting

The script reads failed_status.json produced by yarp-loop and removes the
scratch directories listed there. Run this only after you have inspected the
failed output files and no longer need them.
"""
import argparse
import json
import shutil
from pathlib import Path


def clean_failed_rxns(work_dir: Path, dry_run: bool = False):
    fail_json = work_dir / "failed_status.json"
    if not fail_json.exists():
        print(f"No failed_status.json found in {work_dir}. Nothing to clean.")
        return

    with open(fail_json) as f:
        failed = json.load(f)

    if not failed:
        print("failed_status.json is empty. Nothing to clean.")
        return

    # Collect unique scratch directories across all reactions and tasks
    scratch_dirs = set()
    for rxn_hash, tasks in failed.items():
        for task_id, task_data in tasks.items():
            scratch = task_data.get("scratch_dir", "N/A")
            if scratch != "N/A":
                scratch_dirs.add(Path(scratch))

    if not scratch_dirs:
        print("No scratch directories recorded in failed_status.json.")
        return

    print(f"Found {len(scratch_dirs)} scratch directories across {len(failed)} failed reaction(s).\n")

    deleted = 0
    skipped = 0
    for path in sorted(scratch_dirs):
        if path.exists():
            if dry_run:
                print(f"  [dry-run] would delete: {path}")
            else:
                shutil.rmtree(path)
                print(f"  deleted: {path}")
            deleted += 1
        else:
            print(f"  already gone: {path}")
            skipped += 1

    print()
    if dry_run:
        print(f"Dry run complete. {deleted} directories would be deleted, {skipped} already absent.")
    else:
        print(f"Done. {deleted} directories deleted, {skipped} already absent.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean up scratch directories for failed YARP reactions."
    )
    parser.add_argument(
        "work_dir",
        type=Path,
        help="YARP working directory containing failed_status.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be deleted without actually deleting anything",
    )
    args = parser.parse_args()
    clean_failed_rxns(args.work_dir, dry_run=args.dry_run)
