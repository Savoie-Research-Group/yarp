#!/usr/bin/env python3
"""
Put failed reactions back into a YARP run so their failed step is tried again.

Usage (installed as `yarp-restart`):
    yarp-restart <work_dir> --all                 # restart every failed reaction
    yarp-restart <work_dir> --hash H1 H2          # restart these reactions only
    yarp-restart <work_dir> --all --dry-run       # show what would be done, change nothing
    yarp-restart <work_dir> --all --new-seed      # seeded conformer retries use a new seed
    yarp-restart <work_dir> --all --include-filtered   # also reactions removed by a filter

Then start (or keep running) yarp-loop as usual.

For each chosen reaction, only the step that failed is retried; every step that
finished keeps its results and is fast-forwarded by the next progress pass:
  - the reaction object moves from failed_rxns.pkl back into the run's pickle,
    and its entries in failed_status.json and its `failed` mark in STATUS.json go;
  - each failed task's partial results (if its scrape stopped part-way) are
    removed, so the fast-forward cannot mistake them for a finished task;
  - the failed attempt's scratch folder is renamed to `<folder>.attempt<N>`
    (never overwritten -- files inside may not be writable), and the attempt's
    job id, error and folder are recorded in the task's `attempts` list;
  - the task is set back to pending, and the next pass submits it.

A reaction with a task whose job is still in the scheduler's queue is left
alone (restart it once that job has ended). A task whose job has ended but was
never checked stays "submitted"; the next pass checks its output as usual.

Reactions removed by a filter (IRC 'unintended' with divert_unintended, stage
thresholds) did not fail at random and are only restarted with
--include-filtered. Reactions that failed before this run kept failed reactions'
history in STATUS.json cannot be restarted with this tool.

Backups of STATUS.json, the run's pickle, failed_rxns.pkl and failed_status.json
are written to restart_backups/<timestamp>/ before anything changes. The tool
holds the run lock, so it never runs during a progress pass.
"""
import argparse
import copy
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

from yarp.progress_yarp import load_state, is_failed, _matches_lot
from yarp.util.input import InputParser
from yarp.reaction.external.job_manager import get_job_manager
from yarp.util.run_lock import run_lock, RunLockedError

# Statuses a restart resets. "filtered_out" only with --include-filtered.
FAILED = "finished_with_error"
FILTERED = "filtered_out"
STILL_QUEUED = ("submitted", "running")

# Conformer software whose results depend on a seed (CREST has none).
SEEDED_CONF_SOFTWARE = ("rdkit", "auto3d")


def clear_task_results(rxn, task_type, task_config) -> list:
    """
    Remove whatever results `task_type` (at its level of theory) stored on the
    reaction, so the progress fast-forward will not count it as done.
    Returns the removed keys, for the report.
    """
    lot = getattr(task_config, "gsm_lot" if task_type == "ts_guess" else "lot", None)
    desired = f"{lot}_{getattr(task_config, 'software', None)}"
    species = rxn.reactant if "reactant" in task_type else rxn.product
    removed = []

    def drop(container, test, label):
        for key in [k for k in list(container) if test(k) and _matches_lot(k, desired)]:
            del container[key]
            removed.append(f"{label}:{key}")

    if task_type in ("reactant_pre_opt", "product_pre_opt"):
        drop(species.conformers, lambda k: k.startswith("preopt"), "conformers")
    elif task_type in ("reactant_conformer", "product_conformer"):
        drop(species.conformers, lambda k: k.startswith("conf_gen"), "conformers")
    elif task_type in ("reactant_optimization", "product_optimization"):
        drop(species.conformers, lambda k: k.startswith("rpopt"), "conformers")
    elif task_type == "ts_guess":
        drop(rxn.ts_geom, lambda k: k.startswith("ts_guess"), "ts_geom")
        for side in (rxn.reactant, rxn.product):
            drop(side.conformers, lambda k: k.startswith("guess_conf"), "conformers")
    elif task_type == "transition_state_optimization":
        drop(rxn.ts_geom, lambda k: "_tsopt_" in k, "ts_geom")
    elif task_type == "irc_validation":
        for name in ("barrier", "reverse_barrier", "outcome_label"):
            values = getattr(rxn, name, None)
            if isinstance(values, dict):
                drop(values, lambda k: True, name)
    return removed


def _atomic_write(path: Path, write):
    """Write via a temporary file and rename, so a crash never leaves half a file."""
    tmp = path.with_name(path.name + ".restart_tmp")
    write(tmp)
    os.replace(tmp, path)


def restart_failed(work_dir, hashes=None, all_failed=False, include_filtered=False,
                   new_seed=False, dry_run=False, config=None, job_manager=None) -> dict:
    """
    Restart failed reactions in `work_dir` (see the module docstring).
    Returns {"restarted": [...], "skipped": {hash: reason}}.
    """
    work_dir = Path(work_dir)
    if not all_failed and not hashes:
        raise ValueError("Choose the reactions to restart: --all or --hash H [H ...].")

    with run_lock(work_dir, "yarp-restart"):
        status, reactions = load_state(work_dir)
        config = config or InputParser(status["input_config"])
        job_manager = job_manager or get_job_manager(config.job_manager.scheduler)

        failed_pkl = work_dir / "failed_rxns.pkl"
        failed_json = work_dir / "failed_status.json"
        failed_rxns = pickle.load(open(failed_pkl, "rb")) if failed_pkl.exists() else {}
        failed_status = json.load(open(failed_json)) if failed_json.exists() else {}

        marked = [h for h, m in status["reactions"].items() if is_failed(m)]
        if all_failed:
            chosen = marked + [h for h in failed_rxns if h not in status["reactions"]]
        else:
            chosen = list(hashes)

        restarted, skipped, plans = [], {}, []
        for h in chosen:
            entry = status["reactions"].get(h)
            if entry is None:
                skipped[h] = ("no STATUS.json history (failed before failed reactions kept their history); "
                              "cannot be restarted with this tool" if h in failed_rxns else "not in this run")
                continue
            if not is_failed(entry):
                skipped[h] = "not a failed reaction"
                continue
            if h not in failed_rxns:
                skipped[h] = "marked failed but missing from failed_rxns.pkl"
                continue

            retry_states = (FAILED, FILTERED) if include_filtered else (FAILED,)
            to_reset = [tid for tid, m in entry["tasks"].items() if m["status"] in retry_states]
            if not to_reset:
                filtered = [tid for tid, m in entry["tasks"].items() if m["status"] == FILTERED]
                skipped[h] = (f"removed by a filter ({', '.join(filtered)}); use --include-filtered"
                              if filtered else "no failed task to retry")
                continue

            running = [(tid, m["job_id"]) for tid, m in entry["tasks"].items()
                       if m["status"] in STILL_QUEUED and m.get("job_id")
                       and job_manager.is_running(m["job_id"])]
            if running:
                skipped[h] = "job still running: " + ", ".join(f"{tid} (job {j})" for tid, j in running)
                continue
            plans.append((h, to_reset))

        if not dry_run and plans:
            # A fresh folder per restart; a suffix keeps two restarts in the same second apart.
            stamp = time.strftime("%Y%m%d-%H%M%S")
            backup = work_dir / "restart_backups" / stamp
            n = 1
            while backup.exists():
                backup = work_dir / "restart_backups" / f"{stamp}-{n}"
                n += 1
            backup.mkdir(parents=True)
            for f in (Path(status["status_output_file"]), Path(status["reaction_output_file"]),
                      Path("failed_rxns.pkl"), Path("failed_status.json")):
                if (work_dir / f).exists():
                    shutil.copy2(work_dir / f, backup / f.name)

        for h, to_reset in plans:
            entry = status["reactions"][h]
            rxn = failed_rxns[h]
            report = {"hash": h, "tasks": []}
            for tid in to_reset:
                meta = entry["tasks"][tid]
                task_def = config.pipeline_tasks[tid]
                attempt = meta.get("attempt", 1)
                old_dir = Path(meta["scratch_dir"]) if meta.get("scratch_dir") else None
                kept_dir = old_dir.with_name(f"{old_dir.name}.attempt{attempt}") if old_dir else None
                seed = None
                if new_seed and task_def.task_type in ("reactant_conformer", "product_conformer") \
                        and getattr(task_def.config, "software", None) in SEEDED_CONF_SOFTWARE \
                        and getattr(task_def.config, "seed", None) is not None:
                    seed = task_def.config.seed + attempt
                # In a dry run, report what would be cleared using a copy of the reaction.
                removed = clear_task_results(copy.deepcopy(rxn) if dry_run else rxn,
                                             task_def.task_type, task_def.config)
                report["tasks"].append({"task": tid, "attempt": attempt + 1, "error": meta.get("error_log"),
                                        "scratch_kept_as": str(kept_dir) if kept_dir and old_dir.exists() else None,
                                        "seed": seed, "removed_results": removed})
                if dry_run:
                    continue
                if old_dir is not None and old_dir.exists():
                    old_dir.rename(kept_dir)
                meta.setdefault("attempts", []).append({
                    "attempt": attempt, "status": meta["status"], "job_id": meta.get("job_id"),
                    "error_log": meta.get("error_log"),
                    "scratch_dir": str(kept_dir) if kept_dir and kept_dir.exists() else meta.get("scratch_dir")})
                meta.update(status="pending", job_id=None, scratch_dir=None, attempt=attempt + 1)
                meta.pop("error_log", None)
                if seed is not None:
                    meta["seed"] = seed
            if not dry_run:
                entry.pop("failed", None)
                reactions[h] = rxn
                del failed_rxns[h]
                failed_status.pop(h, None)
            restarted.append(report)

        if not dry_run and plans:
            _atomic_write(work_dir / status["status_output_file"],
                          lambda p: p.write_text(json.dumps(status, indent=4)))
            _atomic_write(work_dir / status["reaction_output_file"],
                          lambda p: p.write_bytes(pickle.dumps(reactions)))
            _atomic_write(failed_pkl, lambda p: p.write_bytes(pickle.dumps(failed_rxns)))
            _atomic_write(failed_json, lambda p: p.write_text(json.dumps(failed_status, indent=4)))

    return {"restarted": restarted, "skipped": skipped, "dry_run": dry_run}


def _print_report(result):
    verb = "Would restart" if result["dry_run"] else "Restarted"
    print(f"{verb} {len(result['restarted'])} reaction(s).")
    for r in result["restarted"]:
        for t in r["tasks"]:
            extra = []
            if t["scratch_kept_as"]:
                extra.append(f"old scratch -> {t['scratch_kept_as']}")
            if t["seed"] is not None:
                extra.append(f"seed {t['seed']}")
            if t["removed_results"]:
                extra.append(f"cleared {len(t['removed_results'])} partial result(s)")
            print(f"  [{r['hash']}] {t['task']}: attempt {t['attempt']} (last error: {t['error']})"
                  + (f"; {'; '.join(extra)}" if extra else ""))
    if result["skipped"]:
        print(f"Skipped {len(result['skipped'])} reaction(s):")
        for h, why in result["skipped"].items():
            print(f"  [{h}] {why}")
    if result["restarted"] and not result["dry_run"]:
        print("Start yarp-loop (or let a running loop continue) to run the restarted steps.")


def cli():
    parser = argparse.ArgumentParser(description="Restart failed reactions in a YARP run.")
    parser.add_argument("work_dir", type=Path, help="YARP run directory")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--all", action="store_true", help="restart every failed reaction")
    which.add_argument("--hash", nargs="+", metavar="H", help="restart only these reaction hashes")
    parser.add_argument("--include-filtered", action="store_true",
                        help="also restart reactions removed by a filter (IRC outcome, stage threshold)")
    parser.add_argument("--new-seed", action="store_true",
                        help="seeded conformer steps (rdkit, auto3d) retry with seed + attempt number")
    parser.add_argument("--dry-run", action="store_true", help="show what would be done; change nothing")
    args = parser.parse_args()
    try:
        result = restart_failed(args.work_dir, hashes=args.hash, all_failed=args.all,
                                include_filtered=args.include_filtered, new_seed=args.new_seed,
                                dry_run=args.dry_run)
    except RunLockedError as e:
        sys.exit(f"Not restarting: {e}")
    _print_report(result)


if __name__ == "__main__":
    cli()
