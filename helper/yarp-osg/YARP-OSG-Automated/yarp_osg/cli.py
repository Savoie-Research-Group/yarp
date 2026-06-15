from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import __version__
from .condor import manifest_rows, query_job, submit as condor_submit
from .config import load_osg_config, state_root
from .egat import prepare_egat_task, prepare_submit_artifacts, read_task_result
from .state import (
    ACTIVE_STATUSES,
    OSGTask,
    event,
    load_state,
    retry_or_quarantine,
    save_state,
    task_from_dict,
    upsert_task,
)
from .yarp_bridge import egat_ready_tasks, is_initialized_yarp_dir, load_input_parser, load_status, load_yarp_state, save_status, save_yarp_state

COMMANDS = {
    "run",
    "plan",
    "prepare-egat",
    "submit",
    "status",
    "harvest",
    "retry",
    "advance",
    "cleanup",
    "record-submit",
}


def resolve_work_dir(value: str | None) -> Path:
    return Path(value or ".").resolve()


def discover_work_dirs(path: Path) -> list[Path]:
    """Return one initialized YARP workdir, or initialized direct children."""
    if is_initialized_yarp_dir(path):
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"YARP workdir or batch directory does not exist: {path}")
    children = sorted(
        child
        for child in path.iterdir()
        if child.is_dir() and is_initialized_yarp_dir(child)
    )
    if children:
        return children
    raise FileNotFoundError(
        f"No initialized YARP workdir found at {path} or in its direct child directories"
    )


def run_for_discovered_work_dirs(args, handler, label: str) -> int | None:
    root = resolve_work_dir(args.work_dir)
    work_dirs = discover_work_dirs(root)
    if len(work_dirs) == 1 and work_dirs[0] == root:
        return None

    print("==============================================")
    print(f"YARP OSG batch {label}")
    print("==============================================")
    print(f"Batch root: {root}")
    print(f"Initialized workdirs: {len(work_dirs)}")
    print("")

    rc = 0
    for index, work_dir in enumerate(work_dirs, start=1):
        print("----------------------------------------------")
        print(f"[{index}/{len(work_dirs)}] {work_dir}")
        print("----------------------------------------------")
        child_args = argparse.Namespace(**vars(args))
        child_args.work_dir = str(work_dir)
        child_rc = handler(child_args)
        rc = rc or child_rc
        print("")
    return rc


def workflow_dir_from_args(value: str | None) -> Path:
    if value:
        return Path(value).resolve()
    return Path(__file__).resolve().parents[1]


def state_paths(work_dir: Path, config):
    root = state_root(work_dir, config)
    return root, root / "state.json"


def plan(args) -> int:
    batch_rc = run_for_discovered_work_dirs(args, plan, "plan")
    if batch_rc is not None:
        return batch_rc

    work_dir = resolve_work_dir(args.work_dir)
    _, status = load_status(work_dir)
    parser = load_input_parser(status["input_config"])
    ready = egat_ready_tasks(status, parser)
    config = load_osg_config(status.get("input_config", {}))
    print(f"YARP OSG version: {__version__}")
    print(f"Work dir: {work_dir}")
    print(f"State dir: {work_dir / config.state_dir_name}")
    if config.local_sif_dir:
        print(f"Local SIF dir: {config.local_sif_dir}")
    if config.egat_local_sif and Path(config.egat_local_sif).exists():
        print(f"Local EGAT SIF: {config.egat_local_sif}")
    print(f"EGAT command: {config.egat_command}")
    print(f"EGAT ready tasks: {len(ready)}")
    for task_id in ready:
        print(f"  - {task_id}")
    if not config.egat_container:
        print("Missing EGAT OSG container: set initialize.job_manager.osg.osdf_namespace, initialize.job_manager.osg.containers.egat, YARP_OSG_OSDF_NAMESPACE, or YARP_OSG_EGAT_CONTAINER")
    return 0


def prepare_egat(args) -> int:
    batch_rc = run_for_discovered_work_dirs(args, prepare_egat, "prepare-egat")
    if batch_rc is not None:
        return batch_rc

    work_dir = resolve_work_dir(args.work_dir)
    workflow_dir = workflow_dir_from_args(args.workflow_dir)
    _, status, reactions = load_yarp_state(work_dir)
    parser = load_input_parser(status["input_config"])
    config = load_osg_config(status.get("input_config", {}))
    root, state_path = state_paths(work_dir, config)
    state = load_state(state_path)

    prepared = []
    for task_id in egat_ready_tasks(status, parser):
        task = prepare_egat_task(
            work_dir=work_dir,
            state=state,
            yarp_task_id=task_id,
            task_def=parser.global_tasks[task_id],
            reactions=reactions,
            job_config=parser.job_manager,
            config=config,
            workflow_dir=workflow_dir,
        )
        prepared.append(task)

    submit_path, batches = prepare_submit_artifacts(
        state_root=root,
        config=config,
        workflow_dir=workflow_dir,
        tasks=prepared,
    )
    for task in prepared:
        data = state["tasks"][task.task_id]
        data["manifest"] = str(root / "egat_jobs.tsv")
        data["updated_at"] = data.get("updated_at")
    event(state, "submit_artifacts_prepared", details={"submit": str(submit_path), "batches": [str(p) for p in batches]})
    save_state(state_path, state)
    print(f"Prepared {len(prepared)} EGAT task(s)")
    print(f"Submit file: {submit_path}")
    print(f"Batch files: {len(batches)}")
    return 0


def record_submit(args) -> int:
    work_dir = resolve_work_dir(args.work_dir)
    _, status = load_status(work_dir)
    config = load_osg_config(status.get("input_config", {}))
    _, state_path = state_paths(work_dir, config)
    state = load_state(state_path)

    rows = manifest_rows(Path(args.batch_file))
    for proc_id, row in enumerate(rows):
        task_id = row["task_id"]
        task = task_from_dict(state["tasks"][task_id])
        task.status = "submitted"
        task.cluster_id = str(args.cluster_id)
        task.proc_id = str(proc_id)
        task.condor_id = f"{args.cluster_id}.{proc_id}"
        upsert_task(state, task)
        yarp_meta = status.get("global_tasks", {}).get(task.yarp_task_id)
        if yarp_meta is not None:
            yarp_meta["status"] = "submitted"
            yarp_meta["job_id"] = task.condor_id
            yarp_meta["scratch_dir"] = task.task_dir
        event(state, "submitted", task_id, condor_id=task.condor_id)

    save_status(work_dir, status)
    save_state(state_path, state)
    print(f"Recorded cluster {args.cluster_id} for {len(rows)} task(s)")
    return 0


def status_cmd(args) -> int:
    batch_rc = run_for_discovered_work_dirs(args, status_cmd, "status")
    if batch_rc is not None:
        return batch_rc

    work_dir = resolve_work_dir(args.work_dir)
    _, status = load_status(work_dir)
    config = load_osg_config(status.get("input_config", {}))
    root, state_path = state_paths(work_dir, config)
    state = load_state(state_path)
    counts: dict[str, int] = {}
    for data in state.get("tasks", {}).values():
        counts[data.get("status", "unknown")] = counts.get(data.get("status", "unknown"), 0) + 1
    print(f"Work dir: {work_dir}")
    print(f"State: {state_path}")
    if not state.get("tasks"):
        print("No OSG tasks recorded")
    else:
        for key in sorted(counts):
            print(f"{key}: {counts[key]}")
    if root.exists():
        print(f"Logs: {root / 'logs'}")
    return 0


def condor_tool_available(name: str) -> bool:
    return shutil.which(name) is not None


def submit_pending(args) -> int:
    batch_rc = run_for_discovered_work_dirs(args, submit_pending, "submit")
    if batch_rc is not None:
        return batch_rc

    work_dir = resolve_work_dir(args.work_dir)
    _, status = load_status(work_dir)
    config = load_osg_config(status.get("input_config", {}))
    root, state_path = state_paths(work_dir, config)
    state = load_state(state_path)
    submit_file = root / "batch_egat.submit"
    submitted_dir = root / "submitted_batches"
    submitted_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        path
        for path in sorted(root.glob("job_*.tsv"))
        if not (submitted_dir / path.name).exists()
    ]
    if not pending:
        print("No pending OSG batch files to submit")
        return 0
    if not submit_file.exists():
        print(f"Missing submit file: {submit_file}")
        return 1
    if not condor_tool_available("condor_submit"):
        print("condor_submit is not available; prepared batches are waiting for submission")
        return 0

    submitted = 0
    for batch_file in pending:
        cluster_id = condor_submit(submit_file, batch_file)
        record_args = argparse.Namespace(
            work_dir=str(work_dir),
            cluster_id=cluster_id,
            batch_file=str(batch_file),
        )
        record_submit(record_args)
        shutil.move(str(batch_file), submitted_dir / batch_file.name)
        state = load_state(state_path)
        event(state, "batch_submitted", details={"batch": str(batch_file), "cluster_id": cluster_id})
        save_state(state_path, state)
        submitted += 1

    print(f"Submitted {submitted} OSG batch file(s)")
    return 0


def _refresh_condor_states(state: dict, runner=subprocess.run) -> None:
    for task_id, data in list(state.get("tasks", {}).items()):
        task = task_from_dict(data)
        if task.status not in {"submitted", "idle", "running", "held"} or not task.condor_id:
            continue
        try:
            condor_status = query_job(task.condor_id, runner=runner)
        except FileNotFoundError:
            print("condor_q is not available; leaving submitted task states unchanged")
            return
        if condor_status.state in {"idle", "running", "held"}:
            task.status = condor_status.state
            task.error_category = condor_status.error_category
            task.error_message = condor_status.hold_reason
            upsert_task(state, task)
            event(state, f"condor_{condor_status.state}", task_id, category=condor_status.error_category)
        elif condor_status.state == "complete":
            task.status = "complete"
            upsert_task(state, task)
            event(state, "condor_complete", task_id)
        elif condor_status.state == "failed":
            task.status = "failed"
            task.error_category = condor_status.error_category
            task.error_message = f"condor_exit_{condor_status.exit_code}"
            upsert_task(state, task)
            event(state, "condor_failed", task_id, category=task.error_category)


def harvest(args) -> int:
    batch_rc = run_for_discovered_work_dirs(args, harvest, "harvest")
    if batch_rc is not None:
        return batch_rc

    work_dir = resolve_work_dir(args.work_dir)
    _, status, reactions = load_yarp_state(work_dir)
    parser = load_input_parser(status["input_config"])
    config = load_osg_config(status.get("input_config", {}))
    _, state_path = state_paths(work_dir, config)
    state = load_state(state_path)

    if not args.skip_condor:
        _refresh_condor_states(state)

    harvested = 0
    eligible_statuses = {"complete"}
    if args.skip_condor:
        eligible_statuses.add("submitted")

    for task_id, data in list(state.get("tasks", {}).items()):
        task = task_from_dict(data)
        if task.status not in eligible_statuses or not task.task_dir:
            continue
        result = read_task_result(Path(task.task_dir))
        if result.get("status") != "success":
            task.status = "failed"
            task.error_category = result.get("error_category") or "missing_output"
            task.error_message = result.get("error_message")
            upsert_task(state, task)
            event(state, "harvest_failed", task_id, category=task.error_category)
            continue

        from yarp.reaction.external.calc_factory import get_calculator

        task_def = parser.global_tasks[task.yarp_task_id]
        calc = get_calculator(task_def, reactions, parser.job_manager)
        calc.set_scratch_dir(Path(task.task_dir))
        if calc.check_output():
            calc.scrape_data()
            task.status = "harvested"
            upsert_task(state, task)
            meta = status.get("global_tasks", {}).get(task.yarp_task_id)
            if meta is not None:
                meta["status"] = "terminated_normally"
                meta["job_id"] = task.condor_id
                meta["scratch_dir"] = task.task_dir
            event(state, "harvested", task_id)
            harvested += 1
        else:
            task.status = "failed"
            task.error_category = "missing_output"
            task.error_message = "YARP EGAT output validation failed"
            upsert_task(state, task)
            event(state, "harvest_validation_failed", task_id)

    save_yarp_state(work_dir, status, reactions)
    save_state(state_path, state)
    print(f"Harvested {harvested} EGAT task(s)")
    return 0


def retry(args) -> int:
    batch_rc = run_for_discovered_work_dirs(args, retry, "retry")
    if batch_rc is not None:
        return batch_rc

    work_dir = resolve_work_dir(args.work_dir)
    _, status = load_status(work_dir)
    config = load_osg_config(status.get("input_config", {}))
    _, state_path = state_paths(work_dir, config)
    state = load_state(state_path)
    changed = 0
    for task_id, data in list(state.get("tasks", {}).items()):
        task = task_from_dict(data)
        if task.status not in {"failed", "held"}:
            continue
        updated = retry_or_quarantine(task, config.retries)
        if updated.status == "planned":
            meta = status.get("global_tasks", {}).get(updated.yarp_task_id)
            if meta is not None:
                meta["status"] = "ready"
                meta["job_id"] = None
        upsert_task(state, updated)
        event(state, "retry_state_updated", task_id, status=updated.status)
        changed += 1
    save_status(work_dir, status)
    save_state(state_path, state)
    print(f"Updated {changed} retry candidate(s)")
    return 0


def advance(args) -> int:
    batch_rc = run_for_discovered_work_dirs(args, advance, "advance")
    if batch_rc is not None:
        return batch_rc

    harvest_args = argparse.Namespace(work_dir=args.work_dir, skip_condor=args.skip_condor)
    harvest(harvest_args)
    prepare_args = argparse.Namespace(work_dir=args.work_dir, workflow_dir=args.workflow_dir)
    return prepare_egat(prepare_args)


def run_single_cycle(args) -> int:
    work_dir = resolve_work_dir(args.work_dir)
    print("==============================================")
    print("YARP OSG controller cycle")
    print("==============================================")
    print(f"Work dir: {work_dir}")
    print("")

    harvest_args = argparse.Namespace(work_dir=str(work_dir), skip_condor=args.skip_condor)
    harvest(harvest_args)

    retry_args = argparse.Namespace(work_dir=str(work_dir))
    retry(retry_args)

    prepare_args = argparse.Namespace(work_dir=str(work_dir), workflow_dir=args.workflow_dir)
    try:
        prepare_egat(prepare_args)
    except ValueError as exc:
        print(f"Preparation blocked: {exc}")
        return 2

    submit_args = argparse.Namespace(work_dir=str(work_dir))
    submit_rc = submit_pending(submit_args)
    status_cmd(submit_args)
    return submit_rc


def run_cycle(args) -> int:
    batch_rc = run_for_discovered_work_dirs(args, run_single_cycle, "controller cycle")
    if batch_rc is not None:
        return batch_rc
    return run_single_cycle(args)


def run_default(args) -> int:
    cycles = 0
    while True:
        cycles += 1
        rc = run_cycle(args)
        if not args.watch:
            return rc
        if args.max_cycles and cycles >= args.max_cycles:
            return rc
        print(f"Sleeping {args.interval} seconds before next YARP OSG cycle")
        sys.stdout.flush()
        time.sleep(args.interval)


def cleanup(args) -> int:
    batch_rc = run_for_discovered_work_dirs(args, cleanup, "cleanup")
    if batch_rc is not None:
        return batch_rc

    work_dir = resolve_work_dir(args.work_dir)
    _, status = load_status(work_dir)
    config = load_osg_config(status.get("input_config", {}))
    root, _ = state_paths(work_dir, config)
    removed = 0
    if args.logs and (root / "logs").exists():
        shutil.rmtree(root / "logs")
        removed += 1
    print(f"Cleanup removed {removed} item(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yarp-osg", description="DFT-style OSG helper workflow for YARP")
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("work_dir", nargs="?", default=".")
    run_cmd.add_argument("--workflow-dir")
    run_cmd.add_argument("--skip-condor", action="store_true")
    run_cmd.add_argument("--watch", action="store_true", help="repeat controller cycles until interrupted")
    run_cmd.add_argument("--interval", type=int, default=300)
    run_cmd.add_argument("--max-cycles", type=int, default=0)
    run_cmd.set_defaults(func=run_default)

    for name, func in [
        ("plan", plan),
        ("prepare-egat", prepare_egat),
        ("submit", submit_pending),
        ("status", status_cmd),
        ("harvest", harvest),
        ("retry", retry),
        ("advance", advance),
        ("cleanup", cleanup),
    ]:
        cmd = sub.add_parser(name)
        cmd.add_argument("work_dir", nargs="?", default=".")
        cmd.set_defaults(func=func)

    sub.choices["prepare-egat"].add_argument("--workflow-dir")
    sub.choices["advance"].add_argument("--workflow-dir")
    sub.choices["harvest"].add_argument("--skip-condor", action="store_true")
    sub.choices["advance"].add_argument("--skip-condor", action="store_true")
    sub.choices["cleanup"].add_argument("--logs", action="store_true")

    record = sub.add_parser("record-submit")
    record.add_argument("work_dir")
    record.add_argument("--cluster-id", required=True)
    record.add_argument("--batch-file", required=True)
    record.set_defaults(func=record_submit)
    return parser


def normalize_argv(argv: list[str]) -> list[str]:
    if not argv:
        return ["run"]
    if argv[0] in {"-h", "--help"}:
        return argv
    if argv[0] not in COMMANDS:
        return ["run", *argv]
    return argv


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    argv = normalize_argv(list(argv))
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
