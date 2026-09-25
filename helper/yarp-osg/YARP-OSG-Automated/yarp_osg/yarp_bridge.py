from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path


def find_status_path(work_dir: Path) -> Path:
    candidates = [path for path in work_dir.glob("*.json") if path.name != "failed_status.json"]
    if not candidates:
        raise FileNotFoundError(f"No STATUS JSON file found in {work_dir}")
    if len(candidates) == 1:
        return candidates[0]
    for path in candidates:
        if path.name == "STATUS.json":
            return path
    return candidates[0]


def is_initialized_yarp_dir(work_dir: Path) -> bool:
    return work_dir.is_dir() and any(path.name != "failed_status.json" for path in work_dir.glob("*.json"))


def load_yarp_state(work_dir: Path):
    status_path = find_status_path(work_dir)
    status = load_status(work_dir)[1]
    rxn_file = status.get("reaction_output_file")
    if not rxn_file:
        raise ValueError(f"{status_path} does not contain reaction_output_file")
    with (work_dir / rxn_file).open("rb") as handle:
        reactions = pickle.load(handle)
    return status_path, status, reactions


def load_status(work_dir: Path):
    status_path = find_status_path(work_dir)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    return status_path, status


def save_status(work_dir: Path, status: dict) -> None:
    status_file = status.get("status_output_file")
    if not status_file:
        raise ValueError("YARP status is missing status_output_file")
    (work_dir / status_file).write_text(json.dumps(status, indent=4) + "\n", encoding="utf-8")


def save_yarp_state(work_dir: Path, status: dict, reactions) -> None:
    status_file = status.get("status_output_file")
    rxn_file = status.get("reaction_output_file")
    if not status_file or not rxn_file:
        raise ValueError("YARP status is missing status_output_file or reaction_output_file")
    (work_dir / status_file).write_text(json.dumps(status, indent=4) + "\n", encoding="utf-8")
    with (work_dir / rxn_file).open("wb") as handle:
        pickle.dump(reactions, handle)


def parser_safe_config(raw_config: dict) -> dict:
    """Return a copy acceptable to today's InputParser, even if raw config says osg."""
    config = copy.deepcopy(raw_config)
    jm = config.setdefault("initialize", {}).setdefault("job_manager", {})
    if str(jm.get("scheduler", "")).lower() == "osg":
        jm["scheduler"] = "condor"
    if str(jm.get("container", "")).lower() == "osg":
        jm["container"] = "docker"
    return config


def load_input_parser(raw_config: dict):
    from yarp.util.input import InputParser

    return InputParser(parser_safe_config(raw_config))


def egat_ready_tasks(status: dict, parser) -> list[str]:
    ready = []
    for task_id, meta in status.get("global_tasks", {}).items():
        task_def = parser.global_tasks.get(task_id)
        if not task_def:
            continue
        model = getattr(task_def.config, "model", "")
        if task_def.task_type == "ml_predict" and model == "egat_rgd1" and meta.get("status") == "ready":
            ready.append(task_id)
    return ready
