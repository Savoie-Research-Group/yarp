"""Shared helpers for SMILES-driven YARP batch scripts."""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = {
    "smiles_csv": "smiles.csv",
    "template_config": "config.template.yaml",
    "output_dir": "runs",
    "batch_size": 25,
    "start_batch": 1,
    "config_name": "input.yaml",
    "status_name": "STATUS.json",
    "reaction_output_name": "auto",
    "init_log_name": "yarp_init.out",
    "progress_log_name": "prog.out",
    "batch_log_name": "yarp_batch.out",
    "yarp_init": "yarp-init",
    "yarp_progress": "yarp-progress",
}


ACTIVE_STATES = {"pending", "ready", "submitted", "running"}


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def load_batch_config(path: Path | None) -> tuple[dict[str, Any], Path]:
    if path is None:
        base_dir = Path.cwd()
        config = dict(DEFAULT_CONFIG)
        return config, base_dir

    config_path = path.resolve()
    config = dict(DEFAULT_CONFIG)
    config.update(load_yaml(config_path))
    return config, config_path.parent


def resolve_config_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def read_smiles_csv(path: Path, smiles_column: str = "smiles") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"SMILES CSV has no header: {path}")
        if smiles_column not in reader.fieldnames:
            raise ValueError(
                f"SMILES CSV must contain a '{smiles_column}' column. "
                f"Found: {', '.join(reader.fieldnames)}"
            )

        for index, row in enumerate(reader, start=1):
            smiles = (row.get(smiles_column) or "").strip()
            if not smiles:
                continue
            rows.append({"index": index, "smiles": smiles, "row": row})
    return rows


def batch_name(index: int) -> str:
    return f"batch_{index:03d}"


def parse_batch_index(path: Path) -> int | None:
    match = re.fullmatch(r"batch_(\d+)", path.name)
    if not match:
        return None
    return int(match.group(1))


def existing_batch_indices(output_dir: Path) -> set[int]:
    if not output_dir.exists():
        return set()
    indices = set()
    for path in output_dir.iterdir():
        if not path.is_dir():
            continue
        index = parse_batch_index(path)
        if index is not None:
            indices.add(index)
    return indices


def next_batch_index(output_dir: Path, start_batch: int) -> int:
    existing = existing_batch_indices(output_dir)
    index = start_batch
    while index in existing:
        index += 1
    return index


def file_safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return cleaned[:80] if cleaned else ""


def fallback_source_identifier(smiles: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", smiles).strip("._-")
    if not cleaned:
        cleaned = "smiles"
    cleaned = cleaned[:60]
    digest = hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned}_{digest}"


def source_identifier(smiles: str) -> str:
    try:
        from yarp.yarpecule.yarpecule import yarpecule

        mol = yarpecule(smiles, mode="yarp")
        mol.get_inchi()
        if mol.inchi and mol.inchi != "ERROR":
            safe_inchi = file_safe_stem(mol.inchi)
            if safe_inchi:
                return safe_inchi
    except Exception:
        pass

    return fallback_source_identifier(smiles)



def render_yarp_config(
    template: dict[str, Any],
    smiles: str,
    reaction_output_name: str,
    status_name: str,
) -> dict[str, Any]:
    config = copy.deepcopy(template)
    init = config.setdefault("initialize", {})
    init["output"] = reaction_output_name
    init["status"] = status_name
    init["initial_structure"] = {
        "source": smiles,
        "type": "smiles",
        "mode": "species",
    }
    return config


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w") as handle:
        json.dump(data, handle, indent=4)


def status_is_quiescent(status_path: Path) -> bool:
    if not status_path.exists():
        return False

    status = read_json(status_path)
    for task in status.get("global_tasks", {}).values():
        if task.get("status") in ACTIVE_STATES:
            return False

    for reaction in status.get("reactions", {}).values():
        for task in reaction.get("tasks", {}).values():
            if task.get("status") in ACTIVE_STATES:
                return False

    return True


def discover_initialized_workdirs(root: Path, status_name: str) -> list[Path]:
    return sorted(path.parent.resolve() for path in root.rglob(status_name) if path.is_file())
