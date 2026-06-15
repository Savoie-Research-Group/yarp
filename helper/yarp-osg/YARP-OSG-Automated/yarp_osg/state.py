from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import RetryConfig


ACTIVE_STATUSES = {"planned", "prepared", "submitted", "idle", "running", "held"}
FINAL_STATUSES = {"harvested", "failed", "quarantined"}
RETRYABLE_INFRA = {"infrastructure", "resource", "missing_output"}
RETRYABLE_CHEMISTRY = {"chemistry"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class OSGTask:
    task_id: str
    yarp_task_id: str
    task_type: str = "egat_ml_predict"
    scope: str = "global"
    status: str = "planned"
    attempt: int = 0
    task_dir: str | None = None
    manifest: str | None = None
    condor_id: str | None = None
    cluster_id: str | None = None
    proc_id: str | None = None
    error_category: str | None = None
    error_message: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_state() -> dict[str, Any]:
    return {"version": 1, "tasks": {}, "events": []}


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return new_state()
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    state.setdefault("version", 1)
    state.setdefault("tasks", {})
    state.setdefault("events", [])
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def event(state: dict[str, Any], action: str, task_id: str | None = None, **extra: Any) -> None:
    state.setdefault("events", []).append(
        {"time": utc_now(), "action": action, "task_id": task_id, **extra}
    )


def upsert_task(state: dict[str, Any], task: OSGTask) -> dict[str, Any]:
    task.updated_at = utc_now()
    state.setdefault("tasks", {})[task.task_id] = task.to_dict()
    return state["tasks"][task.task_id]


def task_from_dict(data: dict[str, Any]) -> OSGTask:
    allowed = OSGTask.__dataclass_fields__
    return OSGTask(**{key: value for key, value in data.items() if key in allowed})


def should_quarantine(task: OSGTask, retry_config: RetryConfig) -> bool:
    return task.attempt >= retry_config.quarantine_after


def can_retry(task: OSGTask, retry_config: RetryConfig) -> bool:
    if task.error_category in RETRYABLE_INFRA:
        return task.attempt < retry_config.infrastructure
    if task.error_category in RETRYABLE_CHEMISTRY:
        return task.attempt < retry_config.chemistry
    return False


def retry_or_quarantine(task: OSGTask, retry_config: RetryConfig) -> OSGTask:
    if should_quarantine(task, retry_config) or not can_retry(task, retry_config):
        task.status = "quarantined"
    else:
        task.status = "planned"
        task.condor_id = None
        task.cluster_id = None
        task.proc_id = None
    task.updated_at = utc_now()
    return task
