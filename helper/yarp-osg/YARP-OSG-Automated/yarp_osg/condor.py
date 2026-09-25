from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .config import OSGConfig, ResourceProfile

Runner = Callable[..., subprocess.CompletedProcess]


@dataclass
class CondorStatus:
    state: str
    error_category: str | None = None
    hold_reason: str | None = None
    exit_code: int | None = None


def parse_cluster_id(output: str) -> str:
    match = re.search(r"cluster\s+(\d+)", output, re.IGNORECASE)
    if not match:
        raise RuntimeError(f"Could not parse condor_submit cluster id from: {output.strip()}")
    return match.group(1)


def submit(submit_file: Path, input_list: Path, runner: Runner = subprocess.run) -> str:
    result = runner(
        ["condor_submit", str(submit_file), f"input_list={input_list}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return parse_cluster_id(result.stdout + "\n" + result.stderr)


def classify_hold(reason: str | None) -> str:
    text = (reason or "").lower()
    if any(token in text for token in ("transfer", "osdf", "pelican", ".sif", "container")):
        return "infrastructure"
    if any(token in text for token in ("memory", "cpu", "disk", "resource")):
        return "resource"
    if any(token in text for token in ("executable", "permission denied", "not found", "no such file")):
        return "configuration"
    return "infrastructure"


def classify_history_exit(exit_code: int | None) -> CondorStatus:
    if exit_code == 0:
        return CondorStatus(state="complete", exit_code=exit_code)
    if exit_code in (64, 65):
        return CondorStatus(state="failed", error_category="configuration", exit_code=exit_code)
    if exit_code in (20, 21, 22):
        return CondorStatus(state="failed", error_category="chemistry", exit_code=exit_code)
    if exit_code is None:
        return CondorStatus(state="unknown")
    return CondorStatus(state="failed", error_category="infrastructure", exit_code=exit_code)


def _parse_json_records(stdout: str) -> list[dict]:
    if not stdout.strip():
        return []
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    return []


def query_job(job_id: str, runner: Runner = subprocess.run) -> CondorStatus:
    q_result = runner(
        [
            "condor_q",
            job_id,
            "-json",
            "-attributes",
            "ClusterId,ProcId,JobStatus,HoldReason",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    records = _parse_json_records(q_result.stdout)
    if records:
        status = int(records[0].get("JobStatus", 0))
        hold_reason = records[0].get("HoldReason")
        if status == 1:
            return CondorStatus(state="idle")
        if status == 2:
            return CondorStatus(state="running")
        if status == 4:
            return CondorStatus(state="complete")
        if status == 5:
            return CondorStatus(
                state="held",
                error_category=classify_hold(hold_reason),
                hold_reason=hold_reason,
            )
        if status == 3:
            return CondorStatus(state="removing")
        if status == 6:
            return CondorStatus(state="transferring")
        if status == 7:
            return CondorStatus(state="suspended")
        return CondorStatus(state="unknown")

    h_result = runner(
        ["condor_history", job_id, "-json", "-limit", "1", "-attributes", "ExitCode"],
        capture_output=True,
        text=True,
        check=False,
    )
    history = _parse_json_records(h_result.stdout)
    if history:
        exit_code = history[0].get("ExitCode")
        return classify_history_exit(int(exit_code) if exit_code is not None else None)

    return CondorStatus(state="unknown")


def write_submit_file(
    path: Path,
    *,
    config: OSGConfig,
    worker_script_name: str,
    log_dir: Path,
    resources: ResourceProfile,
) -> Path:
    if not config.egat_container:
        raise ValueError("Missing EGAT OSG container path. Set initialize.job_manager.osg.containers.egat or YARP_OSG_EGAT_CONTAINER.")

    directive = config.container_directive.strip() or "container_image"
    if directive == "+SingularityImage":
        container_line = f'+SingularityImage = "{config.egat_container}"'
    else:
        container_line = f"{directive} = {config.egat_container}"

    log_dir.mkdir(parents=True, exist_ok=True)
    text = f"""# HTCondor submit file for YARP EGAT OSG jobs
# Generated from helper/yarp-osg/YARP-OSG-Automated.

universe = vanilla
{container_line}

initialdir = $(task_dir)
executable = {worker_script_name}
arguments = $(task_id) $(attempt)

should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = {worker_script_name}, forward_in.csv, reverse_in.csv, egat_command.txt
transfer_output_files = forward_out.csv, reverse_out.csv, forward.log, forward.err, reverse.log, reverse.err, task_result.json

request_cpus = {resources.cpus}
request_memory = {resources.memory_mb} MB
request_disk = {resources.disk_mb} MB

output = {log_dir}/$(Cluster)_$(Process).out
error = {log_dir}/$(Cluster)_$(Process).err
log = {log_dir}/$(Cluster)_$(Process).log

queue task_id, task_dir, attempt from $(input_list)
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def split_manifest(master_manifest: Path, batch_size: int, output_dir: Path) -> list[Path]:
    lines = [line for line in master_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return []
    header, rows = lines[0], lines[1:]
    output_dir.mkdir(parents=True, exist_ok=True)
    batches: list[Path] = []
    for index in range(0, len(rows), batch_size):
        batch = output_dir / f"job_{len(batches) + 1}.tsv"
        batch.write_text("\n".join([header, *rows[index : index + batch_size]]) + "\n", encoding="utf-8")
        batches.append(batch)
    return batches


def manifest_rows(path: Path) -> list[dict[str, str]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return []
    header = re.split(r"\s+", lines[0].strip())
    rows = []
    for line in lines[1:]:
        values = re.split(r"\s+", line.strip(), maxsplit=len(header) - 1)
        rows.append(dict(zip(header, values)))
    return rows
