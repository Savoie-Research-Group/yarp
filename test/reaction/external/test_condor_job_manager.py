from dataclasses import dataclass
from pathlib import Path

from yarp.reaction.external.job_manager import CondorJobManager


@dataclass
class FakeJobConfig:
    request_disk: str = "8GB"
    log_dir: str = "condor_logs"
    getenv: bool = True
    notification: str = "Never"
    condor_universe: str = "vanilla"


@dataclass
class FakeTaskConfig:
    n_cpus: int = 8
    mem_per_cpu: int = 1000


def test_condor_submit_file_contains_resources(tmp_path):
    script = tmp_path / "run_job.sh"
    script.write_text("#!/bin/bash\necho test\n")

    manager = CondorJobManager(FakeJobConfig())
    submit_path = manager.write_submit_file(script, FakeTaskConfig())

    text = submit_path.read_text()

    assert "universe = vanilla" in text
    assert f"executable = {script.resolve()}" in text
    assert "request_cpus = 8" in text
    assert "request_memory = 8000MB" in text
    assert "request_disk = 8GB" in text
    assert "queue 1" in text


def test_condor_submit_file_creates_log_directory(tmp_path):
    script = tmp_path / "run_job.sh"
    script.write_text("#!/bin/bash\n")

    manager = CondorJobManager(FakeJobConfig())
    manager.write_submit_file(script, FakeTaskConfig())

    assert (tmp_path / "condor_logs").is_dir()


def test_condor_script_is_executable(tmp_path):
    script = tmp_path / "run_job.sh"
    script.write_text("#!/bin/bash\n")

    manager = CondorJobManager(FakeJobConfig())
    manager.write_submit_file(script, FakeTaskConfig())

    assert script.stat().st_mode & 0o111