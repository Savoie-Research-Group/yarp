import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "helper" / "yarp-osg" / "YARP-OSG-Automated"))

from yarp_osg.cli import discover_work_dirs, run_for_discovered_work_dirs


def write_minimal_status(work_dir: Path):
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "STATUS.json").write_text(
        json.dumps(
            {
                "status_output_file": "STATUS.json",
                "reaction_output_file": "YARP_RXNS.pkl",
                "input_config": {},
                "global_tasks": {},
                "reactions": {},
            }
        )
    )


def test_discover_single_initialized_workdir(tmp_path):
    write_minimal_status(tmp_path)

    assert discover_work_dirs(tmp_path) == [tmp_path]


def test_discover_batch_direct_child_workdirs_sorted(tmp_path):
    write_minimal_status(tmp_path / "case_b")
    write_minimal_status(tmp_path / "case_a")
    (tmp_path / "notes").mkdir()

    assert discover_work_dirs(tmp_path) == [tmp_path / "case_a", tmp_path / "case_b"]


def test_discover_batch_rejects_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_work_dirs(tmp_path)


def test_batch_runner_calls_handler_for_each_child(tmp_path):
    write_minimal_status(tmp_path / "case_001")
    write_minimal_status(tmp_path / "case_002")
    seen = []

    def handler(args):
        seen.append(Path(args.work_dir).name)
        return 0

    args = type("Args", (), {"work_dir": str(tmp_path)})()

    assert run_for_discovered_work_dirs(args, handler, "test") == 0
    assert seen == ["case_001", "case_002"]
