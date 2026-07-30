import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "containers"
    / "joint_opt"
    / "joint_opt.py"
)


@pytest.fixture
def joint_opt_runner(monkeypatch):
    openbabel_module = ModuleType("openbabel")
    openbabel_module.openbabel = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "openbabel", openbabel_module)

    spec = importlib.util.spec_from_file_location("joint_opt_runner_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_xcontrol_uses_cartesian_optimizer(joint_opt_runner, tmp_path):
    output = tmp_path / "joint_opt.xcontrol"

    joint_opt_runner._write_xcontrol(
        output,
        [
            {"atom_i": 1, "atom_j": 2, "distance": 1.1},
            {"atom_i": 2, "atom_j": 3, "distance": 1.2},
        ],
        0.5,
    )

    assert output.read_text() == (
        "$constrain\n"
        "force constant=0.5\n"
        "distance: 1, 2, 1.1000\n"
        "distance: 2, 3, 1.2000\n"
        "$end\n"
        "$opt\n"
        "engine=inertial\n"
        "$end\n"
    )


def test_read_xyz_rejects_reordered_elements(joint_opt_runner, tmp_path):
    output = tmp_path / "xtbopt.xyz"
    output.write_text("2\n\nH 0 0 0\nC 1 0 0\n")

    with pytest.raises(ValueError, match="Unexpected element order"):
        joint_opt_runner._read_xyz(output, ["C", "H"])


def test_xtb_optimize_accepts_namespaced_success_marker(joint_opt_runner, monkeypatch, tmp_path):
    def fake_run(command, cwd, capture_output, text):
        (cwd / ".joint_opt.xtboptok").touch()
        (cwd / "joint_opt.xtbopt.xyz").write_text("2\n\nC 0 0 0\nH 1 0 0\n")
        return SimpleNamespace(returncode=0, stdout="optimized geometry written", stderr="")

    monkeypatch.setattr(joint_opt_runner.subprocess, "run", fake_run)

    geometry = joint_opt_runner._xtb_optimize(
        ["C", "H"],
        [[0, 0, 0], [1, 0, 0]],
        [{"atom_i": 1, "atom_j": 2, "distance": 1.1}],
        {},
        tmp_path,
    )

    assert geometry == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
