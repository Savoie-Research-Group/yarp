import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("openbabel")

from yarp.reaction.external import joint_opt


class DummyRunResult:
    returncode = 0
    stdout = "container stdout"
    stderr = ""


def test_container_engine_sends_one_batch_and_rehydrates_conformers(monkeypatch, tmp_path):
    config = SimpleNamespace(
        joint_opt_engine="xtb",
        joint_opt_image="example/joint_opt:test",
        xtb_joint_keep_files=True,
        xtb_joint_lot="gfn2",
        charge=0,
        multiplicity=1,
        n_cpus=2,
        xtb_joint_force_constant=1.0,
        xtb_joint_scf_iters=300,
        bias_lot="uff",
    )
    conformers = [
        SimpleNamespace(
            elements=["C", "H"],
            geo=np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
            type="conf_a",
            lot="crest",
            software="crest",
        ),
        SimpleNamespace(
            elements=["C", "H"],
            geo=np.array([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]]),
            type="conf_b",
            lot="crest",
            software="crest",
        ),
    ]

    monkeypatch.setattr(joint_opt, "get_container_prefix", lambda *args: "container-prefix")

    def fake_run(command, **kwargs):
        assert "/opt/yarp_joint_opt/joint_opt.py" in command
        work_dir = tmp_path / "batch_0001"
        payload = json.loads((work_dir / "input.json").read_text())
        assert len(payload["jobs"]) == 2
        assert payload["jobs"][0]["options"]["n_cpus"] == 2
        (work_dir / "output.json").write_text(
            json.dumps(
                {
                    "results": [
                        {"label": "a", "success": True, "geo": [[0, 0, 0], [1.1, 0, 0]]},
                        {"label": "b", "success": False, "error": "not converged"},
                    ]
                }
            )
        )
        return DummyRunResult()

    monkeypatch.setattr(joint_opt.subprocess, "run", fake_run)
    engine = joint_opt.ContainerJointOptimizationEngine(SimpleNamespace(container="docker"), tmp_path, config)
    optimized = engine.optimize_many(conformers, np.array([[0, 1], [1, 0]]), ["a", "b"])

    assert optimized[1] is None
    assert optimized[0].type == "biased_xtb_conf_a"
    assert optimized[0].software == "xtb"
    assert optimized[0].lot == "gfn2"
    np.testing.assert_allclose(optimized[0].geo, [[0, 0, 0], [1.1, 0, 0]])
