import importlib.util
from types import SimpleNamespace
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "yarp" / "reaction" / "external" / "model_scorer.py"
SPEC = importlib.util.spec_from_file_location("model_scorer_under_test", MODULE_PATH)
model_scorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model_scorer)


class DummyRunResult:
    def __init__(self, stdout):
        self.stdout = stdout


def test_docker_prefix_returns_when_image_exists(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return DummyRunResult(stdout="image-id\n")

    monkeypatch.setattr(model_scorer.subprocess, "run", fake_run)

    prefix = model_scorer.get_container_prefix(
        SimpleNamespace(container="docker"),
        "erm42/yarp:model_scorer",
        str(tmp_path),
    )

    assert prefix.startswith("docker run --platform linux/amd64 --rm")
    assert "erm42/yarp:model_scorer" in prefix
    assert calls == [["docker", "images", "-q", "erm42/yarp:model_scorer"]]
