import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "helper" / "yarp-osg" / "YARP-OSG-Automated"))

from yarp_osg.cli import normalize_argv
from yarp_osg.config import DEFAULT_EGAT_COMMAND, image_name_to_sif_name, load_osg_config


def test_path_only_invocation_becomes_run_command():
    assert normalize_argv(["."]) == ["run", "."]
    assert normalize_argv(["/tmp/workdir", "--watch"]) == ["run", "/tmp/workdir", "--watch"]
    assert normalize_argv(["status", "."]) == ["status", "."]


def test_egat_sif_name_follows_yarp_image_convention():
    assert image_name_to_sif_name("erm42/yarp:egat") == "erm42_yarp_egat.sif"


def test_default_config_uses_yarp_egat_command_and_osdf_namespace(monkeypatch, tmp_path):
    monkeypatch.setenv("YARP_OSG_LOCAL_SIF_DIR", str(tmp_path))
    monkeypatch.setenv("YARP_OSG_OSDF_NAMESPACE", "/ospool/ap40/data/test-user")
    monkeypatch.delenv("YARP_OSG_EGAT_CONTAINER", raising=False)
    monkeypatch.delenv("YARP_OSG_EGAT_COMMAND", raising=False)

    config = load_osg_config({})

    assert config.egat_image == "erm42/yarp:egat"
    assert config.egat_local_sif == str(tmp_path / "erm42_yarp_egat.sif")
    assert config.egat_container == "osdf:///ospool/ap40/data/test-user/yarp-containers/v1/erm42_yarp_egat.sif"
    assert config.egat_command == DEFAULT_EGAT_COMMAND


def test_default_config_builds_dft_style_osdf_path(monkeypatch, tmp_path):
    monkeypatch.setenv("USER", "thomas.burton")
    monkeypatch.setenv("YARP_OSG_LOCAL_SIF_DIR", str(tmp_path))
    monkeypatch.delenv("YARP_OSG_OSDF_NAMESPACE", raising=False)
    monkeypatch.delenv("YARP_OSG_EGAT_CONTAINER", raising=False)

    config = load_osg_config({})

    assert config.osdf_namespace == "/ospool/ap40/data/thomas.burton"
    assert config.egat_container == "osdf:///ospool/ap40/data/thomas.burton/yarp-containers/v1/erm42_yarp_egat.sif"
