"""
Tests for the defaults and value checks in the config dataclasses.
"""
from pathlib import Path

import pytest

import yarp.util.config
from yarp.util.config import JobManagerConfig

DEFAULT_SIF_DIR = str(Path(yarp.util.config.__file__).resolve().parents[2] / "containers")


class TestJobManagerModuleContainer:
    """`module_container` is checked as a string, independently of `sif_location`."""

    def test_docker_accepts_a_module_line_without_a_sif_location(self):
        job_config = JobManagerConfig(container="docker", module_container="module load apptainer")

        assert job_config.module_container == "module load apptainer"
        assert job_config.sif_location is None

    def test_non_string_module_line_is_rejected(self):
        with pytest.raises(ValueError, match="'module_container'"):
            JobManagerConfig(container="apptainer", module_container=5)


class TestJobManagerSifLocation:
    """Apptainer and Singularity both read .sif files, so both get the default directory."""

    @pytest.mark.parametrize("runner", ["apptainer", "singularity"])
    def test_defaults_to_the_repo_containers_dir(self, runner):
        assert JobManagerConfig(container=runner).sif_location == DEFAULT_SIF_DIR

    @pytest.mark.parametrize("runner", ["apptainer", "singularity"])
    def test_user_location_is_kept(self, runner):
        assert JobManagerConfig(container=runner, sif_location="/sifs").sif_location == "/sifs"

    def test_docker_has_no_sif_location(self):
        assert JobManagerConfig(container="docker").sif_location is None
