"""
Tests for the defaults and value checks in the config dataclasses.
"""
from functools import partial
from pathlib import Path

import pytest

import yarp.util.config
from yarp.util.config import (
    JobManagerConfig, MLPropConfig, PreOptConfig, ConformerConfig, TSGuessConfig,
    RPOptConfig, TSOptConfig, IRCValConfig,
)

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


class TestJobManagerLocalScratchDir:
    def test_defaults_to_tmpdir(self):
        assert JobManagerConfig().local_scratch_dir == "$TMPDIR"

    def test_user_path_is_kept_verbatim(self):
        """Shell variables are left for the compute node to expand."""
        job_config = JobManagerConfig(local_scratch_dir="/scratch/local/$USER")

        assert job_config.local_scratch_dir == "/scratch/local/$USER"

    @pytest.mark.parametrize("value", [5, "", "   "])
    def test_non_string_or_blank_is_rejected(self, value):
        with pytest.raises(ValueError, match="'local_scratch_dir'"):
            JobManagerConfig(local_scratch_dir=value)


# Each task block's config, built with only its required keys.
TASK_CONFIGS = {
    "ml_rxn_prop": partial(MLPropConfig, model="egat_rgd1"),
    "pre_opt": partial(PreOptConfig),
    "conf_gen": partial(ConformerConfig, software="crest", lot="gfn2", charge=0, n_unpaired_electrons=0),
    "ts_guess": partial(TSGuessConfig, software="pysisyphus", gsm_lot="xtb", charge=0, multiplicity=1),
    "rp_opt": partial(RPOptConfig, software="orca", lot="B3LYP def2-SVP", charge=0, multiplicity=1),
    "ts_opt": partial(TSOptConfig, software="orca", lot="B3LYP def2-SVP", charge=0, multiplicity=1),
    "irc_val": partial(IRCValConfig, software="orca", lot="B3LYP def2-SVP", charge=0, multiplicity=1),
}


@pytest.mark.parametrize("block", TASK_CONFIGS)
class TestUseLocalScratch:
    def test_defaults_to_false(self, block):
        assert TASK_CONFIGS[block]().use_local_scratch is False

    def test_true_is_kept(self, block):
        assert TASK_CONFIGS[block](use_local_scratch=True).use_local_scratch is True

    @pytest.mark.parametrize("value", ["yes", 1, "true"])
    def test_non_boolean_is_rejected(self, block, value):
        with pytest.raises(ValueError, match=f"boolean value .* to {block}.*'use_local_scratch'"):
            TASK_CONFIGS[block](use_local_scratch=value)


@pytest.mark.parametrize("block", TASK_CONFIGS)
class TestKeepCrashLogs:
    def test_defaults_to_false(self, block):
        assert TASK_CONFIGS[block]().keep_crash_logs is False

    def test_true_is_kept(self, block):
        assert TASK_CONFIGS[block](keep_crash_logs=True).keep_crash_logs is True

    @pytest.mark.parametrize("value", ["yes", 1, "true"])
    def test_non_boolean_is_rejected(self, block, value):
        with pytest.raises(ValueError, match=f"boolean value .* to {block}.*'keep_crash_logs'"):
            TASK_CONFIGS[block](keep_crash_logs=value)


@pytest.mark.parametrize("block", ["ts_opt", "irc_val"])
class TestMinImagFreq:
    def test_defaults_to_the_old_tolerance(self, block):
        assert TASK_CONFIGS[block]().min_imag_freq == 0.01

    @pytest.mark.parametrize("value", [50, 50.0, 0.01])
    def test_positive_number_is_kept(self, block, value):
        assert TASK_CONFIGS[block](min_imag_freq=value).min_imag_freq == value

    @pytest.mark.parametrize("value", [0, -50.0, "50", True, None])
    def test_non_positive_or_non_number_is_rejected(self, block, value):
        with pytest.raises(ValueError, match=f"positive number .* to {block}: 'min_imag_freq'"):
            TASK_CONFIGS[block](min_imag_freq=value)
