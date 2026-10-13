"""
Tests for the container command and scheduler header that every job script is
built from.

`get_container_prefix` and `write_scheduler_headers` live on the base class
and every calculator calls both, so together they decide what a job does on
the compute node. These tests pin both as exact text, so any change to the
scripts shows up here as a deliberate diff.
"""
import io
from types import SimpleNamespace

import pytest

from yarp.reaction.external.calc_base import AsyncYarpCalculator
from yarp.util.config import JobManagerConfig, TSOptConfig

WORK = "/nfs/run/SCRATCH/abc123_refine.transition_state_optimization"
HUB_IMAGE = "erm42/yarp:crest"
ENV = {"OMP_NUM_THREADS": 4, "MKL_NUM_THREADS": 4}
SIF_RUNNERS = ["apptainer", "singularity"]


def make_calc(job_config, **cfg):
    config = TSOptConfig(software="orca", lot="B3LYP def2-SVP", charge=0, multiplicity=1, **cfg)
    return AsyncYarpCalculator(SimpleNamespace(config=config), None, job_config)


class RunRecorder:
    """Stands in for `subprocess.run`: records every call, then raises `raises` or returns `stdout`."""

    def __init__(self, stdout="", raises=None):
        self.stdout = stdout
        self.raises = raises
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        if self.raises:
            raise self.raises
        return SimpleNamespace(stdout=self.stdout, returncode=0)


@pytest.fixture
def run(monkeypatch):
    """Install a RunRecorder in place of the `subprocess.run` calc_base calls."""
    def install(stdout="", raises=None):
        recorder = RunRecorder(stdout, raises)
        monkeypatch.setattr("yarp.reaction.external.calc_base.subprocess.run", recorder)
        return recorder
    return install


class TestDockerPrefix:
    def _prefix(self, **kwargs):
        return make_calc(JobManagerConfig(container="docker")).get_container_prefix(HUB_IMAGE, WORK, **kwargs)

    def test_local_image(self, run):
        recorder = run(stdout="0123456789ab\n")

        assert self._prefix() == f"docker run --rm  -v {WORK}:/work -w /work {HUB_IMAGE}"
        assert recorder.calls == [
            (["docker", "images", "-q", HUB_IMAGE], {"capture_output": True, "text": True}),
        ]

    def test_env_vars(self, run):
        run(stdout="0123456789ab\n")

        assert self._prefix(env_vars=ENV) == (
            f"docker run --rm -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4  -v {WORK}:/work -w /work {HUB_IMAGE}"
        )

    def test_apptainer_run_is_ignored(self, run):
        run(stdout="0123456789ab\n")

        assert self._prefix(apptainer_run=True) == self._prefix()

    def test_missing_image_is_pulled(self, run):
        recorder = run(stdout="")

        assert self._prefix() == f"docker run --rm  -v {WORK}:/work -w /work {HUB_IMAGE}"
        assert recorder.calls[1] == (["docker", "pull", HUB_IMAGE], {"check": True})


@pytest.mark.parametrize("runner", SIF_RUNNERS)
class TestSifPrefix:
    """Apptainer and Singularity share one code path, apart from their names."""

    def _calc(self, runner, sif_dir):
        return make_calc(JobManagerConfig(container=runner, sif_location=str(sif_dir)))

    def test_exec_with_an_image_on_disk(self, runner, tmp_path, run):
        sif = tmp_path / "erm42_yarp_crest.sif"
        sif.touch()
        recorder = run()

        prefix = self._calc(runner, tmp_path).get_container_prefix(HUB_IMAGE, WORK)

        assert prefix == f"{runner} exec -e  --bind {WORK}:/work --pwd /work {sif}"
        assert recorder.calls == []

    def test_run_verb(self, runner, tmp_path, run):
        sif = tmp_path / "erm42_yarp_crest.sif"
        sif.touch()
        run()

        prefix = self._calc(runner, tmp_path).get_container_prefix(HUB_IMAGE, WORK, apptainer_run=True)

        assert prefix == f"{runner} run -e  --bind {WORK}:/work --pwd /work {sif}"

    def test_env_vars(self, runner, tmp_path, run):
        sif = tmp_path / "erm42_yarp_crest.sif"
        sif.touch()
        run()

        prefix = self._calc(runner, tmp_path).get_container_prefix(HUB_IMAGE, WORK, env_vars=ENV)

        assert prefix == (
            f"{runner} exec -e --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4  "
            f"--bind {WORK}:/work --pwd /work {sif}"
        )

    def test_local_sif_name(self, runner, tmp_path, run):
        """A name ending in .sif is a file under sif_location, not a Docker Hub image."""
        sif = tmp_path / "orca_6.0.1.sif"
        sif.touch()
        run()

        prefix = self._calc(runner, tmp_path).get_container_prefix("orca_6.0.1.sif", WORK)

        assert prefix == f"{runner} exec -e  --bind {WORK}:/work --pwd /work {sif}"

    def test_missing_local_sif_is_an_error(self, runner, tmp_path, run):
        recorder = run()

        with pytest.raises(FileNotFoundError, match=f"Local {runner.capitalize()} image not found"):
            self._calc(runner, tmp_path).get_container_prefix("orca_6.0.1.sif", WORK)
        assert recorder.calls == []

    def test_missing_hub_image_is_pulled(self, runner, tmp_path, run):
        sif_dir = tmp_path / "not_yet_made"
        recorder = run()

        prefix = self._calc(runner, sif_dir).get_container_prefix(HUB_IMAGE, WORK)

        sif = sif_dir / "erm42_yarp_crest.sif"
        assert recorder.calls == [([runner, "pull", str(sif), f"docker://{HUB_IMAGE}"], {"check": True})]
        assert sif_dir.is_dir()
        assert prefix == f"{runner} exec -e  --bind {WORK}:/work --pwd /work {sif}"

    @pytest.mark.parametrize("os_name, message", [
        ("Linux", "Could not find '{runner}' on your system"),
        ("Darwin", "{Runner} is not supported on Darwin"),
    ], ids=["linux", "darwin"])
    def test_missing_runner_is_explained(self, runner, os_name, message, tmp_path, run, monkeypatch):
        """Pulling an image with no apptainer/singularity installed says what to do instead."""
        run(raises=FileNotFoundError(2, "No such file or directory", runner))
        monkeypatch.setattr("yarp.reaction.external.calc_base.platform.system", lambda: os_name)

        with pytest.raises(RuntimeError, match=message.format(runner=runner, Runner=runner.capitalize())):
            self._calc(runner, tmp_path).get_container_prefix(HUB_IMAGE, WORK)


def test_unsupported_container_is_rejected():
    # JobManagerConfig refuses this at parse time; the prefix has its own guard.
    job_config = JobManagerConfig(container="docker")
    job_config.container = "podman"

    with pytest.raises(ValueError, match="Unsupported container runner: podman"):
        make_calc(job_config).get_container_prefix(HUB_IMAGE, WORK)


def header(job_config):
    f = io.StringIO()
    make_calc(job_config, n_cpus=4, mem_per_cpu=2000, max_runtime="12:00:00").write_scheduler_headers(f)
    return f.getvalue()


class TestSchedulerHeaders:
    def test_sge(self):
        job_config = JobManagerConfig(scheduler="sge", container="apptainer", queue="long", job_name="khp")

        assert header(job_config) == (
            "#$ -N khp\n"
            "#$ -q long\n"
            "#$ -pe smp 4\n"
            "#$ -l h_vmem=8000M\n"
            "#$ -l h_rt=12:00:00\n"
            "#$ -o /dev/null\n"
            "#$ -e /dev/null\n"
            "\n"
        )

    def test_slurm(self):
        job_config = JobManagerConfig(scheduler="slurm", container="apptainer", queue="cpu", job_name="khp")

        assert header(job_config) == (
            "#SBATCH --job-name=khp\n"
            "#SBATCH --partition=cpu\n"
            "#SBATCH -N 1\n"
            "#SBATCH -n 1\n"
            "#SBATCH --cpus-per-task=4\n"
            "#SBATCH --mem-per-cpu=2000M\n"
            "#SBATCH --time=12:00:00\n"
            "#SBATCH --output /dev/null\n"
            "#SBATCH --error /dev/null\n"
            "\n"
        )

    def test_slurm_account_comes_first(self):
        job_config = JobManagerConfig(scheduler="slurm", container="apptainer", queue="cpu", account="abc123")

        assert header(job_config).startswith("#SBATCH -A abc123\n#SBATCH --job-name=yarp\n")

    @pytest.mark.parametrize("scheduler", ["sge", "slurm"])
    def test_module_line_follows_the_header(self, scheduler):
        job_config = JobManagerConfig(scheduler=scheduler, container="apptainer", queue="long",
                                      module_container="module load apptainer")

        assert header(job_config).endswith("/dev/null\n\nmodule load apptainer\n\n")

    def test_local_writes_nothing(self):
        job_config = JobManagerConfig(scheduler="local", container="apptainer",
                                      module_container="module load apptainer")

        assert header(job_config) == ""
