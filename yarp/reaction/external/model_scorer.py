"""Container-backed conformer-pair model scoring.

The host YARP process prepares conformer-pair indicator features. The scorer
container owns only the sklearn/pickle model dependency stack and returns
``predict_proba`` values as JSON.
"""

import json
import platform
import shlex
import subprocess
from pathlib import Path

import pandas as pd


class ContainerModelScorer:
    """Score indicator rows using the pinned sklearn model container."""

    image_name = "erm42/yarp:model_scorer"

    def __init__(self, job_manager, work_dir, model_name, log=None):
        if job_manager is None:
            raise ValueError("ContainerModelScorer requires a job_manager configuration")
        if model_name not in {"poor_model", "rich_model"}:
            raise ValueError(f"Unknown conformer-pair model: {model_name}")

        self.job_manager = job_manager
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.log = log
        self.calls = 0
        self.model_path = None
        self.model_sha256 = None

    def predict_proba(self, indicators):
        """Return sklearn ``predict_proba`` output for one or more feature rows."""
        if isinstance(indicators, list):
            features = pd.concat(indicators, ignore_index=True)
        else:
            features = indicators.reset_index(drop=True)

        self.calls += 1
        call_dir = self.work_dir / f"score_{self.calls:04d}"
        call_dir.mkdir(parents=True, exist_ok=True)

        input_path = call_dir / "features.csv"
        output_path = call_dir / "proba.json"
        stdout_path = call_dir / "scorer.out"
        stderr_path = call_dir / "scorer.err"

        features.to_csv(input_path, index=False)

        prefix = get_container_prefix(self.job_manager, self.image_name, str(call_dir))
        cmd = " ".join([
            prefix,
            "python",
            "/opt/yarp_model_scorer/score_model.py",
            "--model",
            shlex.quote(self.model_name),
            "--input",
            "/work/features.csv",
            "--output",
            "/work/proba.json",
        ])
        self._log(f"scoring {len(features)} row(s) with {self.model_name}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        stdout_path.write_text(result.stdout or "", encoding="utf-8")
        stderr_path.write_text(result.stderr or "", encoding="utf-8")

        if result.returncode != 0:
            raise RuntimeError(
                "Conformer-pair model scorer container failed. "
                f"See {stdout_path} and {stderr_path}."
            )
        if not output_path.exists():
            raise RuntimeError(f"Conformer-pair model scorer did not write {output_path}")

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.model_path = payload.get("model_path")
        self.model_sha256 = payload.get("model_sha256")
        return payload["proba"]

    def _log(self, message):
        if self.log is not None:
            print(f"[ContainerModelScorer] {message}", file=self.log, flush=True)


def get_container_prefix(job_manager, image_name, work_dir, *, apptainer_run=False):
    """Return a docker/apptainer/singularity command prefix for a work dir."""
    runner = job_manager.container
    if runner == "docker":
        check_cmd = subprocess.run(
            ["docker", "images", "-q", image_name],
            capture_output=True,
            text=True,
        )
        if not check_cmd.stdout.strip():
            subprocess.run(["docker", "pull", "--platform", "linux/amd64", image_name], check=True)
            return (
                "docker run --platform linux/amd64 --rm "
                f"-v {shlex.quote(work_dir)}:/work -w /work {shlex.quote(image_name)}"
            )

    if runner in {"apptainer", "singularity"}:
        sanitized = image_name.replace("/", "_").replace(":", "_")
        if sanitized.endswith(".sif"):
            safe_filename = sanitized
            local_sif_only = True
        else:
            safe_filename = f"{sanitized}.sif"
            local_sif_only = False

        sif_path = Path(job_manager.sif_location) / safe_filename
        if not sif_path.exists():
            if local_sif_only:
                raise FileNotFoundError(f"Local {runner} image not found: {sif_path}")
            sif_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run(
                    [runner, "pull", str(sif_path), f"docker://{image_name}"],
                    check=True,
                )
            except FileNotFoundError:
                os_name = platform.system()
                if os_name in {"Darwin", "Windows"}:
                    raise RuntimeError(
                        f"{runner} is not supported on {os_name}; use docker for this job."
                    ) from None
                raise RuntimeError(f"Could not find '{runner}' on your system.") from None

        verb = "run" if apptainer_run else "exec"
        return (
            f"{runner} {verb} -e "
            f"--bind {shlex.quote(work_dir)}:/work "
            f"--pwd /work {shlex.quote(str(sif_path))}"
        )

    raise ValueError(f"Unsupported container runner: {runner}")
