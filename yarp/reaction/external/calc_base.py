import platform
from pathlib import Path
import shutil
import subprocess

class AsyncYarpCalculator:
    """
    Base class defining the asynchronous lifecycle interface required by progress_yarp.py.
    """
    def __init__(self, task_def, rxn_data, job_config):
        self.task_def = task_def
        self.config = task_def.config
        if isinstance(rxn_data, dict):
            self.reactions = rxn_data
            self.rxn = None
        else:
            self.rxn = rxn_data
            self.reactions = None
        self.job_manager = job_config
        self.scratch_dir = None

    def set_scratch_dir(self, path: Path):
        self.scratch_dir = path
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

    def get_container_prefix(self, image_name: str, work_dir: str, *, apptainer_run: bool = False) -> str:
        """
        The universal toggle for container execution.
        Maps the scratch directory to /work inside the container.
        Automatically checks for and pulls missing images.

        apptainer_run: If True, use ``apptainer run`` (OCI ENTRYPOINT/CMD, like ``docker run``).
        If False, use ``apptainer exec`` (requires an explicit executable before flags).
        """
        if self.job_manager.container == "docker":
            # 1. Check if the image exists locally.
            # Use `docker images -q` rather than `docker image inspect` because
            # the latter fails silently with the containerd image store in Docker Desktop.
            check_cmd = subprocess.run(
                ["docker", "images", "-q", image_name],
                capture_output=True, text=True
            )

            # 2. If output is empty, the image isn't local. Pull it!
            if not check_cmd.stdout.strip():
                print(f"Docker image '{image_name}' not found locally. Pulling from registry...")
                subprocess.run(["docker", "pull", "--platform", "linux/amd64", image_name], check=True)

            return f"docker run --platform linux/amd64 --rm -v {work_dir}:/work -w /work {image_name}"

        elif self.job_manager.container == "apptainer":
            # Sanitize the image name so it works as a safe, flat filename
            # e.g., "erm42/yarp:crest" -> "erm42_yarp_crest.sif"
            sanitized = image_name.replace("/", "_").replace(":", "_")
            if sanitized.endswith(".sif"):
                safe_filename = sanitized
                local_sif_only = True
            else:
                safe_filename = f"{sanitized}.sif"
                local_sif_only = False

            sif_path = Path(self.job_manager.sif_location) / safe_filename
            # 1. Check if the .sif file exists on disk
            if not sif_path.exists():
                if local_sif_only:
                    raise FileNotFoundError(
                        f"Local Apptainer image not found: {sif_path}\n"
                        f"(image_name={image_name!r}). This name is treated as a filename under "
                        f"`job_manager.sif_location`, not a Docker Hub image.\n"
                        "Build the .sif from your definition file, for example:\n"
                        f"  apptainer build {sif_path} /path/to/orca_6.0.1.def\n"
                        "Place the ORCA installer tarball next to the .def as documented in that file."
                    )
                print(f"Apptainer image not found at {sif_path}. Pulling from Docker Hub...")
                # Ensure the target directory actually exists before pulling
                sif_path.parent.mkdir(parents=True, exist_ok=True)
                # 2. Pull the docker image and convert it to a .sif file
                try:
                    subprocess.run(
                        ["apptainer", "pull", str(sif_path), f"docker://{image_name}"],
                        check=True,
                    )
                except FileNotFoundError:
                    os_name = platform.system()
                    if os_name in ("Darwin", "Windows"):
                        msg = (
                            f"Apptainer is not supported on {os_name}. "
                            "Switch the 'container' field in your input file to 'docker' and ensure Docker Desktop is installed."
                        )
                    else:
                        msg = (
                            "Could not find 'apptainer' on your system. To fix this, either:\n"
                            "  1. Install Apptainer (https://apptainer.org/docs/admin/main/installation.html)\n"
                            "  2. Switch the 'container' field in your input file to 'docker'"
                        )
                    raise RuntimeError(msg) from None

            verb = "run" if apptainer_run else "exec"
            return f"apptainer {verb} -e --bind {work_dir}:/work --pwd /work {sif_path}"

        elif self.job_manager.container == "singularity":
            # Sanitize the image name so it works as a safe, flat filename
            # e.g., "erm42/yarp:crest" -> "erm42_yarp_crest.sif"
            sanitized = image_name.replace("/", "_").replace(":", "_")
            if sanitized.endswith(".sif"):
                safe_filename = sanitized
                local_sif_only = True
            else:
                safe_filename = f"{sanitized}.sif"
                local_sif_only = False

            sif_path = Path(self.job_manager.sif_location) / safe_filename
            # 1. Check if the .sif file exists on disk
            if not sif_path.exists():
                if local_sif_only:
                    raise FileNotFoundError(
                        f"Local Singularity image not found: {sif_path}\n"
                        f"(image_name={image_name!r}). This name is treated as a filename under "
                        f"`job_manager.sif_location`, not a Docker Hub image.\n"
                        "Build the .sif from your definition file, for example:\n"
                        f"  singularity build {sif_path} /path/to/orca_6.0.1.def\n"
                        "Place the ORCA installer tarball next to the .def as documented in that file."
                    )
                print(f"Singularity image not found at {sif_path}. Pulling from Docker Hub...")
                # Ensure the target directory actually exists before pulling
                sif_path.parent.mkdir(parents=True, exist_ok=True)
                # 2. Pull the docker image and convert it to a .sif file
                subprocess.run(
                    ["singularity", "pull", str(sif_path), f"docker://{image_name}"],
                    check=True,
                )

            verb = "run" if apptainer_run else "exec"
            return f"singularity {verb} -e --bind {work_dir}:/work --pwd /work {sif_path}"

        else:
            raise ValueError(f"Unsupported container runner: {self.job_manager.container}")

    def write_scheduler_headers(self, f):
        """Writes the top portion of the bash script based on scheduler type."""
        scheduler = self.job_manager.scheduler
        job_name = self.job_manager.job_name
        queue = self.job_manager.queue
        cpus = self.config.n_cpus
        mem = self.config.mem_per_cpu
        time = self.config.max_runtime

        if scheduler == "slurm":
            if self.job_manager.account:
                f.write(f"#SBATCH -A {self.job_manager.account}\n")
            f.write(f"#SBATCH --job-name={job_name}\n")
            f.write(f"#SBATCH --partition={queue}\n")
            f.write("#SBATCH -N 1\n") # nodes
            f.write("#SBATCH -n 1\n") # tasks
            f.write(f"#SBATCH --cpus-per-task={cpus}\n")
            f.write(f"#SBATCH --mem-per-cpu={mem}M\n")
            f.write(f"#SBATCH --time={time}\n")
            f.write("#SBATCH --output /dev/null\n")
            f.write("#SBATCH --error /dev/null\n\n")

            if self.job_manager.module_container:
                f.write(f"{self.job_manager.module_container}\n\n")

        elif scheduler == "sge":
            f.write(f"#$ -N {job_name}\n")
            f.write(f"#$ -q {queue}\n")
            f.write(f"#$ -pe smp {cpus}\n") # ERM: this might be specific to CRC at ND
            f.write(f"#$ -l h_vmem={mem * cpus}M\n")
            f.write(f"#$ -l h_rt={time}\n")
            f.write("#$ -o /dev/null\n")
            f.write("#$ -e /dev/null\n\n")

            if self.job_manager.module_container:
                f.write(f"{self.job_manager.module_container}\n\n")

        # if 'condor', we don't need scheduler headers!
        # If 'local', we don't need scheduler headers!

    # --- Pre-flight Check (Overridden by Task classes) ---
    def has_prerequisites(self) -> bool:
        return True

    # --- The 5 Lifecycle Methods (Overridden by Software classes) ---
    def generate_input(self):
        """1. Write input files (e.g., .xyz, .inp) based on self.rxn geometries."""
        raise NotImplementedError

    def write_submission_script(self) -> Path:
        """2. Write the SLURM/QSE bash script (including container commands) and return its Path."""
        raise NotImplementedError

    def check_output(self) -> bool:
        """3. Validate completeness (e.g., check for 'Normal termination' or expected files)."""
        raise NotImplementedError

    def scrape_data(self):
        """4. Extract data from outputs and update self.rxn in memory."""
        raise NotImplementedError

    def cleanup(self):
        """5. Clean up large unneeded files, but keep logs if necessary."""
        # Default behavior: nuke the whole folder
        if self.scratch_dir and self.scratch_dir.exists():
            shutil.rmtree(self.scratch_dir)

