import platform
from pathlib import Path
import shutil
import subprocess


class CalculatorInputError(Exception):
    """
    `generate_input()` could not build a usable input; discard this reaction.

    There are otherwise only two places a reaction can be dropped:
    `has_prerequisites()` returning False, and a job failing its
    `check_output()`. Neither covers input construction that legitimately
    fails -- the product pre-optimization's UFF patch onto the product BEM can
    simply not converge to the target connectivity, at which point there is no
    geometry to submit. Returning quietly would submit a job against a stale
    or missing file and surface an hour later as an opaque output-validation
    error; raising anything else would propagate out of `progress_yarp` and
    kill the whole invocation for every other reaction in the pass.

    Raised by a calculator, caught narrowly by PASS 3.1/3.2, which route the
    reaction to `failed_rxns` with this exception's message.
    """


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

    def _use_local_scratch(self) -> bool:
        """Whether this job runs in node-local scratch: the task asks for it, and only SGE implements it."""
        return self.job_manager.scheduler == "sge" and self.config.use_local_scratch

    def _mounts(self, flag: str, work_dir: str) -> str:
        """
        The container's bind mounts. Normally the task dir is /work. A staged
        job instead mounts its node-local copy as /work and the NFS task dir as
        /nfs, using the variables `_write_local_scratch_staging` sets.
        """
        if self._use_local_scratch():
            return f'{flag} "$YARP_LOCAL_DIR":/work {flag} "$YARP_NFS_DIR":/nfs'
        return f"{flag} {work_dir}:/work"

    def get_container_prefix(self, image_name: str, work_dir: str, *, apptainer_run: bool = False, env_vars: dict = None) -> str:
        """
        The universal toggle for container execution.
        Maps the scratch directory to /work inside the container.
        Automatically checks for and pulls missing images.

        For a job staged in node-local scratch, /work is the local copy instead,
        and the prefix starts with `yarp_bg` so the container runs in the
        background (see `_write_local_scratch_staging`).

        apptainer_run: If True, use ``apptainer run`` (OCI ENTRYPOINT/CMD, like ``docker run``).
        If False, use ``apptainer exec`` (requires an explicit executable before flags).

        env_vars: Optional dict of environment variables to inject into the container
        (e.g. {"OMP_NUM_THREADS": 4}). Passed via -e for Docker and --env for Apptainer.
        """
        launch = "yarp_bg " if self._use_local_scratch() else ""

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
                subprocess.run(["docker", "pull", image_name], check=True)

            env_flags = ""
            if env_vars:
                env_flags = " ".join(f"-e {k}={v}" for k, v in env_vars.items()) + " "
            return f"{launch}docker run --rm {env_flags} {self._mounts('-v', work_dir)} -w /work {image_name}"

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

            env_flags = ""
            if env_vars:
                env_flags = " ".join(f"--env {k}={v}" for k, v in env_vars.items()) + " "
            verb = "run" if apptainer_run else "exec"
            return f"{launch}apptainer {verb} -e {env_flags} {self._mounts('--bind', work_dir)} --pwd /work {sif_path}"

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
                try:
                    subprocess.run(
                        ["singularity", "pull", str(sif_path), f"docker://{image_name}"],
                        check=True,
                    )
                except FileNotFoundError:
                    os_name = platform.system()
                    if os_name in ("Darwin", "Windows"):
                        msg = (
                            f"Singularity is not supported on {os_name}. "
                            "Switch the 'container' field in your input file to 'docker' and ensure Docker Desktop is installed."
                        )
                    else:
                        msg = (
                            "Could not find 'singularity' on your system. To fix this, either:\n"
                            "  1. Install SingularityCE (https://docs.sylabs.io/guides/latest/admin-guide/installation.html)\n"
                            "  2. Switch the 'container' field in your input file to 'docker'"
                        )
                    raise RuntimeError(msg) from None

            env_flags = ""
            if env_vars:
                env_flags = " ".join(f"--env {k}={v}" for k, v in env_vars.items()) + " "
            verb = "run" if apptainer_run else "exec"
            return f"{launch}singularity {verb} -e {env_flags} {self._mounts('--bind', work_dir)} --pwd /work {sif_path}"

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
            # Warn with SIGUSR2 before the h_rt SIGKILL (and SIGUSR1 before a
            # suspend), so a job staged in local scratch can copy back first.
            f.write("#$ -notify\n")
            f.write("#$ -o /dev/null\n")
            f.write("#$ -e /dev/null\n\n")

            if self.job_manager.module_container:
                f.write(f"{self.job_manager.module_container}\n\n")

            if self._use_local_scratch():
                self._write_local_scratch_staging(f)

        # If 'local', we don't need scheduler headers!

    def _write_local_scratch_staging(self, f):
        """
        Run the job in node-local scratch instead of the NFS task dir.

        Copies the task dir into a new dir under `local_scratch_dir`, and copies
        everything back when the script exits, removing the local copy only if
        that succeeded. Nothing is filtered on the way back; pruning is each
        calculator's `cleanup()`.

        SGE's SIGUSR2 (from `-notify`, at h_rt) becomes a normal exit so the
        copy-back still runs. SIGUSR1 (before a suspend) copies back and keeps
        going.

        `yarp_bg` runs the container in the background and waits for it, since
        bash holds a trap until a foreground child exits. It keeps waiting
        across a trapped SIGUSR1, and returns the container's exit status.
        """
        f.write("# Stage this task in node-local scratch (use_local_scratch)\n")
        f.write(f'YARP_NFS_DIR="{self.scratch_dir}"\n')
        f.write(f'YARP_LOCAL_DIR=$(mktemp -d "{self.job_manager.local_scratch_dir}/yarp.XXXXXX") || exit 1\n')
        f.write('yarp_copy_back() { rsync -a "$YARP_LOCAL_DIR"/ "$YARP_NFS_DIR"/; }\n')
        f.write('trap \'yarp_copy_back && rm -rf "$YARP_LOCAL_DIR"\' EXIT\n')
        f.write("trap 'yarp_copy_back' USR1\n")
        f.write("trap 'exit 140' USR2\n")
        f.write('cp -a "$YARP_NFS_DIR"/. "$YARP_LOCAL_DIR"/ || exit 1\n')
        f.write("yarp_bg() {\n")
        f.write('    "$@" &\n')
        f.write("    local pid=$!\n")
        f.write("    while kill -0 $pid 2>/dev/null; do wait $pid; done\n")
        f.write("    wait $pid\n")
        f.write("}\n\n")

    # --- Pre-flight Check (Overridden by Task classes) ---
    def has_prerequisites(self) -> bool:
        return True

    # --- The 5 Lifecycle Methods (Overridden by Software classes) ---
    def generate_input(self):
        """
        1. Write input files (e.g., .xyz, .inp) based on self.rxn geometries.

        Raise `CalculatorInputError` if a usable input cannot be built; the
        caller discards the reaction cleanly. Any other exception is a bug and
        will take down the whole `yarp-progress` invocation.
        """
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

