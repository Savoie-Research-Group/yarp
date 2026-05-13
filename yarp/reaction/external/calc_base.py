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

    def get_container_prefix(self, image_name: str, work_dir: str) -> str:
        """
        The universal toggle for container execution.
        Maps the scratch directory to /work inside the container.
        Automatically checks for and pulls missing images.
        """
        if self.job_manager.container == "docker":
            # 1. Check if the image exists locally
            inspect_cmd = subprocess.run(
                ["docker", "image", "inspect", image_name], 
                capture_output=True
            )
            
            # 2. If returncode is non-zero, it doesn't exist. Pull it!
            if inspect_cmd.returncode != 0:
                print(f"Docker image '{image_name}' not found locally. Pulling from registry...")
                subprocess.run(["docker", "pull", "--platform", "linux/amd64", image_name], check=True)

            return f"docker run --platform linux/amd64 --rm -v {work_dir}:/work -w /work {image_name}"

        elif self.job_manager.container == "apptainer":
            # Sanitize the image name so it works as a safe, flat filename
            # e.g., "erm42/yarp:crest" -> "erm42_yarp_crest.sif"
            safe_filename = image_name.replace("/", "_").replace(":", "_") + ".sif"
            sif_path = Path(self.job_manager.sif_location) / safe_filename

            # 1. Check if the .sif file exists on disk
            if not sif_path.exists():
                print(f"Apptainer image not found at {sif_path}. Pulling from Docker Hub...")
                
                # Ensure the target directory actually exists before pulling
                sif_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 2. Pull the docker image and convert it to a .sif file
                subprocess.run(
                    ["apptainer", "pull", str(sif_path), f"docker://{image_name}"], 
                    check=True
                )

            return f"apptainer exec --bind {work_dir}:/work --pwd /work {sif_path}"

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

