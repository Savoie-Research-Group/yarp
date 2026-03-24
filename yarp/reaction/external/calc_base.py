from pathlib import Path
import shutil

class AsyncYarpCalculator:
    """
    Base class defining the asynchronous lifecycle interface required by progress_yarp.py.
    """
    def __init__(self, task_def, rxn_obj, container_runner="docker"):
        self.task_def = task_def
        self.rxn = rxn_obj
        self.config = task_def.config
        self.scratch_dir = None
        self.container_runner = container_runner

    def set_scratch_dir(self, path: Path):
        self.scratch_dir = path
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

    def get_container_prefix(self, image_name: str, work_dir: str) -> str:
        """
        The universal toggle for container execution.
        Maps the scratch directory to /work inside the container.
        """
        if self.container_runner == "docker":
            # --rm removes the container after it finishes
            # -v mounts the host scratch dir to /work
            # -u $(id -u):$(id -g) ensures files aren't created as root (optional but good practice)
            return f"docker run --rm -v {work_dir}:/work {image_name}"
            
        elif self.container_runner == "apptainer":
            # Apptainer automatically mounts the current working directory, 
            # but explicit binding is safer.
            # Assuming the user has downloaded the .sif file to a known location, or pulls it.
            # E.g., docker://ghcr.io/username/yarp_crest:latest
            return f"apptainer exec --bind {work_dir}:/work {image_name}.sif"
            
        else:
            raise ValueError(f"Unsupported container runner: {self.container_runner}")

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

