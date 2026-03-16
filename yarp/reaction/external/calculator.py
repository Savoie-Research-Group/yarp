"""
Definition of the YARP Calculator base class and software-specific subclasses.
"""
import shutil
import subprocess
from pathlib import Path

class YarpCalculator:
    """
    Base class for managing external software calculations in YARP.
    """
    def __init__(self, job_name, scratch_base="./SCRATCH", runner="docker", n_threads=1):
        self.job_name = job_name
        self.job_dir = Path(scratch_base).resolve() / job_name
        self.runner = runner
        self.n_threads = n_threads
        
        # Will hold the parsed results to be passed back to YARP
        self.results = None 

    def execute(self):
        """
        The main orchestrator method. This executes the 5 stages of a calculation's lifecycle.
        """
        self.job_dir.mkdir(parents=True, exist_ok=True)
        print(f"Starting job: {self.job_name} in {self.job_dir}")

        try:
            # Step 1: Generate Input
            self.generate_input()
            
            # Step 2: Run Container
            self.run_container()
            
            return None

        except Exception as e:
            print(f"ERROR: Calculation {self.job_name} failed!")
            self._write_crash_log(str(e))
            print(f"Files retained in {self.job_dir} for debugging.")
            return None

    def _write_crash_log(self, error_msg):
        with open(self.job_dir / "crash_log.txt", "w") as f:
            f.write(f"YARP Calculator Exception:\n{error_msg}")

    # ==========================================
    # Abstract Methods (To be implemented by subclasses)
    # ==========================================
    def generate_input(self): pass
    def run_container(self): pass
    def validate_outputs(self): pass
    def extract_data(self): pass
    
    def clean_up(self):
        """Step 5: Default cleanup behavior (can be overridden)."""
        print(f"Calculation successful. Cleaning up {self.job_dir}...")
        shutil.rmtree(self.job_dir)


class CrestCalculator(YarpCalculator):
    """
    Subclass specifically for handling CREST conformer generation.
    """
    def __init__(self, init_xyz, job_id, lot="gfn2", **kwargs):
        self.lot = lot
        self.xyz_str = init_xyz
        self.xyz_file = "input.xyz"
        
        # Initialize the base class
        job_name = f"crest_{lot}_{job_id}"
        super().__init__(job_name=job_name, **kwargs)

    def generate_input(self):
        """Step 1: Write the specific input files CREST needs."""
        input_xyz_path = self.job_dir / self.xyz_file
        with open(input_xyz_path, "w") as f:
             f.write(self.xyz_str)

    def run_container(self):
        """Step 2: Construct and execute the container command."""
        if self.runner == "docker":
            cmd = [
                "docker", "run", "--rm", 
                "-v", f"{self.job_dir}:/work", 
                "yarp_crest:latest", 
                "crest", self.xyz_file, f"--{self.lot}", "-T", str(self.n_threads)
            ]
        elif self.runner == "apptainer":
            cmd = [
                "apptainer", "exec", 
                "--bind", f"{self.job_dir}:/work", 
                "yarp_crest.sif", 
                "crest", self.xyz_file, f"--{self.lot}", "-T", str(self.n_threads)
            ]
        else:
            raise ValueError(f"Unknown runner: {self.runner}")

        # Execute and check for container-level errors
        result = subprocess.run(cmd, cwd=self.job_dir, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"CREST failed:\n{result.stderr}")

    def validate_outputs(self):
        """Step 4: Check if CREST successfully generated what we need."""
        self.out_xyz = self.job_dir / "crest_conformers.xyz"
        self.out_ene = self.job_dir / "crest.energies"
        
        if not self.out_xyz.exists() or not self.out_ene.exists():
            raise FileNotFoundError("CREST completed, but required output files are missing.")

    def extract_data(self):
        """Step 3: Read the files and package the data."""
        # Using a placeholder for your actual parsing logic
        # elements, geometries = xyz_parse(self.out_xyz, multiple=True)
        
        # Let's assume we package the parsed data into a neat dictionary or list 
        # to hand back to the state object
        self.results = {
            "software": "crest",
            "lot": self.lot,
            "geometries": [], # Insert parsed geometries here
            "energies": []    # Insert parsed energies here
        }