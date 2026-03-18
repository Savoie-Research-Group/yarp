"""
Definition of the YARP Calculator base classes and software-specific implementations.
"""
import shutil
from pathlib import Path

from yarp.reaction.external.crest import CrestConfCalculator

# =====================================================================
#   THE ROOT BASE CLASS
# =====================================================================
class AsyncYarpCalculator:
    """
    Base class defining the asynchronous lifecycle interface required by progress_yarp.py.
    """
    def __init__(self, task_def, rxn_obj):
        self.task_def = task_def
        self.rxn = rxn_obj
        self.config = task_def.config
        self.scratch_dir = None

    def set_scratch_dir(self, path: Path):
        self.scratch_dir = path
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

    def get_container_prefix(self, image_name: str) -> str:
        """
        The universal toggle for container execution.
        Maps the scratch directory to /work inside the container.
        """
        if self.container_runner == "docker":
            # --rm removes the container after it finishes
            # -v mounts the host scratch dir to /work
            # -u $(id -u):$(id -g) ensures files aren't created as root (optional but good practice)
            return f"docker run --rm -v {self.scratch_dir}:/work {image_name}"
            
        elif self.container_runner == "apptainer":
            # Apptainer automatically mounts the current working directory, 
            # but explicit binding is safer.
            # Assuming the user has downloaded the .sif file to a known location, or pulls it.
            # E.g., docker://ghcr.io/username/yarp_crest:latest
            return f"apptainer exec --bind {self.scratch_dir}:/work {image_name}.sif"
            
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


# =====================================================================
#    TASK BASE CLASSES (Handles pre-flight checks)
# =====================================================================
class MLPredictTask(AsyncYarpCalculator):
    def has_prerequisites(self) -> bool:
        # ML usually just needs the 2D graph (SMILES/InChI), which is guaranteed to exist.
        return True

class ConfTask(AsyncYarpCalculator):
    def has_prerequisites(self) -> bool:
        return True # Just needs the base yarpecule graph

class TSGuessTask(AsyncYarpCalculator):
    def has_prerequisites(self) -> bool:
        if not self.rxn.reactant.conformers or not self.rxn.product.conformers:
            print(f"[{self.rxn.hash}] Missing R/P conformers for TS Guess.")
            return False
        return True

class MinOptTask(AsyncYarpCalculator):
    def has_prerequisites(self) -> bool:
        # Needs at least an initial unoptimized geometry
        return True 

class TSOptTask(AsyncYarpCalculator):
    def has_prerequisites(self) -> bool:
        if not getattr(self.rxn, 'ts_geom', None):
            print(f"[{self.rxn.hash}] Missing TS guess for TS Opt.")
            return False
        return True

class IRCValTask(AsyncYarpCalculator):
    def has_prerequisites(self) -> bool:
        if not getattr(self.rxn, 'ts_geom', None) or not self.rxn.ts_geom.get("optimized"):
            print(f"[{self.rxn.hash}] Missing optimized TS geometry for IRC.")
            return False
        return True


# =====================================================================
#    SOFTWARE-SPECIFIC IMPLEMENTATIONS (The "Meat and Potatoes")
# =====================================================================

# --- TASK 1: ML PREDICT ---
class EgatMLPredict(MLPredictTask):
    def generate_input(self):
        # Write SMILES to a text file for EGAT to read
        pass
    def write_submission_script(self) -> Path:
        # Write script calling the EGAT container
        pass
    def check_output(self) -> bool:
        return (self.scratch_dir / "egat_results.csv").exists()
    def scrape_data(self):
        # Read CSV, assign self.rxn.barrier['egat'] = value
        pass
    def cleanup(self):
        # EGAT is lightweight, maybe just delete the folder
        shutil.rmtree(self.scratch_dir)


# --- TASK 3: TS GUESS ---
class PysisyphusTSGuessCalculator(TSGuessTask):
    def generate_input(self):
        # Write reactant.xyz, product.xyz, and pysisyphus.ini
        pass
    def write_submission_script(self) -> Path:
        # Write script calling pysisyphus container
        pass
    def check_output(self) -> bool:
        # Look for the GSM output string or the specific TS guess xyz
        pass
    def scrape_data(self):
        # Extract TS guess, save to self.rxn.ts_geom['ts_guess_xtb']
        pass
    def cleanup(self):
        pass


# --- TASK 4/5: OPTIMIZATIONS (ORCA Example) ---
class OrcaOptCalculator(MinOptTask):
    """Can be used for Min Opt (or subclassed further for TS Opt)."""
    def generate_input(self):
        # Write ORCA .inp file using self.rxn conformer coords
        pass
    def write_submission_script(self) -> Path:
        # Write script calling ORCA container
        pass
    def check_output(self) -> bool:
        # Read orca.out and check for "ORCA TERMINATED NORMALLY"
        out_file = self.scratch_dir / "orca.out"
        if not out_file.exists(): return False
        with open(out_file, 'r') as f:
            return "ORCA TERMINATED NORMALLY" in f.read()
    def scrape_data(self):
        # Extract optimized geometry and energy
        pass
    def cleanup(self):
        # Keep .inp, .out, .xyz. Delete .tmp, .densities, etc.
        pass


# =====================================================================
#    THE FACTORY ROUTER
# =====================================================================
def get_calculator(task_def, rxn_obj, container_runner="docker") -> AsyncYarpCalculator:
    """
    Routes the task to the specific combination of Task Type and Software.
    """
    t_type = task_def.task_type
    software = getattr(task_def.config, 'software', 'unknown').lower()

    # Task 1: ML Predict
    if t_type == "ml_predict":
        if software == "egat" or software == "unknown": # Fallback for now
            return EgatMLPredict(task_def, rxn_obj)

    # Task 2: Conformers
    elif t_type in ["reactant_conformer", "product_conformer"]:
        if software == "crest":
            return CrestConfCalculator(task_def, rxn_obj, container_runner)

    # Task 3: TS Guess
    elif t_type == "gsm":
        if software == "pysisyphus":
            return PysisyphusTSGuessCalculator(task_def, rxn_obj)

    # Tasks 4 & 5: Optimizations
    elif t_type in ["reactant_optimization", "product_optimization", "transition_state_optimization"]:
        if software == "orca":
            return OrcaOptCalculator(task_def, rxn_obj)
        elif software == "pysisyphus":
            # Would return a PysisyphusOptCalculator(...)
            pass

    # Task 6: IRC Validation
    elif t_type == "irc_validation":
        # Would return your IRC calculator
        pass

    raise ValueError(f"No calculator implemented for Task: '{t_type}' with Software: '{software}'")