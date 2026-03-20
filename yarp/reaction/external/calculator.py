"""
Definition of the YARP Calculator base classes and software-specific implementations.
"""
import shutil
from pathlib import Path

from yarp.yarpecule.input_parsers import xyz_parse
from yarp.reaction.conformer import conformer
from yarp.reaction.conf_bias_select import select_gsm_pairs

# from yarp.reaction.external.crest import CrestConfCalculator

# =====================================================================
#   THE ROOT BASE CLASS
# =====================================================================
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
        # ERM: Ehhhhh is it though? We should put an actual check in this slot eventually...
        return True

class ConfTask(AsyncYarpCalculator):
    def has_prerequisites(self) -> bool:
        if not self.rxn.reactant.conformers.get('initial_geom') or not self.rxn.product.conformers.get('initial_geom'):
            return False
        return True

class TSGuessTask(AsyncYarpCalculator):
    def has_prerequisites(self) -> bool:
        r_node = self.rxn.reactant
        p_node = self.rxn.product
        if not r_node.conformers or not p_node.conformers:
            return False

        r_keys = r_node.conformers.keys()
        r_match = False
        for rk in r_keys:
            if 'conf_gen' in rk and r_node.conformers[rk].geo is not None: 
                r_match = True
        p_keys = p_node.conformers.keys()
        p_match = False
        for pk in p_keys:
            if 'conf_gen' in pk and p_node.conformers[pk].geo is not None:
                p_match = True

        if not r_match or not p_match:
            return False

        return True

class MinOptTask(AsyncYarpCalculator):
    def has_prerequisites(self) -> bool:
        # Needs at least an initial unoptimized geometry
        return True 

class TSOptTask(AsyncYarpCalculator):
    def has_prerequisites(self) -> bool:
        if not getattr(self.rxn, 'ts_geom', None):
            return False
        return True

class IRCValTask(AsyncYarpCalculator):
    def has_prerequisites(self) -> bool:
        if not getattr(self.rxn, 'ts_geom', None) or not self.rxn.ts_geom.get("optimized"):
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

class CrestConfCalculator(ConfTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_name = "yarp_crest:latest" # Change to ghcr.io/url later???
        self.xyz_file = "input.xyz"

        # Determine if we are working on the reactant or the product
        if "reactant" in self.task_def.task_type:
            self.target_species = self.rxn.reactant
        else:
            self.target_species = self.rxn.product

    def generate_input(self):
        """Write the initial 3D geometry for CREST to start from."""
        input_xyz_path = self.scratch_dir / self.xyz_file
        with open(input_xyz_path, "w") as f:
            # Assuming yarpecule has a method to get a basic 3D string
            # (e.g., generated via RDKit/ETKDG during initialization)
            f.write(self.target_species.conformers.get('initial_geom').to_xyz_string())

    def write_submission_script(self) -> Path:
        """Write the bash script that the JobManager will execute."""
        script_path = self.scratch_dir / "run_crest_cmd.sh"

        # Construct the core command
        prefix = self.get_container_prefix(self.image_name)
        crest_cmd = self._get_crest_command()
        full_command = f"{prefix} {crest_cmd}"

        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"cd {self.scratch_dir}\n")
            # Pipe output to a log file so it doesn't flood the local terminal
            f.write(f"{full_command} > crest_run.log 2> crest_run.err\n")

        # Make the script executable (important for LocalJobManager)
        script_path.chmod(0x755)

        return script_path

    def check_output(self) -> bool:
        """Verify CREST actually finished and produced conformers."""
        # ERM: Should we add a check here to make sure there are at minimum n_conf available?
        xyz_file_name = self.scratch_dir / "crest_conformers.xyz"
        ene_file_name = self.scratch_dir / "crest.energies"

        xyz_exists = xyz_file_name.exists()
        energies_exists = ene_file_name.exists()

        termination_msg_exists = False
        outfile = self.scratch_dir / "crest_run.log"
        if outfile.exists():
            try: 
                lines = open(outfile, 'r', encoding="utf-8").readlines()
                for n_line, line in enumerate(reversed(lines)):
                    if 'CREST terminated normally.' in line:
                        termination_msg_exists = True
            except:
                termination_msg_exists = False

        return xyz_exists and energies_exists and termination_msg_exists

    def scrape_data(self):
        """Parse the XYZ and update self.target_species."""
        confs = self._get_all_conformers()
        for conf in confs:
            conf['lot'] = self.config.lot
            conf['software'] = 'crest'
            conf_obj = conformer(calc_type='conf_gen', calc_data=conf)
            self.target_species.conformers[conf_obj.type] = conf_obj

    def cleanup(self):
        """Delete files generated by CREST."""
        print(f"  * [{self.rxn.hash}] Data cleanup for {self.task_def.task_type} not yet implemented.")
        # for file in self.scratch_dir.iterdir():
        #     keep_files = ["crest_conformers.xyz", "crest.energies", "submit.sh", "crest_run.log", self.xyz_file]
        #     if file.name not in keep_files:
        #         if file.is_file(): file.unlink()
        #         elif file.is_dir(): shutil.rmtree(file)

    def _get_crest_command(self):

        # basic command (ERM: no way to set memory_per_cpu in CREST????)
        cmd = f"crest {self.xyz_file} --{self.config.lot} -nozs -T {self.config.n_cpus}"

        # molecular descriptors
        cmd += f" -chrg {self.config.charge} -uhf {self.config.multiplicity}"

        # conformer generation thresholds (ERM: expand this later, if needed)
        # ERM: no current way to cap CREST outputs at a set number of generated conformers!
        # You can damp down via adjusting the energy window threshold, but that's it
        cmd += f" - ewin {self.config.energy_window}"

        # implicit solvation models
        alpb_solv = set(
            'acetone', 'acetonitrile', 'aniline', 'benzaldehyde', 'benzene',
            'ch2cl2', 'chcl3', 'cs2', 'dioxane', 'dmf', 'dmso', 'ether',
            'ethylacetate', 'furane', 'hexandecane', 'hexane', 'methanol',
            'nitromethane', 'octanol', 'woctanol', 'phenol', 'toluene',
            'thf', 'water'
        )
        gbsa_solv = set(
            'acetone', 'acetonitrile', 'aniline', 'benzaldehyde',
            'CH2Cl2', 'CHCl3', 'CS2', 'DMSO', 'ether', 'H2O', 'methanol',
            'THF', 'toluene'
        )
        if self.config.solvent is not None:
            model = self.config.solvent.get('model', None)
            solv = self.config.solvent.get('solvent', None)
            if model == 'alpb' and solv in alpb_solv:
                cmd += f" --{model} {solv}"
            elif model == 'gbsa' and solv in gbsa_solv:
                cmd += f" --{model} {solv}"

        return cmd

    def _get_all_conformers(self):
        """
        Get the entire set of geometry (and elements) from crest output files.
        Returns a dictionary for each conformer with the geometry, elements,
        relative energy ranking, and total energy in Eh
        """
        xyz_file_name = self.scratch_dir / "crest_conformers.xyz"

        confs=[]
        elements, geometries = xyz_parse(xyz_file_name, multiple=True)
        for count_i, i in enumerate(elements):
            conf = {
                'conf_rank': count_i,
                'elements': elements[count_i],
                'geometry': geometries[count_i]
            }
            confs.append(conf)

        return confs


# --- TASK 3: TS GUESS ---
class PysisyphusTSGuessCalculator(TSGuessTask):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_name = "yarp_pysisyphus:latest" # Future GHCR link
        self.n_pairs = getattr(self.config, 'n_conf', 1)
        self.pairs_to_run = []

    def generate_input(self):
        """
        Runs the Joint Opt + ML Selection, then writes the files 
        required for the Pysisyphus GSM run.
        """
        print(f"  * [{self.rxn.hash}] Selecting {self.n_pairs} conformer pairs for GSM...")
        self.pairs_to_run = select_gsm_pairs(self.rxn, self.config)
        
        # Write inputs for each pair
        for i, pair in enumerate(self.pairs_to_run):
            idx = i + 1
            r_xyz_path = self.scratch_dir / f"reactant_{idx}.xyz"
            p_xyz_path = self.scratch_dir / f"product_{idx}.xyz"
            ini_path = self.scratch_dir / f"gsm_{idx}.ini"
            
            # Write XYZs
            with open(r_xyz_path, "w") as f:
                f.write(pair["r_conf"].to_xyz_string())
            with open(p_xyz_path, "w") as f:
                f.write(pair["p_conf"].to_xyz_string())
                
            # Write a standard Pysisyphus INI file (template example)
            # You can adapt this to whatever specific Pysisyphus parameters you use
            lot = getattr(self.config, 'lot', 'xtb')
            ini_content = f"""[geometry]
type = string
r_xyz = reactant_{idx}.xyz
p_xyz = product_{idx}.xyz

[calculator]
type = {lot}

[opt]
type = gsm
max_cycles = 100
"""
            with open(ini_path, "w") as f:
                f.write(ini_content)

    def write_submission_script(self) -> Path:
        """Writes a bash script that executes all N pairs sequentially."""
        script_path = self.scratch_dir / "submit.sh"
        prefix = self.get_container_prefix(self.image_name)
        
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"cd {self.scratch_dir}\n\n")
            
            # Loop over however many pairs were actually generated
            for i in range(len(self.pairs_to_run)):
                idx = i + 1
                cmd = f"pysisyphus gsm_{idx}.ini"
                f.write(f"echo 'Running GSM for Pair {idx}...'\n")
                f.write(f"{prefix} {cmd} > gsm_{idx}.log 2>&1\n")
                
                # Pysisyphus usually generates a specific output file like 'gsm_ts_guess.xyz'. 
                # We rename it so it isn't overwritten by the next loop iteration.
                f.write(f"if [ -f \"gsm_ts_guess.xyz\" ]; then\n")
                f.write(f"    mv gsm_ts_guess.xyz ts_guess_{idx}.xyz\n")
                f.write(f"fi\n\n")
                
        script_path.chmod(0x755)
        return script_path

    def check_output(self) -> bool:
        """
        Returns True if AT LEAST ONE of the N pairs generated a TS guess.
        We don't want to fail the whole reaction just because 1 out of 3 strings broke.
        """
        for i in range(len(self.pairs_to_run)):
            if (self.scratch_dir / f"ts_guess_{i+1}.xyz").exists():
                return True
        return False

    def scrape_data(self):
        """Scrape all successful TS guesses into the reaction object."""
        pass
        # self.rxn.ts_geom = {} # Ensure dict exists
        
        # for i, pair in enumerate(self.pairs_to_run):
        #     idx = i + 1
        #     ts_file = self.scratch_dir / f"ts_guess_{idx}.xyz"
            
        #     if ts_file.exists():
        #         # Parse the XYZ and create a conformer object
        #         # ts_calc_data = parse_xyz(ts_file) 
        #         # ts_conf = conformer(calc_type='conf_gen', calc_data=ts_calc_data)
                
        #         # Save it with a unique key
        #         key = f"tsguess_pysis_{idx}"
        #         # self.rxn.ts_geom[key] = ts_conf
        #         print(f"[{self.rxn.hash}] Scraped TS guess from pair {idx} into '{key}'.")

    def cleanup(self):
        """Clean up but keep logs and final TS xyz files."""
        pass
        # for file in self.scratch_dir.iterdir():
        #     if file.name.startswith("ts_guess_") or file.name.startswith("gsm_") or file.name == "submit.sh":
        #         continue
        #     if file.is_file(): file.unlink()
        #     elif file.is_dir(): shutil.rmtree(file)


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