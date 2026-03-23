"""
Definition of the YARP Calculator base classes and software-specific implementations.
"""
import os
import shutil
from pathlib import Path

from yarp.yarpecule.input_parsers import xyz_parse
from yarp.reaction.conformer import conformer
from yarp.reaction.conf_sampling.select_pairs import select_gsm_pairs

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
        r_node = self.rxn.reactant
        p_node = self.rxn.product
        if not r_node.conformers or not p_node.conformers:
            return False

        r_keys = r_node.conformers.keys()
        r_match = False
        for rk in r_keys:
            if r_node.conformers[rk].geo is not None: 
                r_match = True

        p_keys = p_node.conformers.keys()
        p_match = False
        for pk in p_keys:
            if p_node.conformers[pk].geo is not None:
                p_match = True

        return r_match and p_match 

class TSOptTask(AsyncYarpCalculator):
    def has_prerequisites(self) -> bool:

        ts_keys = self.rxn.ts_geom.keys()
        ts_match = False
        for k in ts_keys:
            if 'ts_guess' in k and self.rxn.ts_geom[k].geo is not None: 
                ts_match = True
        
        return ts_match

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
        prefix = self.get_container_prefix(self.image_name, self.scratch_dir)
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
        cmd += f" --chrg {self.config.charge} --uhf {self.config.n_unpaired_electrons}"

        # conformer generation thresholds (ERM: expand this later, if needed)
        # ERM: no current way to cap CREST outputs at a set number of generated conformers!
        # You can damp down via adjusting the energy window threshold, but that's it
        # cmd += f" -ewin {self.config.energy_window}"

        # implicit solvation models
        alpb_solv = set([
            'acetone', 'acetonitrile', 'aniline', 'benzaldehyde', 'benzene',
            'ch2cl2', 'chcl3', 'cs2', 'dioxane', 'dmf', 'dmso', 'ether',
            'ethylacetate', 'furane', 'hexandecane', 'hexane', 'methanol',
            'nitromethane', 'octanol', 'woctanol', 'phenol', 'toluene',
            'thf', 'water'
        ])
        gbsa_solv = set([
            'acetone', 'acetonitrile', 'aniline', 'benzaldehyde',
            'CH2Cl2', 'CHCl3', 'CS2', 'DMSO', 'ether', 'H2O', 'methanol',
            'THF', 'toluene'
        ])
        if self.config.solvent is not None:
            model = self.config.solvent.get('model', '')
            solv = self.config.solvent.get('solvent', '')
            if model == 'alpb' and solv.lower() in alpb_solv:
                cmd += f" --{model} {solv}"
            elif model == 'gbsa' and solv.lower() in gbsa_solv:
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


class PysisyphusTSGuessCalculator(TSGuessTask):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_name = "yarp_pysisyphus:latest" # Future GHCR link
        self.n_pairs = self.config.n_conf
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
            pair_dir = self.scratch_dir / f"gsm_run{idx}"
            os.makedirs(pair_dir, exist_ok=True)
            r_xyz_path = pair_dir / f"reactant_{idx}.xyz"
            p_xyz_path = pair_dir / f"product_{idx}.xyz"
            inp_path = pair_dir / f"gsm_{idx}_input.yaml"
            
            # Write XYZs
            with open(r_xyz_path, "w") as f:
                f.write(pair["r_conf"].to_xyz_string())
            with open(p_xyz_path, "w") as f:
                f.write(pair["p_conf"].to_xyz_string())
            
            # Write a Pysisyphus input file for GSM
            self._write_pysis_gsm_input(inp_path, f"reactant_{idx}.xyz", f"product_{idx}.xyz")

    def write_submission_script(self):
        """
        Creates a sequential runner script for inside the container, 
        and a host submission script to launch the container.
        """
        # ---------------------------------------------------------
        # 1. The INNER Script (Runs inside the Docker container)
        # ---------------------------------------------------------
        inner_script_path = self.scratch_dir / "run_pysis_inner.sh"
        
        with open(inner_script_path, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("echo 'Starting YARP serial GSM execution...'\n\n")
            
            for i in range(1, len(self.pairs_to_run) + 1):
                f.write(f"echo '--- Running GSM pair {i} ---'\n")
                # The container mounts self.scratch_dir to /work
                f.write(f"cd /work/gsm_run{i}\n") 
                
                # Execute pysis within the specific folder, saving logs locally
                f.write(f"pysis gsm_{i}_input.yaml > gsm_{i}.log 2> gsm_{i}.err\n")
                f.write(f"echo '--- Finished GSM pair {i} ---'\n\n")
                
        # Make executable
        inner_script_path.chmod(0o755)

        # ---------------------------------------------------------
        # 2. The OUTER Script (Runs on the Host / Job Manager)
        # ---------------------------------------------------------
        submit_script_path = self.scratch_dir / "submit_gsm.sh"
        
        # Pulls the docker prefix (e.g., 'docker run --rm -v /scratch:/work -u UID:GID yarp_pysisyphus')
        docker_cmd_prefix = self.get_container_prefix("yarp_pysisyphus", str(self.scratch_dir))
        
        with open(submit_script_path, "w") as f:
            f.write("#!/bin/bash\n")
            # Launch the container and tell it to run the inner script
            f.write(f"{docker_cmd_prefix} bash /work/run_pysis_inner.sh\n")
            
        submit_script_path.chmod(0o755)
        
        # Return the outer script for the Job Manager to execute
        return str(submit_script_path)

    def check_output(self) -> bool:
        """
        Returns True if AT LEAST ONE of the N pairs generated a TS guess.
        """
        num_runs = self._get_num_runs()
        if num_runs == 0:
            return False # No runs found at all!

        one_successful = False
        
        for i in range(1, num_runs + 1):
            run_dir = self.scratch_dir / f"gsm_run{i}"
            log_file = run_dir / f"gsm_{i}.log"
            trj_file = run_dir / "final_geometries.trj"
            xyz_file = run_dir / "splined_hei.xyz"

            # 1. File existence check
            if not (log_file.exists() and trj_file.exists() and xyz_file.exists()):
                print(f"   * Run {i} failed: Missing expected output files.")
                continue

            # 2. Log file termination check
            with open(log_file, "r") as f:
                log_text = f.read()

            if "Wrote splined HEI" not in log_text or "pysisyphus run took" not in log_text:
                print(f"   * Run {i} failed: Did not find successful termination message in log.")
                continue

            # If it passes all checks, at least one run succeeded!
            one_successful = True
        
        # We only care if at least one succeeded
        return one_successful

    def scrape_data(self) -> bool:
        """
        Validates the Pysisyphus GSM run and extracts the geometries into 
        the reaction object. 
        """
        num_runs = self._get_num_runs()
        
        for i in range(1, num_runs + 1):
            run_dir = self.scratch_dir / f"gsm_run{i}"
            log_file = run_dir / f"gsm_{i}.log"
            trj_file = run_dir / "final_geometries.trj"
            xyz_file = run_dir / "splined_hei.xyz"

            # --- SAFEGUARD: Skip failed runs during partial success ---
            if not (log_file.exists() and trj_file.exists() and xyz_file.exists()):
                continue
            with open(log_file, "r") as f:
                if "Wrote splined HEI" not in f.read():
                    continue
            # ---------------------------------------------------------

            # Parse and store the splined TS guess
            ts_elements, ts_geo = xyz_parse(xyz_file, multiple=False)
            
            ts_conf = conformer()
            ts_conf.elements = ts_elements
            ts_conf.geo = ts_geo
            ts_conf.lot = self.config.gsm_lot
            ts_conf.software = self.config.software
            ts_conf.type = f"ts_guess_{i}_{ts_conf.lot}_{ts_conf.software}"
            
            # Store in the reaction object's TS dictionary
            self.rxn.ts_geom[ts_conf.type] = ts_conf

            # Parse the final geometries trajectory for Reactant/Product pairs
            trj_elements, trj_geo = xyz_parse(trj_file, multiple=True)
            
            if len(trj_elements) >= 2:
                # Store Reactant guess (first frame)
                r_conf = conformer()
                r_conf.elements = trj_elements[0]
                r_conf.geo = trj_geo[0]
                r_conf.lot = self.config.gsm_lot
                r_conf.software = self.config.software
                r_conf.type = f"guess_conf_{i}_{r_conf.lot}_{r_conf.software}"
                
                self.rxn.reactant.conformers[r_conf.type] = r_conf
                
                # Store Product guess (last frame)
                p_conf = conformer()
                p_conf.elements = trj_elements[-1]
                p_conf.geo = trj_geo[-1]
                p_conf.lot = self.config.gsm_lot
                p_conf.software = self.config.software
                p_conf.type = f"guess_conf_{i}_{p_conf.lot}_{p_conf.software}"
                
                self.rxn.product.conformers[p_conf.type] = p_conf

        print(f'Reaction TS guess keys:\n {list(self.rxn.ts_geom.keys())}')
        print(f'Reactant state conformer keys:\n {list(self.rxn.reactant.conformers.keys())}')
        print(f'Product state conformer keys:\n {list(self.rxn.product.conformers.keys())}')
        
        return True

    def cleanup(self):
        """Clean up but keep logs and final TS xyz files."""
        pass
        # for file in self.scratch_dir.iterdir():
        #     if file.name.startswith("ts_guess_") or file.name.startswith("gsm_") or file.name == "submit.sh":
        #         continue
        #     if file.is_file(): file.unlink()
        #     elif file.is_dir(): shutil.rmtree(file)

    def _write_pysis_gsm_input(self, input_path, r_xyz_path, p_xyz_path):
        # Make sure lot is xTB (ERM: We'll make this more robust later! Hopefully!)
        lot = self.config.gsm_lot.lower()
        assert (self.config.gsm_lot.lower() == 'xtb'), "GSM with Pysisyphus is xTB or bust right now, friend..."

        # Write the file! Yay, YAML friend!
        with open(input_path, 'a') as f:
            # set geom block
            input_geo = [r_xyz_path, p_xyz_path]
            f.write(f'geom:\n type: cart\n fn: {input_geo}\n')

            # set calc block
            # ERM: I left out the option for solvent,
            # because what I saw in classy YARP didn't make sense to me...
            f.write(f'calc:\n type: {lot}\n pal: {self.config.n_cpus}\n mem: {self.config.mem_per_cpu}\n charge: {self.config.charge}\n mult: {self.config.multiplicity}\n')

            # set cos block
            f.write(f'cos:\n type: gs\n max_nodes: {self.config.max_gsm_nodes}\n climb: True\n climb_rms: 0.005\n climb_lanczos: False\n reparam_check: rms\n reparam_every: 1\n reparam_every_full: 1\n')

            # set opt block
            f.write(f'opt:\n type: string\n stop_in_when_full: -1\n align: True\n scale_step: global\n')

    def _get_num_runs(self) -> int:
            """
            Dynamically counts the number of gsm_run subdirectories.
            Crucial for stateless execution where self.pairs_to_run is lost between runs.
            """
            # Finds all directories matching 'gsm_run*' in the scratch folder
            run_dirs = list(self.scratch_dir.glob("gsm_run*"))
            return len(run_dirs)

class PysisyphusMinOptCalculator(MinOptTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_name = "yarp_pysisyphus:latest" # Future GHCR link

    def generate_input(self):
        inp_path = self.scratch_dir / "min_opt.yaml"
        xyz_file = "initial_geom.xyz" # need to figure out how to set this!!!
        self._write_pysis_rp_opt_input(inp_path, xyz_file)

    def write_submission_script(self) -> Path:
        """Write the bash script that the JobManager will execute."""
        script_path = self.scratch_dir / "run_pysis_rpopt.sh"

        # Construct the core command
        prefix = self.get_container_prefix(self.image_name, self.scratch_dir)
        pysis_cmd = "pysis min_opt.yaml > min_opt.log 2> min_opt.err"
        full_command = f"{prefix} {pysis_cmd}"

        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"cd {self.scratch_dir}\n")
            f.write(f"{full_command}\n")

        # Make the script executable (important for LocalJobManager)
        script_path.chmod(0x755)

        return script_path

    def check_output(self) -> bool:
        pass
    def scrape_data(self):
        # Extract optimized geometry and energy
        pass
    def cleanup(self):
        # Keep .inp, .out, .xyz. Delete .tmp, .densities, etc.
        pass

    def _write_pysis_rp_opt_input(self, input_path, input_geo_xyz):
        # Make sure lot is xTB (ERM: We'll make this more robust later! Hopefully!)
        lot = self.config.lot.lower()
        assert (lot == 'xtb'), "Calculations with Pysisyphus are xTB or bust right now, friend..."

        # Write the file! Yay, YAML friend!
        with open(input_path, 'a') as f:
            # set geom block
            f.write(f'geom:\n type: cart\n fn: {input_geo_xyz}\n')

            # set calc block
            # ERM: I left out the option for solvent,
            # because what I saw in classy YARP didn't make sense to me...
            f.write(f'calc:\n type: {lot}\n pal: {self.config.n_cpus}\n mem: {self.config.mem_per_cpu}\n charge: {self.config.charge}\n mult: {self.config.multiplicity}\n')

            # set opt block
            if self.config.do_hess:
                f.write(f'opt:\n type: rfo\n max_cycles: {self.config.max_cycles}\n overachieve_factor: 3\n hessian_recalc: {self.config.hessian_recalc}\n do_hess: True\n')
            else:
                f.write(f'opt:\n type: rfo\n max_cycles: {self.config.max_cycles}\n overachieve_factor: 3\n')


class PysisyphusTSOptCalculator(TSOptTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_name = "yarp_pysisyphus:latest" # Future GHCR link

    def generate_input(self):
        initial_guesses = []
        for k in self.rxn.ts_geom.keys():
            if "ts_guess" in k:
                initial_guesses.append(self.rxn.ts_geom[k])

        # Write inputs for each guess
        for i, conf in enumerate(initial_guesses):
            idx = i + 1
            guess_dir = self.scratch_dir / f"tsopt_run{idx}"
            os.makedirs(guess_dir, exist_ok=True)
            xyz_file = guess_dir / f"ts_guess_{idx}.xyz"
            inp_path = guess_dir / f"tsopt_{idx}_input.yaml"
            
            # Write XYZs
            with open(xyz_file, "w") as f:
                f.write(conf.to_xyz_string())
            
            # Write a Pysisyphus input file for GSM
            self._write_pysis_ts_opt_input(inp_path, xyz_file)


    def write_submission_script(self):
        """
        Creates a sequential runner script for inside the container, 
        and a host submission script to launch the container.
        """
        # ---------------------------------------------------------
        # 1. The INNER Script (Runs inside the Docker container)
        # ---------------------------------------------------------
        inner_script_path = self.scratch_dir / "run_pysis_inner.sh"
        
        with open(inner_script_path, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("echo 'Starting YARP serial TSOPT execution...'\n\n")
            
            for i in range(1, len(self.pairs_to_run) + 1):
                f.write(f"echo '--- Running TSOPT {i} ---'\n")
                # The container mounts self.scratch_dir to /work
                f.write(f"cd /work/tsopt_run{i}\n") 
                
                # Execute pysis within the specific folder, saving logs locally
                f.write(f"pysis tsopt_{i}_input.yaml > tsopt_{i}.log 2> tsopt_{i}.err\n")
                f.write(f"echo '--- Finished TSOPT {i} ---'\n\n")
                
        # Make executable
        inner_script_path.chmod(0o755)

        # ---------------------------------------------------------
        # 2. The OUTER Script (Runs on the Host / Job Manager)
        # ---------------------------------------------------------
        submit_script_path = self.scratch_dir / "submit_tsopt.sh"
        
        # Pulls the docker prefix (e.g., 'docker run --rm -v /scratch:/work -u UID:GID yarp_pysisyphus')
        docker_cmd_prefix = self.get_container_prefix("yarp_pysisyphus", str(self.scratch_dir))
        
        with open(submit_script_path, "w") as f:
            f.write("#!/bin/bash\n")
            # Launch the container and tell it to run the inner script
            f.write(f"{docker_cmd_prefix} bash /work/run_pysis_inner.sh\n")
            
        submit_script_path.chmod(0o755)
        
        # Return the outer script for the Job Manager to execute
        return str(submit_script_path)

    def check_output(self) -> bool:
        pass
    def scrape_data(self):
        # Extract optimized geometry and energy
        pass
    def cleanup(self):
        # Keep .inp, .out, .xyz. Delete .tmp, .densities, etc.
        pass

    def _write_pysis_ts_opt_input(self, input_path, input_geo_xyz):
        # Make sure lot is xTB (ERM: We'll make this more robust later! Hopefully!)
        lot = self.config.lot.lower()
        assert (lot == 'xtb'), "Calculations with Pysisyphus are xTB or bust right now, friend..."

        # Write the file! Yay, YAML friend!
        with open(input_path, 'a') as f:
            # set geom block
            f.write(f'geom:\n type: cart\n fn: {input_geo_xyz}\n')

            # set calc block
            # ERM: I left out the option for solvent,
            # because what I saw in classy YARP didn't make sense to me...
            f.write(f'calc:\n type: {lot}\n pal: {self.config.n_cpus}\n mem: {self.config.mem_per_cpu}\n charge: {self.config.charge}\n mult: {self.config.multiplicity}\n')

            # set opt block
            if self.config.do_hess:
                f.write(f'tsopt:\n type: rsprfo\n do_hess: True\n hessian_recalc: {self.config.hessian_recalc}\n thresh: {self.config.conv_thresh}\n max_cycles: {self.config.max_cycles}\n')
            else:
                f.write(f'tsopt:\n type: rsprfo\n do_hess: False\n thresh: {self.config.conv_thresh}\n max_cycles: {self.config.max_cycles}\n')


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
        if software == "pysisyphus":
            if t_type in ["reactant_optimization", "product_optimization"]:
                return PysisyphusMinOptCalculator(task_def, t_type, rxn_obj)
            elif t_type == "transition_state_optimization":
                return PysisyphusTSOptCalculator(task_def, rxn_obj)
        else:
            pass

    # Task 6: IRC Validation
    elif t_type == "irc_validation":
        # Would return your IRC calculator
        pass

    raise ValueError(f"No calculator implemented for Task: '{t_type}' with Software: '{software}'")