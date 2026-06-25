import os
import shutil
import fnmatch
import pickle

from yarp.reaction.external.calc_base import AsyncYarpCalculator
from yarp.yarpecule.input_parsers import xyz_parse
from yarp.reaction.conformer import conformer
from yarp.reaction.conf_sampling.select_pairs import select_gsm_pairs

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

class PysisyphusTSGuessCalculator(TSGuessTask):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self.config, "containerize_pre_gsm", False):
            self.image_name = "erm42/yarp:jo_opt"
        else:
            self.image_name = "erm42/yarp:pysis_xtb"
        self.n_pairs = self.config.n_conf
        self.pairs_to_run = []

    def generate_input(self):
        """
        Runs the Joint Opt + ML Selection, then writes the files 
        required for the Pysisyphus GSM run.
        """
        if getattr(self.config, "containerize_pre_gsm", False):
            payload_path = self.scratch_dir / "payload.pkl"
            with open(payload_path, "wb") as f:
                pickle.dump({"rxn": self.rxn, "config": self.config}, f)
            return

        print(f"     * [{self.rxn.hash}] Selecting {self.n_pairs} conformer pairs for GSM...")
        self.pairs_to_run = select_gsm_pairs(self.rxn, self.config, scratch_dir=self.scratch_dir / "joint_opt")
        
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

            if getattr(self.config, "containerize_pre_gsm", False):
                f.write("python -m yarp.reaction.external.scripts.run_ts_guess_container /work/payload.pkl\n")
            else:
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
        prefix = self.get_container_prefix(self.image_name, str(self.scratch_dir))
        
        with open(submit_script_path, "w") as f:
            f.write("#!/bin/bash\n")
            self.write_scheduler_headers(f)
            # Launch the container and tell it to run the inner script
            f.write(f"{prefix} bash /work/run_pysis_inner.sh\n")
            
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
                print(f"     * Run {i} failed: Missing expected output files.")
                continue

            # 2. Log file termination check
            with open(log_file, "r") as f:
                log_text = f.read()

            if "Wrote splined HEI" not in log_text or "pysisyphus run took" not in log_text:
                print(f"     ! Run {i} failed: Did not find successful termination message in log. Try increasing mem_per_cpu for tasks using 'pysisyphus'.")
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
        
        return True

    def cleanup(self):
        """Per run dir: keep inputs, logs, xyzs, and trajectories; delete xTB calc dirs."""
        num_runs = self._get_num_runs()
        for i in range(1, num_runs + 1):
            run_dir = self.scratch_dir / f"gsm_run{i}"
            if not run_dir.exists():
                continue

            keep_exact = {f"gsm_{i}_input.yaml", f"gsm_{i}.log", "splined_hei.xyz"}
            keep_patterns = ["*.trj", "*.xyz"]

            for item in run_dir.iterdir():
                if item.name in keep_exact:
                    continue
                if item.is_file() and any(fnmatch.fnmatch(item.name, p) for p in keep_patterns):
                    continue
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)

    def _write_pysis_gsm_input(self, input_path, r_xyz_path, p_xyz_path):
        write_pysis_gsm_input(input_path, r_xyz_path, p_xyz_path, self.config)

    def _get_num_runs(self) -> int:
            """
            Dynamically counts the number of gsm_run subdirectories.
            Crucial for stateless execution where self.pairs_to_run is lost between runs.
            """
            # Finds all directories matching 'gsm_run*' in the scratch folder
            run_dirs = list(self.scratch_dir.glob("gsm_run*"))
            return len(run_dirs)


def write_pysis_gsm_input(input_path, r_xyz_path, p_xyz_path, config):
    # Make sure lot is xTB (ERM: We'll make this more robust later! Hopefully!)
    lot = config.gsm_lot.lower()

    with open(input_path, 'w') as f:
        input_geo = [r_xyz_path, p_xyz_path]
        f.write(f'geom:\n type: cart\n fn: {input_geo}\n')

        # ERM: I left out the option for solvent,
        # because what I saw in classy YARP didn't make sense to me...
        f.write(f'calc:\n type: {lot}\n pal: {config.n_cpus}\n mem: {config.mem_per_cpu}\n charge: {config.charge}\n mult: {config.multiplicity}\n')

        f.write(f'cos:\n type: gs\n max_nodes: {config.max_gsm_nodes}\n climb: True\n climb_rms: 0.005\n climb_lanczos: False\n reparam_check: rms\n reparam_every: 1\n reparam_every_full: 1\n')

        f.write(f'opt:\n type: string\n stop_in_when_full: -1\n align: True\n scale_step: global\n')
