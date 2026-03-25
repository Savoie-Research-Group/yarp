import os
import re
import numpy as np

from yarp.reaction.external.calc_base import AsyncYarpCalculator
from yarp.yarpecule.input_parsers import xyz_parse
from yarp.yarpecule.graph.adjacency import table_generator
from yarp.util.constants import Constants

class IRCValTask(AsyncYarpCalculator):
    def has_prerequisites(self) -> bool:
        ts_keys = self.rxn.ts_geom.keys()
        ts_match = False
        for k in ts_keys:
            if 'tsopt' in k and self.rxn.ts_geom[k].geo is not None: 
                ts_match = True
        
        return ts_match

class PysisyphusIRCValCalculator(IRCValTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_name = "yarp_pysisyphus:latest" # Future GHCR link

    def generate_input(self):
        initial_guesses = []
        for k in self.rxn.ts_geom.keys():
            if "tsopt" in k:
                initial_guesses.append(self.rxn.ts_geom[k])

        # Write inputs for each guess
        for i, conf in enumerate(initial_guesses):
            idx = i + 1
            guess_dir = self.scratch_dir / f"irc_run{idx}"
            os.makedirs(guess_dir, exist_ok=True)
            xyz_file = guess_dir / f"ts_opt_{idx}.xyz"
            inp_path = guess_dir / f"irc_{idx}_input.yaml"
            
            # Write XYZs
            with open(xyz_file, "w") as f:
                f.write(conf.to_xyz_string())
            
            # Write a Pysisyphus input file for GSM
            self._write_pysis_irc_input(inp_path, f"ts_opt_{idx}.xyz")

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
            f.write("echo 'Starting YARP serial IRC execution...'\n\n")
            
            for i in range(1, self._get_num_runs() + 1):
                f.write(f"echo '--- Running IRC {i} ---'\n")
                # The container mounts self.scratch_dir to /work
                f.write(f"cd /work/irc_run{i}\n") 
                
                # Execute pysis within the specific folder, saving logs locally
                f.write(f"pysis irc_{i}_input.yaml > irc_{i}.log 2> irc_{i}.err\n")
                f.write(f"echo '--- Finished IRC {i} ---'\n\n")
                
        # Make executable
        inner_script_path.chmod(0o755)

        # ---------------------------------------------------------
        # 2. The OUTER Script (Runs on the Host / Job Manager)
        # ---------------------------------------------------------
        submit_script_path = self.scratch_dir / "submit_irc.sh"
        
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
        num_runs = self._get_num_runs()
        if num_runs == 0:
            return False # No runs found at all!

        one_successful = False
        
        for i in range(1, num_runs + 1):
            run_dir = self.scratch_dir / f"irc_run{i}"
            log_file = run_dir / f"irc_{i}.log"
            backward_xyz_file = run_dir / "backward_end_opt.xyz"
            forward_xyz_file = run_dir / "forward_end_opt.xyz"

            # 1. File existence check
            if not (log_file.exists() and backward_xyz_file.exists() and forward_xyz_file.exists()):
                print(f"   * Run {i} failed: Missing expected output files.")
                continue

            # 2. Log file termination check
            with open(log_file, "r") as f:
                log_text = f.read()

            if "Wrote optimized end-geometries and TS to" not in log_text or "pysisyphus run took" not in log_text:
                print(f"   * Run {i} failed: Did not find successful termination message in log.")
                continue

            # If it passes all checks, at least one run succeeded!
            one_successful = True
        
        # We only care if at least one succeeded
        return one_successful

    def scrape_data(self) -> bool:
        num_runs = self._get_num_runs()
        for i in range(1, num_runs + 1):
            rxn_key = f'irc_{i}_{self.config.lot}_{self.config.software}'
            run_dir = self.scratch_dir / f"irc_run{i}"

            forward_xyz_file = run_dir / "forward_end_opt.xyz"
            f_elements, f_geometry = self._parse_opt_geo(forward_xyz_file)
            f_adj_mat = table_generator(f_elements, f_geometry)

            backward_xyz_file = run_dir / "backward_end_opt.xyz"
            b_elements, b_geometry = self._parse_opt_geo(backward_xyz_file)
            b_adj_mat = table_generator(b_elements, b_geometry)

            rxn_outcome = self._get_rxn_label(forward=f_adj_mat, backward=b_adj_mat)
            self.rxn.outcome_label[rxn_key] = rxn_outcome

            log_file = run_dir / f"irc_{i}.log"
            f_barrier, b_barrier = self._get_barriers(log_file)

            if rxn_outcome == 'inverse_intended':
                self.rxn.barrier[rxn_key + "_kcal_per_mol"] = b_barrier
                self.rxn.reverse_barrier[rxn_key + "_kcal_per_mol"] = f_barrier
            else:
                self.rxn.barrier[rxn_key + "_kcal_per_mol"] = f_barrier
                self.rxn.reverse_barrier[rxn_key + "_kcal_per_mol"] = b_barrier

        return True

    def cleanup(self):
        pass

    def _write_pysis_irc_input(self, input_path, input_geo_xyz):
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

            # set irc block
            f.write(f'irc:\n type: eulerpc\n forward: True\n backward: True\n downhill: False\n')

            # set endopt block
            f.write(f'endopt:\n fragments: False\n do_hess: False\n thresh: {self.config.conv_thresh}\n max_cycles: {self.config.max_cycles}\n')

    def _get_num_runs(self) -> int:
            run_dirs = list(self.scratch_dir.glob("irc_run*"))
            return len(run_dirs)

    def _parse_opt_geo(self, xyz_file):
        opt_elements, opt_geo = xyz_parse(xyz_file, multiple=False)
        return opt_elements, opt_geo
    
    def _get_rxn_label(self, forward, backward):
        label = None

        f_b_diff = np.abs(forward - backward)
        if f_b_diff.sum() == 0:
            # forward and reverse reactions have the same connectivity, uh oh...
            label = 'no_adjmat_change'
            return label

        r_adj = self.rxn.reactant.graph.adj_mat
        p_adj = self.rxn.product.graph.adj_mat

        r_f_diff = np.abs(forward - r_adj)
        r_b_diff = np.abs(backward - r_adj)
        p_f_diff = np.abs(forward - p_adj)
        p_b_diff = np.abs(backward - p_adj)

        if p_f_diff.sum() == 0 and r_b_diff.sum() == 0:
            label = 'intended'
        elif p_b_diff.sum() == 0 and r_f_diff.sum() == 0:
            # forward and reverse barriers are inverted from IRC outputs!
            label = 'inverse_intended'
        elif (p_f_diff.sum() == 0 or p_b_diff.sum() == 0) and not (r_f_diff.sum() != 0 or r_b_diff.sum() != 0):
            # product is reproduced, but reactant is not
            label = 'reactant_unintended'
        elif (r_f_diff.sum() == 0 or r_b_diff.sum() == 0) and not (p_f_diff.sum() != 0 or p_b_diff.sum() != 0):
            # reactant is reproduced, but product is not
            label = 'product_unintended'
        else:
            # neither the reactant nor the product is reproduced
            label = 'unintended'
        
        return label
        
    def _get_barriers(self, log_file):
        """
        Returns left-hand-side (forward) and right-hand-side (backward) barriers
        in units of kcal per mol
        """
        with open(log_file, "r") as f:
            log_text = f.read()

        anchor_pattern = r"Minimum energy of .*? at '(\w+)'\."
        anchors = list(re.finditer(anchor_pattern, log_text))
        if not anchors:
            return None, None
        
        last_anchor_pos = anchors[-1].start()
        relevant_text = log_text[last_anchor_pos:]
        energy_pattern = r"^\s+(Left|TS|Right):\s+([\d.]+)\s+kJ mol"
        matches = re.findall(energy_pattern, relevant_text, re.MULTILINE)
        results = {}
        for label, value in matches:
            results[label] = float(value)
        
        forward_barrier = results['TS'] - results['Left']
        reverse_barrier = results['TS'] - results['Right']
        
        return forward_barrier / Constants.kcal2kJ, reverse_barrier / Constants.kcal2kJ