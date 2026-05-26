import os
import re
import numpy as np

from yarp.reaction.external.calc_base import AsyncYarpCalculator
from yarp.yarpecule.input_parsers import xyz_parse
from yarp.yarpecule.graph.adjacency import table_generator
from yarp.util.constants import Constants

class IRCValTask(AsyncYarpCalculator):
    def has_prerequisites(self) -> bool:
        """
        Ensure R, P, and TS with the same LOT and software are available.
        Also check that TS is a valid saddle point
        """
        found_ts = False
        ts_keys = self.rxn.ts_geom.keys()
        expected_ts = f"tsopt_{self.config.lot}_{self.config.software}"
        for k in ts_keys:
            if expected_ts in k and self.rxn.ts_geom[k].is_valid_ts():
                found_ts = True

        found_r = False
        r_keys = self.rxn.reactant.conformers.keys()
        expected_r = f"rpopt_{self.config.lot}_{self.config.software}"
        for k in r_keys:
            if expected_r in k:
                found_r = True

        found_p = False
        p_keys = self.rxn.product.conformers.keys()
        expected_p = f"rpopt_{self.config.lot}_{self.config.software}"
        for k in p_keys:
            if expected_p in k:
                found_p = True

        return found_ts and found_r and found_p

    def _get_num_runs(self) -> int:
        run_dirs = list(self.scratch_dir.glob("irc_run*"))
        return len(run_dirs)

    def _get_rxn_label(self, forward, backward):
        """
        Classify a given IRC outcome based on how forward/backward
        geometries align with original reactant/product graphs
        """
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
        # ERM: All of the "unintended" types are going to be thrown out for now
        # Will revisit later how we can incorporate them in a future development
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

    def _get_final_results(self, irc_runs_dict):
        """
        Determine which TS, label, and barriers to use as the final result for this reaction.
        Preferentially return intended > unintended > reaction-less TS
        If multiple in each category, return TS with lowest forward barrier
        """

        intended = []
        no_rxn = []
        unintended = []
        for data in irc_runs_dict.values():
            label = data['outcome']

            if label == "intended" or label == "inverse_intended":
                intended.append(data)
            elif label == "no_adjmat_change":
                no_rxn.append(data)
            else:
                unintended.append(data)

        # First, check for intended TS and return if present
        if len(intended) >= 1:
            best_ts = None
            best_label = None
            best_f_bar = 100000.00
            best_r_bar = 100000.00
            for data in intended:
                if data['lhs_barrier'] < best_f_bar and data['ts_geom'] is not None:
                    best_ts = data['ts_geom']
                    best_label = data['outcome']
                    best_f_bar = data['lhs_barrier']
                    best_r_bar = data['rhs_barrier']
            
            return best_ts, best_label, best_f_bar, best_r_bar

        # Second, check for unintended TS and return if present
        elif len(unintended) >= 1:
            best_ts = None
            best_label = None
            best_f_bar = 100000.00
            best_r_bar = 100000.00
            for data in unintended:
                if data['lhs_barrier'] < best_f_bar and data['ts_geom'] is not None:
                    best_ts = data['ts_geom']
                    best_label = data['outcome']
                    best_f_bar = data['lhs_barrier']
                    best_r_bar = data['rhs_barrier']
            
            return best_ts, best_label, best_f_bar, best_r_bar

        # If no other option, return "reaction-less" TS
        else:
            best_ts = None
            best_label = None
            best_f_bar = 100000.00
            best_r_bar = 100000.00
            for data in no_rxn:
                if data['lhs_barrier'] < best_f_bar and data['ts_geom'] is not None:
                    best_ts = data['ts_geom']
                    best_label = data['outcome']
                    best_f_bar = data['lhs_barrier']
                    best_r_bar = data['rhs_barrier']
            
            return best_ts, best_label, best_f_bar, best_r_bar


class PysisyphusIRCValCalculator(IRCValTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_name = "erm42/yarp:pysis_xtb"

    def generate_input(self):
        initial_guesses = []
        expected_key = f"tsopt_{self.config.lot}_{self.config.software}"
        
        for k in self.rxn.ts_geom.keys():
            if expected_key in k: 
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
        prefix = self.get_container_prefix(self.image_name, str(self.scratch_dir))
        
        with open(submit_script_path, "w") as f:
            f.write("#!/bin/bash\n")
            self.write_scheduler_headers(f)
            f.write(f"{prefix} bash /work/run_pysis_inner.sh\n")
            
        submit_script_path.chmod(0o755)
        
        # Return the outer script for the Job Manager to execute
        return str(submit_script_path)

    def _is_run_successful(self, i: int) -> bool:
        """Helper method to validate if a specific Pysisyphus IRC run completed successfully."""
        run_dir = self.scratch_dir / f"irc_run{i}"
        log_file = run_dir / f"irc_{i}.log"
        backward_xyz_file = run_dir / "backward_end_opt.xyz"
        forward_xyz_file = run_dir / "forward_end_opt.xyz"

        # 1. File existence check
        if not (log_file.exists() and backward_xyz_file.exists() and forward_xyz_file.exists()):
            return False

        # 2. Log file termination check
        with open(log_file, "r") as f:
            log_text = f.read()

        if "Wrote optimized end-geometries and TS to" not in log_text or "pysisyphus run took" not in log_text:
            return False

        return True

    def check_output(self) -> bool:
        num_runs = self._get_num_runs()
        if num_runs == 0:
            return False # No runs found at all!

        one_successful = False
        for i in range(1, num_runs + 1):
            if self._is_run_successful(i):
                one_successful = True
            else:
                print(f"     * Run {i} failed or did not finish successfully.")
        
        return one_successful

    def scrape_data(self) -> bool:
        # Pull out barriers and outcome labels for all IRC runs
        irc_runs = dict()
        num_runs = self._get_num_runs()
        for i in range(1, num_runs + 1):
            # Skip scraping for this index if the run failed!
            if not self._is_run_successful(i):
                continue

            run_dir = self.scratch_dir / f"irc_run{i}"

            log_file = run_dir / f"irc_{i}.log"
            f_barrier, b_barrier = self._get_barriers(log_file)

            forward_xyz_file = run_dir / "forward_end_opt.xyz"
            f_elements, f_geometry = self._parse_opt_geo(forward_xyz_file)
            f_adj_mat = table_generator(f_elements, f_geometry)

            backward_xyz_file = run_dir / "backward_end_opt.xyz"
            b_elements, b_geometry = self._parse_opt_geo(backward_xyz_file)
            b_adj_mat = table_generator(b_elements, b_geometry)

            irc_outcome = self._get_rxn_label(forward=f_adj_mat, backward=b_adj_mat)

            if irc_outcome == 'inverse_intended':
                lhs = b_barrier
                rhs = f_barrier
            else:
                lhs = f_barrier
                rhs= b_barrier

            target_ts_key = f'{i}' + "_tsopt_" + f'{self.config.lot}_{self.config.software}'
            target_ts = self.rxn.ts_geom.get(target_ts_key, None)
            irc_runs[i] = {
                "outcome": irc_outcome,
                "ts_geom": target_ts,
                "lhs_barrier": lhs,
                "rhs_barrier": rhs
            }
            print(f"     * TS {i} validated: {irc_outcome} with barrier of {lhs} kcal/mol")
        
        # Choose final TS opt structure and reaction outcome label to save to reaction object
        ts, outcome, f_barrier, b_barrier = self._get_final_results(irc_runs)

        self.rxn.ts_geom[f"validated_ts_{self.config.lot}_{self.config.software}"] = ts
        self.rxn.outcome_label[f"{self.config.lot}_{self.config.software}"] = outcome
        self.rxn.barrier[f"{self.config.lot}_{self.config.software}"] = f_barrier
        self.rxn.reverse_barrier[f"{self.config.lot}_{self.config.software}"] = b_barrier

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

    def _parse_opt_geo(self, xyz_file):
        opt_elements, opt_geo = xyz_parse(xyz_file, multiple=False)
        return opt_elements, opt_geo

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

class OrcaIRCValCalculator(IRCValTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ERM: Add a failure check here if the user hasn't built these already
        if self.job_manager.container == "docker":
            self.image_name = "orca:6.0.1"
        elif self.job_manager.container == "apptainer" or self.job_manager.container == "singularity":
            self.image_name = "orca_6.0.1.sif"

    def generate_input(self):
        initial_guesses = []
        expected_key = f"tsopt_{self.config.lot}_{self.config.software}"
        
        for k in self.rxn.ts_geom.keys():
            if expected_key in k: 
                initial_guesses.append(self.rxn.ts_geom[k])

        # Write inputs for each guess
        for i, conf in enumerate(initial_guesses):
            idx = i + 1
            guess_dir = self.scratch_dir / f"irc_run{idx}"
            os.makedirs(guess_dir, exist_ok=True)
            xyz_file = guess_dir / f"ts_opt_{idx}.xyz"
            inp_path = guess_dir / f"irc_{idx}.inp"
            
            # Write XYZs
            with open(xyz_file, "w") as f:
                f.write(conf.to_xyz_string())
            
            # Write a ORCA input file for IRC
            self._write_orca_irc_input(inp_path, f"ts_opt_{idx}.xyz")

    def write_submission_script(self):
        """
        Creates a sequential runner script for inside the container, 
        and a host submission script to launch the container.
        """
        # ---------------------------------------------------------
        # 1. The INNER Script (Runs inside the Docker container)
        # ---------------------------------------------------------
        inner_script_path = self.scratch_dir / "run_orca_inner.sh"
        
        with open(inner_script_path, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("echo 'Starting YARP serial IRC execution...'\n\n")
            
            for i in range(1, self._get_num_runs() + 1):
                f.write(f"echo '--- Running IRC {i} ---'\n")
                # The container mounts self.scratch_dir to /work
                f.write(f"cd /work/irc_run{i}\n") 
                
                # Execute orca within the specific folder, saving logs locally
                f.write(f"orca=$(which orca)\n")
                f.write(f"$orca irc_{i}.inp > irc_{i}.out 2> irc_{i}.err\n")
                f.write(f"echo '--- Finished IRC {i} ---'\n\n")
                
        # Make executable
        inner_script_path.chmod(0o755)

        # ---------------------------------------------------------
        # 2. The OUTER Script (Runs on the Host / Job Manager)
        # ---------------------------------------------------------
        submit_script_path = self.scratch_dir / "submit_irc.sh"

        prefix = self.get_container_prefix(self.image_name, str(self.scratch_dir))
        
        with open(submit_script_path, "w") as f:
            f.write("#!/bin/bash\n")
            self.write_scheduler_headers(f)
            f.write(f"{prefix} bash /work/run_orca_inner.sh\n")
            
        submit_script_path.chmod(0o755)
        
        # Return the outer script for the Job Manager to execute
        return str(submit_script_path)

    def _is_run_successful(self, i: int) -> bool:
        """Helper method to validate if a specific ORCA IRC run completed successfully."""
        run_dir = self.scratch_dir / f"irc_run{i}"
        log_file = run_dir / f"irc_{i}.out"
        backward_xyz_file = run_dir / f"irc_{i}_IRC_B.xyz"
        forward_xyz_file = run_dir / f"irc_{i}_IRC_F.xyz"

        # 1. File existence check
        if not (log_file.exists() and backward_xyz_file.exists() and forward_xyz_file.exists()):
            return False

        # 2. Log file termination check
        with open(log_file, "r") as f:
            log_text = f.read()

        if "THE IRC HAS CONVERGED" not in log_text or "ORCA TERMINATED NORMALLY" not in log_text:
            return False

        return True

    def check_output(self) -> bool:
        num_runs = self._get_num_runs()
        if num_runs == 0:
            return False # No runs found at all!

        one_successful = False
        for i in range(1, num_runs + 1):
            if self._is_run_successful(i):
                one_successful = True
            else:
                print(f"     * Run {i} failed or did not finish successfully.")
        
        return one_successful

    def scrape_data(self) -> bool:
        # Pull out barriers and outcome labels for all IRC runs
        expected_rp = f"rpopt_{self.config.lot}_{self.config.software}"
        gibbs_r = self.rxn.reactant.conformers[expected_rp].properties['gibbs_free_energy_kcal_per_mol']
        gibbs_p = self.rxn.product.conformers[expected_rp].properties['gibbs_free_energy_kcal_per_mol']

        irc_runs = dict()
        num_runs = self._get_num_runs()
        for i in range(1, num_runs + 1):
            # Skip scraping for this index if the run failed!
            if not self._is_run_successful(i):
                continue

            run_dir = self.scratch_dir / f"irc_run{i}"

            target_ts_key = f'{i}' + "_tsopt_" + f'{self.config.lot}_{self.config.software}'
            target_ts = self.rxn.ts_geom.get(target_ts_key, None)
            gibbs_ts = target_ts.properties['gibbs_free_energy_kcal_per_mol']
            f_barrier = gibbs_ts - gibbs_r
            b_barrier = gibbs_ts - gibbs_p

            forward_xyz_file = run_dir / f"irc_{i}_IRC_F.xyz"
            f_elements, f_geometry = self._parse_opt_geo(forward_xyz_file)
            f_adj_mat = table_generator(f_elements, f_geometry)

            backward_xyz_file = run_dir / f"irc_{i}_IRC_B.xyz"
            b_elements, b_geometry = self._parse_opt_geo(backward_xyz_file)
            b_adj_mat = table_generator(b_elements, b_geometry)

            irc_outcome = self._get_rxn_label(forward=f_adj_mat, backward=b_adj_mat)

            if irc_outcome == 'inverse_intended':
                lhs = b_barrier
                rhs = f_barrier
            else:
                lhs = f_barrier
                rhs= b_barrier

            irc_runs[i] = {
                "outcome": irc_outcome,
                "ts_geom": target_ts,
                "lhs_barrier": lhs,
                "rhs_barrier": rhs
            }
            print(f"     * TS {i} validated: {irc_outcome} with barrier of {lhs} kcal/mol")
        
        # Choose final TS opt structure and reaction outcome label to save to reaction object
        ts, outcome, f_barrier, b_barrier = self._get_final_results(irc_runs)

        self.rxn.ts_geom[f"validated_ts_{self.config.lot}_{self.config.software}"] = ts
        self.rxn.outcome_label[f"{self.config.lot}_{self.config.software}"] = outcome
        self.rxn.barrier[f"{self.config.lot}_{self.config.software}"] = f_barrier
        self.rxn.reverse_barrier[f"{self.config.lot}_{self.config.software}"] = b_barrier

        return True

    def cleanup(self):
        pass

    def _write_orca_irc_input(self, input_path, input_geo_xyz):

        # Write the file!
        with open(input_path, 'a') as f:
            # set keywords for level of theory
            f.write(f'! {self.config.lot}\n\n')

            # set keywords to specify IRC calculation
            f.write(f'! Freq IRC\n\n')

            # set parallelization and memory blocks
            f.write(f"%pal\n  nproc {self.config.n_cpus}\nend\n\n")
            f.write(f"%maxcore {self.config.mem_per_cpu}\n\n")

            # set scf opt block (ERM: Make this a user-set number one day?)
            f.write(f"%scf\n  MaxIter 200\nend\n\n")

            # set irc block
            f.write('%irc\n')
            f.write(f'  MaxIter {self.config.max_cycles}\n')
            f.write(f'  InitHess calc_anfreq\n')
            f.write('end\n\n')

            # set XYZ input file
            f.write(f'*xyzfile {self.config.charge} {self.config.multiplicity} {input_geo_xyz}\n')
            f.write('\n# Never forget your bonus lines!!!\n')

    def _parse_opt_geo(self, xyz_file):
        opt_elements, opt_geo = xyz_parse(xyz_file, multiple=False)
        return opt_elements, opt_geo
