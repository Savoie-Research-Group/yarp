import os
import h5py
import re
import numpy as np

from yarp.reaction.external.calc_base import AsyncYarpCalculator
from yarp.yarpecule.input_parsers import xyz_parse
from yarp.reaction.conformer import conformer
from yarp.util.constants import Constants


class TSOptTask(AsyncYarpCalculator):
    def has_prerequisites(self) -> bool:

        ts_keys = self.rxn.ts_geom.keys()
        ts_match = False
        for k in ts_keys:
            if 'ts_guess' in k and self.rxn.ts_geom[k].geo is not None: 
                ts_match = True
        
        return ts_match

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
            self._write_pysis_ts_opt_input(inp_path, f"ts_guess_{idx}.xyz")


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
            
            for i in range(1, self._get_num_runs() + 1):
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
        """
        Returns True if AT LEAST ONE of the TS optimizations finished
        """
        num_runs = self._get_num_runs()
        if num_runs == 0:
            return False # No runs found at all!

        one_successful = False
        
        for i in range(1, num_runs + 1):
            run_dir = self.scratch_dir / f"tsopt_run{i}"
            log_file = run_dir / f"tsopt_{i}.log"
            xyz_file = run_dir / "ts_final_geometry.xyz"
            hess_file = run_dir / "ts_final_hessian.h5"
            imag_file = run_dir / "ts_imaginary_mode_000.trj"

            # 1. File existence check
            if not (log_file.exists() and xyz_file.exists()):
                print(f"   * Run {i} failed: Missing expected output files.")
                continue

            # 2. Log file termination check
            with open(log_file, "r") as f:
                log_text = f.read()

            if "Wrote final, hopefully optimized, geometry to" not in log_text or "pysisyphus run took" not in log_text:
                print(f"   * Run {i} failed: Did not find successful termination message in log.")
                continue

            # 3. Hessian file termination check
            if self.config.do_hess:
                if not os.path.exists(hess_file) and not os.path.exists(imag_file):
                    print(f"   * Run {i} failed: Did not find final hessian or imaginary frequencies.")
                    continue

            # If it passes all checks, at least one run succeeded!
            one_successful = True
        
        # We only care if at least one succeeded
        return one_successful

    def scrape_data(self):
        num_runs = self._get_num_runs()
        for i in range(1, num_runs + 1):
            conf = conformer()
            conf.lot = self.config.lot
            conf.software = self.config.software
            conf.type = f"tsopt_{i}_{self.config.lot}_{self.config.software}"

            run_dir = self.scratch_dir / f"tsopt_run{i}"
            log_file = run_dir / f"tsopt_{i}.log"
            conf.properties['internal_energy_Eh'] = self._parse_energy(log_file)

            xyz_file = run_dir / "ts_final_geometry.xyz"
            opt_elements, opt_geo = self._parse_opt_geo(xyz_file)
            conf.elements = opt_elements
            conf.geo = opt_geo

            if self.config.do_hess:
                hess_file = run_dir / "ts_final_hessian.h5"
                hess, freq = self._parse_hessian_freq(hess_file)
                conf.hessian = hess
                conf.vibrational_freqs = freq

                imag_file = run_dir / "ts_imaginary_mode_000.trj"
                conf.imaginary_freq_mode = self._parse_imag_freq_mode(imag_file)

            self.rxn.ts_geom[conf.type] = conf

        print(f'Reaction TS guess keys:\n {list(self.rxn.ts_geom.keys())}')
        return True

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

    def _get_num_runs(self) -> int:
            run_dirs = list(self.scratch_dir.glob("tsopt_run*"))
            return len(run_dirs)

    def _parse_opt_geo(self, xyz_file):
        opt_elements, opt_geo = xyz_parse(xyz_file, multiple=False)
        return opt_elements, opt_geo
    
    def _parse_energy(self, log_file):
        with open(log_file, "r") as f:
            log_text = f.read()
        pattern = r"energy:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+hartree"
        matches = re.findall(pattern, log_text)
        if not matches:
            raise RuntimeError(f"Could not find energy in {log_file}")
        return float(matches[-1])
    
    def _parse_hessian_freq(self, hess_file):
        data = h5py.File(hess_file, 'r')
        hessian = np.array(data['hessian']) / Constants.a0_to_ang**2
        freq = np.array(data['vibfreqs'])
        return hessian, freq
    
    def _parse_imag_freq_mode(self, imag_file):
        elements, geometries = xyz_parse(imag_file, multiple=True)
        imag_freq_mode = []
        for count, el in enumerate(elements):
            imag_freq_mode.append((el, geometries[count]))
        
        return imag_freq_mode