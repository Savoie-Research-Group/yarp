import os
import shutil
import h5py
import re
import numpy as np

from yarp.reaction.external.calc_base import AsyncYarpCalculator
from yarp.yarpecule.input_parsers import xyz_parse
from yarp.reaction.conformer import conformer
from yarp.util.constants import Constants


class TSOptTask(AsyncYarpCalculator):
    def has_prerequisites(self) -> bool:
        source = self.config.initial_geom.transition_state
        if source.label == "ts_guess":
            expected_key = "ts_guess"
        elif source.label == "ts_opt":
            expected_key = f"validated_ts_{source.lot}_{source.software}"
        else:
            raise ValueError(f"Unknown initial geom label for TSOpt: {source.label}")

        for k in self.rxn.ts_geom.keys():
            if expected_key in k and self.rxn.ts_geom[k].geo is not None: 
                return True

        return False

    def _get_num_runs(self) -> int:
            run_dirs = list(self.scratch_dir.glob("tsopt_run*"))
            return len(run_dirs)

class PysisyphusTSOptCalculator(TSOptTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_name = "erm42/yarp:pysis_xtb"

    def generate_input(self):
        source = self.config.initial_geom.transition_state
        if source.label == "ts_guess":
            expected_key = "ts_guess"
        elif source.label == "ts_opt":
            expected_key = f"validated_ts_{source.lot}_{source.software}"
        else:
            raise ValueError(f"Unknown initial geom label for TSOpt: {source.label}")

        initial_guesses = []
        for k in self.rxn.ts_geom.keys():
            if expected_key in k:
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
            
            # Write a Pysisyphus input file for TSOPT
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
        
        prefix = self.get_container_prefix(self.image_name, str(self.scratch_dir))
        
        with open(submit_script_path, "w") as f:
            f.write("#!/bin/bash\n")
            self.write_scheduler_headers(f)
            f.write(f"{prefix} bash /work/run_pysis_inner.sh\n")
            
        submit_script_path.chmod(0o755)
        
        # Return the outer script for the Job Manager to execute
        return str(submit_script_path)

    def check_output(self) -> bool:
        """Returns True if AT LEAST ONE of the TS optimizations finished."""
        num_runs = self._get_num_runs()
        if num_runs == 0:
            return False 

        one_successful = False
        for i in range(1, num_runs + 1):
            if self._is_run_successful(i):
                one_successful = True
            else:
                print(f"     ! Run {i} failed or did not finish successfully. Try increasing mem_per_cpu for tasks using 'pysisyphus'.")
                
        return one_successful

    def scrape_data(self) -> bool:
        num_runs = self._get_num_runs()
        for i in range(1, num_runs + 1):
            # Skip scraping for this index if the run failed!
            if not self._is_run_successful(i):
                continue

            conf = conformer()
            conf.lot = self.config.lot
            conf.software = self.config.software
            conf.type = f"{i}_tsopt_{self.config.lot}_{self.config.software}"

            run_dir = self.scratch_dir / f"tsopt_run{i}"
            log_file = run_dir / f"tsopt_{i}.log"
            conf.properties['internal_energy_Eh'] = self._parse_energy(log_file)

            xyz_file = run_dir / "ts_final_geometry.xyz"
            opt_elements, opt_geo = self._parse_opt_geo(xyz_file)
            conf.elements = opt_elements
            conf.geo = opt_geo

            hess_file = run_dir / "ts_final_hessian.h5"
            hess, freq = self._parse_hessian_freq(hess_file)
            conf.hessian = hess
            conf.vibrational_freqs = freq

            self.rxn.ts_geom[conf.type] = conf

        return True

    def cleanup(self):
        """Per run dir: keep yaml input, log, and final TS geometry; delete Hessian (scraped) and xTB calc dirs."""
        num_runs = self._get_num_runs()
        for i in range(1, num_runs + 1):
            run_dir = self.scratch_dir / f"tsopt_run{i}"
            if not run_dir.exists():
                continue
            keep = {f"tsopt_{i}_input.yaml", f"tsopt_{i}.log", "ts_final_geometry.xyz"}
            for item in run_dir.iterdir():
                if item.name not in keep:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)

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
            f.write(f'tsopt:\n type: rsprfo\n do_hess: True\n hessian_recalc: {self.config.hessian_recalc}\n thresh: {self.config.conv_thresh}\n max_cycles: {self.config.max_cycles}\n')

    def _is_run_successful(self, i: int) -> bool:
        """Helper method to validate if a specific run completed successfully."""
        run_dir = self.scratch_dir / f"tsopt_run{i}"
        log_file = run_dir / f"tsopt_{i}.log"
        xyz_file = run_dir / "ts_final_geometry.xyz"
        hess_file = run_dir / "ts_final_hessian.h5"

        # 1. File existence check
        if not (log_file.exists() and xyz_file.exists() and hess_file.exists()):
            return False

        # 2. Log file termination check
        with open(log_file, "r") as f:
            log_text = f.read()

        if "Wrote final, hopefully optimized, geometry to" not in log_text or "pysisyphus run took" not in log_text:
            return False

        return True

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

class OrcaTSOptCalculator(TSOptTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.job_manager.container == "docker":
            self.image_name = "orca:6.0.1"
        elif self.job_manager.container == "apptainer" or self.job_manager.container == "singularity":
            self.image_name = "orca_6.0.1.sif"

    def generate_input(self):
        source = self.config.initial_geom.transition_state
        if source.label == "ts_guess":
            expected_key = "ts_guess"
        elif source.label == "ts_opt":
            expected_key = f"validated_ts_{source.lot}_{source.software}"
        else:
            raise ValueError(f"Unknown initial geom label for TSOpt: {source.label}")

        initial_guesses = []
        for k in self.rxn.ts_geom.keys():
            if expected_key in k:
                initial_guesses.append(self.rxn.ts_geom[k])

        # Write inputs for each guess
        for i, conf in enumerate(initial_guesses):
            idx = i + 1
            guess_dir = self.scratch_dir / f"tsopt_run{idx}"
            os.makedirs(guess_dir, exist_ok=True)
            xyz_file = guess_dir / f"ts_guess_{idx}.xyz"
            inp_path = guess_dir / f"tsopt_{idx}.inp"
            
            # Write XYZs
            with open(xyz_file, "w") as f:
                f.write(conf.to_xyz_string())
            
            # Write a ORCA input file for TSOPT
            self._write_orca_ts_opt_input(inp_path, f"ts_guess_{idx}.xyz")


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
            f.write("echo 'Starting YARP serial TSOPT execution...'\n\n")
            
            for i in range(1, self._get_num_runs() + 1):
                f.write(f"echo '--- Running TSOPT {i} ---'\n")
                # The container mounts self.scratch_dir to /work
                f.write(f"cd /work/tsopt_run{i}\n") 
                
                # Execute orca within the specific folder, saving logs locally
                f.write(f"orca=$(which orca)\n")
                f.write(f"$orca tsopt_{i}.inp > tsopt_{i}.out 2> tsopt_{i}.err\n")
                f.write(f"echo '--- Finished TSOPT {i} ---'\n\n")
                
        # Make executable
        inner_script_path.chmod(0o755)

        # ---------------------------------------------------------
        # 2. The OUTER Script (Runs on the Host / Job Manager)
        # ---------------------------------------------------------
        submit_script_path = self.scratch_dir / "submit_tsopt.sh"
        
        prefix = self.get_container_prefix(self.image_name, str(self.scratch_dir))
        
        with open(submit_script_path, "w") as f:
            f.write("#!/bin/bash\n")
            self.write_scheduler_headers(f)
            f.write(f"{prefix} bash /work/run_orca_inner.sh\n")
            
        submit_script_path.chmod(0o755)
        
        # Return the outer script for the Job Manager to execute
        return str(submit_script_path)

    def check_output(self) -> bool:
        """Returns True if AT LEAST ONE of the TS optimizations finished."""
        num_runs = self._get_num_runs()
        if num_runs == 0:
            return False 

        one_successful = False
        for i in range(1, num_runs + 1):
            if self._is_run_successful(i):
                one_successful = True
            else:
                print(f"     * Run {i} failed or did not finish successfully.")
        
        return one_successful

    def scrape_data(self) -> bool:
        num_runs = self._get_num_runs()
        for i in range(1, num_runs + 1):
            # Skip scraping for this index if the run failed!
            if not self._is_run_successful(i):
                continue

            conf = conformer()
            conf.lot = self.config.lot
            conf.software = self.config.software
            conf.type = f"{i}_tsopt_{self.config.lot}_{self.config.software}"

            run_dir = self.scratch_dir / f"tsopt_run{i}"
            log_file = run_dir / f"tsopt_{i}.out"
            
            conf.properties['internal_energy_Eh'] = self._parse_energy(log_file)

            enthalpy, entropy, gibbs = self._parse_orca_thermo(log_file)
            conf.properties['gibbs_free_energy_kcal_per_mol'] = gibbs
            conf.properties['enthalpy_kcal_per_mol'] = enthalpy
            conf.properties['entropy_temp_kcal_per_mol'] = entropy

            xyz_file = run_dir / f"tsopt_{i}.xyz"
            opt_elements, opt_geo = self._parse_opt_geo(xyz_file)
            conf.elements = opt_elements
            conf.geo = opt_geo

            hess_file = run_dir / f"tsopt_{i}.hess"
            hess, freq = self._parse_hessian_freq(hess_file)
            conf.hessian = hess
            conf.vibrational_freqs = freq

            self.rxn.ts_geom[conf.type] = conf

        return True

    def cleanup(self):
        """Per run dir: keep inp, output log, and final TS geometry; delete Hessian (scraped), .gbw, .densities, etc."""
        num_runs = self._get_num_runs()
        for i in range(1, num_runs + 1):
            run_dir = self.scratch_dir / f"tsopt_run{i}"
            if not run_dir.exists():
                continue
            keep = {f"tsopt_{i}.inp", f"tsopt_{i}.out", f"tsopt_{i}.xyz"}
            for item in run_dir.iterdir():
                if item.name not in keep:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)

    def _write_orca_ts_opt_input(self, input_path, input_geo_xyz):

        # Write the file!
        with open(input_path, 'a') as f:
            # set keywords for level of theory
            f.write(f'! {self.config.lot}\n\n')

            # set keywords to specify saddle point optimization
            f.write(f'! OptTS\n\n')

            # set parallelization and memory blocks
            f.write(f"%pal\n  nproc {self.config.n_cpus}\nend\n\n")
            f.write(f"%maxcore {self.config.mem_per_cpu}\n\n")

            # set scf opt block (ERM: Make this a user-set number one day?)
            f.write(f"%scf\n  MaxIter 200\nend\n\n")

            # set geom opt block
            f.write('%geom\n')
            f.write(f'  MaxIter {self.config.max_cycles}\n')
            f.write(f'  Calc_Hess true\n  Recalc_Hess {self.config.hessian_recalc}\n')
            f.write('end\n\n')

            # set XYZ input file
            f.write(f'*xyzfile {self.config.charge} {self.config.multiplicity} {input_geo_xyz}\n')
            f.write('\n# Never forget your bonus lines!!!\n')

    def _is_run_successful(self, i: int) -> bool:
        """Helper method to validate if a specific ORCA run completed successfully."""
        run_dir = self.scratch_dir / f"tsopt_run{i}"
        log_file = run_dir / f"tsopt_{i}.out"
        xyz_file = run_dir / f"tsopt_{i}.xyz"
        hess_file = run_dir / f"tsopt_{i}.hess"

        if not (log_file.exists() and xyz_file.exists() and hess_file.exists()):
            return False

        with open(log_file, "r") as f:
            log_text = f.read()

        if "THE OPTIMIZATION HAS CONVERGED" not in log_text or "ORCA TERMINATED NORMALLY" not in log_text:
            return False

        return True

    def _parse_opt_geo(self, xyz_file):
        opt_elements, opt_geo = xyz_parse(xyz_file, multiple=False)
        return opt_elements, opt_geo
    
    def _parse_energy(self, log_file):
        with open(log_file, "r") as f:
            log_text = f.read()
        pattern = r"FINAL SINGLE POINT ENERGY\s+([-+]?\d*\.\d+)"
        matches = re.findall(pattern, log_text)
        if not matches:
            raise RuntimeError(f"Could not find energy in {log_file}")
        return float(matches[-1])
    
    def _parse_hessian_freq(self, hess_file):
        """
        Parses an ORCA .hess file to extract the Hessian matrix and 
        vibrational frequencies.
        
        Returns:
            hessian (np.ndarray): Square N x N matrix (Hartree/Bohr^2)
            frequencies (np.ndarray): Vector of length N (cm^-1)
        """
        with open(hess_file, 'r') as f:
            lines = f.readlines()

        hessian = None
        frequencies = None
        dim = 0

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # 1. Parse Hessian Matrix
            if line == "$hessian":
                i += 1
                dim = int(lines[i].strip())
                hessian = np.zeros((dim, dim))
                i += 1

                # The Hessian is printed in blocks of 5 columns
                while True:
                    # Check if we hit the next section or end of data
                    if i >= len(lines) or lines[i].startswith('$') or not lines[i].strip():
                        break

                    # These are the column indices (e.g., 0 1 2 3 4)
                    col_indices = [int(x) for x in lines[i].split()]
                    i += 1

                    # Read the next 'dim' lines for these specific columns
                    for _ in range(dim):
                        parts = lines[i].split()
                        row_idx = int(parts[0])
                        values = [float(x) for x in parts[1:]]

                        for col_offset, val in enumerate(values):
                            col_idx = col_indices[col_offset]
                            hessian[row_idx, col_idx] = val
                        i += 1

                    # Check for blank lines between blocks
                    while i < len(lines) and not lines[i].strip():
                        i += 1
                continue

            # 2. Parse Vibrational Frequencies
            elif line == "$vibrational_frequencies":
                i += 1
                n_freqs = int(lines[i].strip())
                frequencies = np.zeros(n_freqs)
                i += 1
                for _ in range(n_freqs):
                    parts = lines[i].split()
                    idx = int(parts[0])
                    val = float(parts[1])
                    frequencies[idx] = val
                    i += 1
                continue

            i += 1

        return hessian, frequencies

    def _parse_orca_thermo(self, filename):
        """
        Parses enthalpy, entropy correction, and Gibbs free energy 
        by searching for key phrases and capturing the first float.
        """
        enthalpy = None
        entropy = None
        gibbs = None

        # We use this flag to ensure we only grab values from the 
        # 'GIBBS FREE ENERGY' section, ignoring earlier sections.
        in_gibbs_block = False

        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Detect the start of the final summary section
                if "GIBBS FREE ENERGY" in line:
                    in_gibbs_block = True
                
                if in_gibbs_block:
                    # Capture 'Total enthalpy'
                    if "Total enthalpy" in line and "..." in line:
                        match = re.search(r"(-?\d+\.\d+)", line)
                        if match:
                            enthalpy = float(match.group(1)) * Constants.ha_to_kcalmol
                    
                    # Capture 'Total entropy correction'
                    elif "Total entropy correction" in line and "..." in line:
                        match = re.search(r"(-?\d+\.\d+)", line)
                        if match:
                            entropy = float(match.group(1)) * Constants.ha_to_kcalmol
                    
                    # Capture 'Final Gibbs free energy'
                    elif "Final Gibbs free energy" in line:
                        match = re.search(r"(-?\d+\.\d+)", line)
                        if match:
                            gibbs = float(match.group(1)) * Constants.ha_to_kcalmol
                            # Once we have the final value, we can stop
                            break

        return enthalpy, entropy, gibbs