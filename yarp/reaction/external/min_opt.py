import os
import h5py
import re
from pathlib import Path
import numpy as np

from yarp.reaction.external.calc_base import AsyncYarpCalculator
from yarp.yarpecule.input_parsers import xyz_parse
from yarp.reaction.conformer import conformer
from yarp.util.constants import Constants

class MinOptTask(AsyncYarpCalculator):
    def has_prerequisites(self) -> bool:
        # 1. Determine which state we care about for this specific task
        if self.task_def.task_type == "reactant_optimization":
            node = self.rxn.reactant
            source = self.config.initial_geom.reactant
        elif self.task_def.task_type == "product_optimization":
            node = self.rxn.product
            source = self.config.initial_geom.reactant
        else:
            raise ValueError(f"Unknown task type for MinOpt: {self.task_def.task_type}")

        if not node.conformers:
            return False

        # 2. Determine the target key to look for
        if source.label == "conf_gen":
            expected_key = "conf_gen_rank0"
        elif source.label == "rp_opt":
            expected_key = f"rpopt_{source.lot}_{source.software}"
        else:
            raise ValueError(f"Unknown initial geom label for R/P MinOpt: {source.label}")

        # 3. Check if the target conformer exists
        for key in node.conformers.keys():
            if expected_key in key and node.conformers[key].geo is not None: 
                return True

        return False

class PysisyphusMinOptCalculator(MinOptTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_name = "erm42/yarp:pysis_xtb"

    def generate_input(self):
        if self.task_def.task_type == "reactant_optimization":
            node = self.rxn.reactant
            source = self.config.initial_geom.reactant
        elif self.task_def.task_type == "product_optimization":
            node = self.rxn.product
            source = self.config.initial_geom.product
        else:
            raise ValueError(f"Unknown task type for MinOpt: {self.task_def.task_type}")

        if source.label == "conf_gen":
            expected_key = "conf_gen_rank0"
        elif source.label == "rp_opt":
            expected_key = f"rpopt_{source.lot}_{source.software}"

        initial_conf = None
        for key in node.conformers.keys():
            if expected_key in key:
                initial_conf = node.conformers[key]
                break
                
        if not initial_conf:
            raise ValueError(f"Could not find requested geometry: {expected_key}")

        input_xyz_path = self.scratch_dir / "initial_geom.xyz"
        with open(input_xyz_path, "w") as f:
            f.write(initial_conf.to_xyz_string())

        inp_path = self.scratch_dir / "min_opt.yaml"
        self._write_pysis_rp_opt_input(inp_path, "initial_geom.xyz")

    def write_submission_script(self) -> Path:
        """Write the bash script that the JobManager will execute."""
        script_path = self.scratch_dir / "run_pysis_rpopt.sh"

        # Construct the core command
        prefix = self.get_container_prefix(self.image_name, self.scratch_dir)
        pysis_cmd = "pysis min_opt.yaml > min_opt.log 2> min_opt.err"
        full_command = f"{prefix} {pysis_cmd}"

        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            self.write_scheduler_headers(f)
            f.write(f"cd {self.scratch_dir}\n")
            f.write(f"{full_command}\n")

        # Make the script executable (important for LocalJobManager)
        script_path.chmod(0x755)

        return script_path

    def check_output(self) -> bool:
        log_file = self.scratch_dir / f"min_opt.log"
        xyz_file = self.scratch_dir / "final_geometry.xyz"
        hess_file = self.scratch_dir / "final_hessian.h5"

        success = True

        # 1. File existence check
        if not (log_file.exists() and xyz_file.exists()):
            print(f"     * Run failed: Missing expected output files.")
            success = False

        # 2. Log file termination check
        with open(log_file, "r") as f:
            log_text = f.read()

        if "Wrote final, hopefully optimized, geometry to" not in log_text or "pysisyphus run took" not in log_text:
            print(f"     * Run failed: Did not find successful termination message in log.")
            success = False

        # 3. Hessian file termination check
        if self.config.do_hess:
            if not os.path.exists(hess_file):
                print(f"     * Run failed: Did not find final hessian file.")
                success = False

        return success            

    def scrape_data(self) -> bool:
        conf = conformer()
        conf.lot = self.config.lot
        conf.software = self.config.software
        conf.type = f"rpopt_{self.config.lot}_{self.config.software}"

        opt_elements, opt_geo = self._parse_opt_geo()
        conf.elements = opt_elements
        conf.geo = opt_geo

        conf.properties['internal_energy_Eh'] = self._parse_energy()

        if self.config.do_hess:
            hess, freq = self._parse_hessian_freq()
            conf.vibrational_freqs = freq
            conf.hessian = hess
        
        if self.task_def.task_type == "reactant_optimization":
            self.rxn.reactant.conformers[conf.type] = conf
        elif self.task_def.task_type == "product_optimization":
            self.rxn.product.conformers[conf.type] = conf
        else:
            raise ValueError(f"Unknown task type for MinOpt: {self.task_def.task_type}")

        return True

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

    def _parse_opt_geo(self):
        xyz_file = self.scratch_dir / "final_geometry.xyz"
        opt_elements, opt_geo = xyz_parse(xyz_file, multiple=False)
        return opt_elements, opt_geo
    
    def _parse_energy(self):
        log_file = self.scratch_dir / f"min_opt.log"
        with open(log_file, "r") as f:
            log_text = f.read()
        pattern = r"energy:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+hartree"
        matches = re.findall(pattern, log_text)
        if not matches:
            raise RuntimeError(f"Could not find energy in {log_file}")
        return float(matches[-1])
    
    def _parse_hessian_freq(self):
        hess_file = self.scratch_dir / f"final_hessian.h5"
        if os.path.exists(hess_file):
            data = h5py.File(hess_file, 'r')
            hessian = np.array(data['hessian']) / Constants.a0_to_ang**2
            freq = np.array(data['vibfreqs'])
            return hessian, freq
        else:
            return None, None

class OrcaMinOptCalculator(MinOptTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.job_manager.container == "docker":
            self.image_name = "orca:6.0.1"
        elif self.job_manager.container == "apptainer":
            self.image_name = "orca_6.0.1.sif"

    def generate_input(self):
        if self.task_def.task_type == "reactant_optimization":
            node = self.rxn.reactant
            source = self.config.initial_geom.reactant
        elif self.task_def.task_type == "product_optimization":
            node = self.rxn.product
            source = self.config.initial_geom.product
        else:
            raise ValueError(f"Unknown task type for MinOpt: {self.task_def.task_type}")

        if source.label == "conf_gen":
            expected_key = "conf_gen_rank0"
        elif source.label == "rp_opt":
            expected_key = f"rpopt_{source.lot}_{source.software}"

        initial_conf = None
        for key in node.conformers.keys():
            if expected_key in key:
                initial_conf = node.conformers[key]
                break
                
        if not initial_conf:
            raise ValueError(f"Could not find requested geometry: {expected_key}")

        input_xyz_path = self.scratch_dir / "initial_geom.xyz"
        with open(input_xyz_path, "w") as f:
            f.write(initial_conf.to_xyz_string())

        inp_path = self.scratch_dir / "min_opt.inp"
        self._write_orca_rp_opt_input(inp_path, "initial_geom.xyz")

    def write_submission_script(self) -> Path:
        """Write the bash script that the JobManager will execute."""
        script_path = self.scratch_dir / "run_orca_rpopt.sh"

        # Construct the core command
        prefix = self.get_container_prefix(self.image_name, self.scratch_dir)
        orca_cmd = "/bin/bash -c 'orca=$(which orca) && $orca min_opt.inp > min_opt.out 2> min_opt.err'"
        full_command = f"{prefix} {orca_cmd}"

        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            self.write_scheduler_headers(f)
            f.write(f"cd {self.scratch_dir}\n")
            f.write(f"{full_command}\n")

        # Make the script executable (important for LocalJobManager)
        script_path.chmod(0x755)

        return script_path

    def check_output(self) -> bool:
        out_file = self.scratch_dir / f"min_opt.out"
        xyz_file = self.scratch_dir / "min_opt.xyz"

        success = True

        # 1. File existence check
        if not (out_file.exists() and xyz_file.exists()):
            print(f"     * Run failed: Missing expected output files.")
            success = False

        # 2. Log file termination check
        with open(out_file, "r") as f:
            log_text = f.read()

        if "THE OPTIMIZATION HAS CONVERGED" not in log_text or "ORCA TERMINATED NORMALLY" not in log_text:
            print(f"     * Run failed: Did not find successful termination message in output.")
            success = False

        return success            

    def scrape_data(self) -> bool:
        conf = conformer()
        conf.lot = self.config.lot
        conf.software = self.config.software
        conf.type = f"rpopt_{self.config.lot}_{self.config.software}"

        opt_elements, opt_geo = self._parse_opt_geo()
        conf.elements = opt_elements
        conf.geo = opt_geo

        conf.properties['internal_energy_Eh'] = self._parse_energy()

        if self.config.do_hess:
            hess, freq = self._parse_hessian_freq()
            conf.vibrational_freqs = freq
            conf.hessian = hess
        
        if self.task_def.task_type == "reactant_optimization":
            self.rxn.reactant.conformers[conf.type] = conf
        elif self.task_def.task_type == "product_optimization":
            self.rxn.product.conformers[conf.type] = conf
        else:
            raise ValueError(f"Unknown task type for MinOpt: {self.task_def.task_type}")

        return True

    def cleanup(self):
        # Keep .inp, .out, .xyz. Delete .tmp, .densities, etc.
        pass

    def _write_orca_rp_opt_input(self, input_path, input_geo_xyz):

        # Write the file!
        with open(input_path, 'a') as f:
            # set keywords for level of theory
            f.write(f'! {self.config.lot}\n\n')

            # set keywords to specify local minimum optimization
            f.write(f'! OPT\n\n')

            # set parallelization and memory blocks
            f.write(f"%pal\n  nproc {self.config.n_cpus}\nend\n\n")
            f.write(f"%maxcore {self.config.mem_per_cpu}\n\n")

            # set scf opt block (ERM: Make this a user-set number one day?)
            f.write(f"%scf\n  MaxIter 200\nend\n\n")

            # set geom opt block
            f.write('%geom\n')
            f.write(f'  MaxIter {self.config.max_cycles}\n')
            if self.config.do_hess: f.write(f'  Calc_Hess true\n  Recalc_Hess {self.config.hessian_recalc}\n')
            f.write('end\n\n')

            # set XYZ input file
            f.write(f'*xyzfile {self.config.charge} {self.config.multiplicity} {input_geo_xyz}\n')
            f.write('\n# Never forget your bonus lines!!!\n')


    def _parse_opt_geo(self):
        xyz_file = self.scratch_dir / "min_opt.xyz"
        opt_elements, opt_geo = xyz_parse(xyz_file, multiple=False)
        return opt_elements, opt_geo
    
    def _parse_energy(self):
        log_file = self.scratch_dir / f"min_opt.out"
        with open(log_file, "r") as f:
            log_text = f.read()
        pattern = r"FINAL SINGLE POINT ENERGY\s+([-+]?\d*\.\d+)"
        matches = re.findall(pattern, log_text)
        if not matches:
            raise RuntimeError(f"Could not find energy in {log_file}")
        return float(matches[-1])
    
    def _parse_hessian_freq(self):
        return False, False