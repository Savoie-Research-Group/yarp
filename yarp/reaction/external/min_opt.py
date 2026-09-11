import os
import shutil
import h5py
import re
from pathlib import Path
import numpy as np

from yarp.reaction.external.calc_base import AsyncYarpCalculator, CalculatorInputError
from yarp.yarpecule.input_parsers import xyz_parse
from yarp.reaction.conformer import conformer
from yarp.reaction.conf_sampling.joint_opt import joint_optimize
from yarp.yarpecule.graph.adjacency import compare_adjacency, describe_adjacency_change
from yarp.util.constants import Constants

# Which side of the reaction each task type operates on, and whether it is the
# cheap pre-optimization that runs ahead of conformer generation or the
# refinement optimization that runs after it. Both do the same thing --
# minimize a structure -- so they share a calculator; these two maps are the
# only places the difference is spelled out.
_TASK_SIDE = {
    "reactant_pre_opt": "reactant",
    "product_pre_opt": "product",
    "reactant_optimization": "reactant",
    "product_optimization": "product",
}
_PRE_OPT_TASKS = frozenset({"reactant_pre_opt", "product_pre_opt"})


class MinOptTask(AsyncYarpCalculator):
    """
    Shared behaviour for minimizing a reactant or product structure.

    Serves four task types: the two pre-optimization legs and the two
    refinement legs. The pre-opt legs differ only in where their starting
    geometry comes from, which optimizer runs, and what the resulting
    conformer is called.
    """

    @property
    def side(self) -> str:
        """'reactant' or 'product', from the task type."""
        try:
            return _TASK_SIDE[self.task_def.task_type]
        except KeyError:
            raise ValueError(f"Unknown task type for MinOpt: {self.task_def.task_type}")

    @property
    def is_pre_opt(self) -> bool:
        return self.task_def.task_type in _PRE_OPT_TASKS

    @property
    def node(self):
        """The state this task optimizes."""
        return self.rxn.reactant if self.side == "reactant" else self.rxn.product

    def output_key(self) -> str:
        """Conformer key this task writes its result under."""
        prefix = "preopt" if self.is_pre_opt else "rpopt"
        return f"{prefix}_{self.config.lot}_{self.config.software}"

    def source_key(self) -> str:
        """
        Conformer key this task reads its starting geometry from.

        Pre-opt has no `initial_geom` block to consult: the reactant leg always
        starts from the yarpecule's own geometry, and the product leg starts
        from the relaxed reactant, which it patches onto the product BEM in
        `generate_input` rather than reading directly.
        """
        if self.is_pre_opt:
            if self.side == "reactant":
                return "initial_geom"
            # The product leg reads the RELAXED REACTANT, not a product
            # conformer -- see `_patched_product_geometry`.
            return f"preopt_{self.config.lot}_{self.config.software}"

        source = getattr(self.config.initial_geom, self.side)
        if source.label == "conf_gen":
            return "conf_gen_rank0"
        elif source.label == "rp_opt":
            return f"rpopt_{source.lot}_{source.software}"
        raise ValueError(f"Unknown initial geom label for R/P MinOpt: {source.label}")

    def source_node(self):
        """
        The state the starting geometry is read from.

        Everything reads its own side, except the product pre-opt, which starts
        from the relaxed reactant.
        """
        if self.is_pre_opt and self.side == "product":
            return self.rxn.reactant
        return self.node

    def find_conformer(self, node, expected_key):
        """First conformer on `node` whose key contains `expected_key`."""
        for key in node.conformers.keys():
            if expected_key in key and node.conformers[key].geo is not None:
                return node.conformers[key]
        return None

    def has_prerequisites(self) -> bool:
        node = self.source_node()
        if not node.conformers:
            return False
        return self.find_conformer(node, self.source_key()) is not None

    def _starting_conformer(self):
        """The starting geometry, or raise if it has gone missing."""
        node = self.source_node()
        initial_conf = self.find_conformer(node, self.source_key())
        if initial_conf is None:
            raise CalculatorInputError(
                f"Could not find requested geometry: {self.source_key()}"
            )
        return initial_conf

    def _patched_product_geometry(self):
        """
        Build the product pre-opt's starting structure.

        Takes the xTB-relaxed reactant and force-field patches it onto the
        product's OWN bonding. Note `product.paired_bem` is the *reactant's*
        BEM -- that is what the GSM machinery wants -- so the target here has
        to come off the product graph directly.

        Products inherit the parent's atom ordering verbatim and index-aligned
        (`canon=False` at enumeration), which is the only reason a reactant
        geometry can be reinterpreted under product bonding at all.

        Raises `CalculatorInputError` if neither force field can reach the
        product connectivity, which discards the reaction.
        """
        relaxed_reactant = self._starting_conformer()
        target_bem = self.rxn.product.graph.bond_mats[0]

        patched = joint_optimize(relaxed_reactant, target_bem, lot=self.config.bias_lot)
        if patched is None:
            raise CalculatorInputError(
                "Product pre-optimization could not patch the relaxed reactant "
                f"geometry onto the product bonding (bias_lot={self.config.bias_lot}). "
                "Neither RDKit nor Open Babel reproduced the product connectivity."
            )

        matches, n_broken, n_formed = compare_adjacency(
            patched.elements, patched.geo, self.rxn.product.graph.adj_mat
        )
        if not matches:
            raise CalculatorInputError(
                "Product pre-optimization UFF patch did not reproduce the product "
                f"graph: {describe_adjacency_change(n_broken, n_formed)}."
            )

        return patched

    def _check_preopt_adjacency(self, elements, geo) -> bool:
        """
        Compare a finished pre-optimization against the graph it should have.

        Returns True if the task should be considered successful. The reactant
        leg gates on this: a reactant that is not a minimum of its own graph
        invalidates the whole reaction. The product leg only warns, because the
        design calls for feeding the xTB geometry to CREST either way -- an
        enumerated product that is not a GFN2 minimum is a real result, not a
        failure, and discarding it would throw away chemistry.
        """
        matches, n_broken, n_formed = compare_adjacency(
            elements, geo, self.node.graph.adj_mat
        )
        if matches:
            return True

        change = describe_adjacency_change(n_broken, n_formed)
        if self.side == "reactant":
            print(f"     * Pre-optimization changed the reactant graph ({change}). "
                  f"Rejecting reaction.")
            return False

        print(f"     ! Pre-optimization changed the product graph ({change}). "
              f"Keeping it anyway and passing it to conformer generation.")
        return True

class PysisyphusMinOptCalculator(MinOptTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_name = "erm42/yarp:pysis_xtb"

    def generate_input(self):
        # The product pre-opt is the one case with no ready-made starting
        # geometry: it has to be built by patching the relaxed reactant onto
        # the product bonding first.
        if self.is_pre_opt and self.side == "product":
            initial_conf = self._patched_product_geometry()
        else:
            initial_conf = self._starting_conformer()

        input_xyz_path = self.scratch_dir / "initial_geom.xyz"
        with open(input_xyz_path, "w") as f:
            f.write(initial_conf.to_xyz_string())

        inp_path = self.scratch_dir / "min_opt.yaml"
        self._write_pysis_rp_opt_input(inp_path, "initial_geom.xyz")

    def write_submission_script(self) -> Path:
        """Write the bash script that the JobManager will execute."""
        script_path = self.scratch_dir / "run_pysis_rpopt.sh"

        env_vars = {
            "OMP_NUM_THREADS": self.config.n_cpus,
            "MKL_NUM_THREADS": self.config.n_cpus,
        }
        prefix = self.get_container_prefix(self.image_name, self.scratch_dir, env_vars=env_vars)
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

        # The pre-optimization does not request a Hessian, so there is no
        # final_hessian.h5 to look for.
        expected = [log_file, xyz_file]
        if not self.is_pre_opt:
            expected.append(self.scratch_dir / "final_hessian.h5")

        success = True

        # 1. File existence check
        if not all(path.exists() for path in expected):
            print(f"     * Run failed: Missing expected output files.")
            return False

        # 2. Log file termination check
        with open(log_file, "r") as f:
            log_text = f.read()

        if "Wrote final, hopefully optimized, geometry to" not in log_text or "pysisyphus run took" not in log_text:
            print(f"     * Run failed: Did not find successful termination message in log.")
            success = False

        # 3. Did the pre-optimization keep the graph it was given?
        #    Gates the reactant leg, warns on the product leg.
        if success and self.is_pre_opt:
            opt_elements, opt_geo = self._parse_opt_geo()
            success = self._check_preopt_adjacency(opt_elements, opt_geo)

        return success

    def scrape_data(self) -> bool:
        conf = conformer()
        conf.lot = self.config.lot
        conf.software = self.config.software
        conf.type = self.output_key()

        opt_elements, opt_geo = self._parse_opt_geo()
        conf.elements = opt_elements
        conf.geo = opt_geo

        conf.properties['internal_energy_Eh'] = self._parse_energy()

        # No Hessian is requested for the pre-optimization: nothing downstream
        # consumes its frequencies, and the geometry goes straight to CREST.
        if not self.is_pre_opt:
            hess, freq = self._parse_hessian_freq()
            conf.vibrational_freqs = freq
            conf.hessian = hess

        self.node.conformers[conf.type] = conf

        return True

    def cleanup(self):
        # Keep input, log, and final geometry; delete Hessian (scraped) and xTB calc dirs
        keep = {"min_opt.yaml", "min_opt.log", "final_geometry.xyz", "initial_geom.xyz", "run_pysis_rpopt.sh"}
        for item in self.scratch_dir.iterdir():
            if item.name not in keep:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)

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
            #
            # The pre-optimization defaults to lbfgs and skips the Hessian.
            # 'rfo' cannot optimize a free diatomic -- a linear fragment yields
            # a 7th small Hessian eigenvalue and trips a pysisyphus assertion --
            # and products shedding H2 or O2 are common. The Hessian is skipped
            # because nothing downstream reads pre-opt frequencies.
            f.write(f'opt:\n type: {self.config.opt_type}\n max_cycles: {self.config.max_cycles}\n overachieve_factor: 3\n')
            if not self.is_pre_opt:
                f.write(f' hessian_recalc: {self.config.hessian_recalc}\n do_hess: True\n')

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
        elif self.job_manager.container == "apptainer" or self.job_manager.container == "singularity":
            self.image_name = "orca_6.0.1.sif"

    def generate_input(self):
        initial_conf = self._starting_conformer()

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
            return False

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
        conf.type = self.output_key()

        xyz_file = self.scratch_dir / "min_opt.xyz"
        opt_elements, opt_geo = self._parse_opt_geo(xyz_file)
        conf.elements = opt_elements
        conf.geo = opt_geo

        log_file = self.scratch_dir / f"min_opt.out"
        conf.properties['internal_energy_Eh'] = self._parse_energy(log_file)

        enthalpy, entropy, gibbs = self._parse_orca_thermo(log_file)
        conf.properties['gibbs_free_energy_kcal_per_mol'] = gibbs
        conf.properties['enthalpy_kcal_per_mol'] = enthalpy
        conf.properties['entropy_temp_kcal_per_mol'] = entropy

        hess_file = self.scratch_dir / "min_opt.hess"
        hess, freq = self._parse_hessian_freq(hess_file)
        conf.vibrational_freqs = freq
        conf.hessian = hess

        self.node.conformers[conf.type] = conf

        return True

    def cleanup(self):
        # Keep input, log, and final geometry; delete Hessian (scraped), .gbw, .densities, etc.
        keep = {"min_opt.inp", "min_opt.out", "min_opt.xyz", "initial_geom.xyz", "run_orca_rpopt.sh"}
        for item in self.scratch_dir.iterdir():
            if item.name not in keep:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)

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
            f.write(f'  Calc_Hess true\n  Recalc_Hess {self.config.hessian_recalc}\n')
            f.write('end\n\n')

            # set XYZ input file
            f.write(f'*xyzfile {self.config.charge} {self.config.multiplicity} {input_geo_xyz}\n')
            f.write('\n# Never forget your bonus lines!!!\n')

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
