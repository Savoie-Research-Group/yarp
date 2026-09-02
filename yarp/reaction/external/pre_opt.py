"""Geometry-preparation calculators for xTB -> UFF-joint -> xTB workflows."""

import re
import shutil
from pathlib import Path

from yarp.reaction.conformer import conformer
from yarp.reaction.conf_sampling.joint_opt import joint_optimize
from yarp.reaction.external.calc_base import AsyncYarpCalculator
from yarp.yarpecule.input_parsers import xyz_parse


class PysisyphusPreOptCalculator(AsyncYarpCalculator):
    """Run a geometry-only xTB optimization before GSM generation.

    Reactant tasks optimize the reaction object's initial reactant geometry.
    Product tasks first project that optimized reactant onto the product BEM
    with a UFF joint optimization, then optimize the projected product at xTB.
    The final conformer deliberately uses the normal ``rpopt_*`` key so that
    the existing refinement and IRC code can reuse it without another R/P
    optimization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_name = "erm42/yarp:pysis_xtb"

    @property
    def _key(self):
        return f"rpopt_{self.config.lot}_{self.config.software}"

    @property
    def _is_reactant(self):
        return self.task_def.task_type == "reactant_optimization"

    def _find_preoptimized_reactant(self):
        for key, conf in self.rxn.reactant.conformers.items():
            if self._key in key and conf.geo is not None:
                return conf
        return None

    def has_prerequisites(self):
        if self._is_reactant:
            return self.rxn.reactant.conformers.get("initial_geom") is not None
        return self._find_preoptimized_reactant() is not None

    def generate_input(self):
        if self._is_reactant:
            initial_conf = self.rxn.reactant.conformers["initial_geom"]
        else:
            reactant_conf = self._find_preoptimized_reactant()
            joint_cfg = self.config.joint_opt
            initial_conf = joint_optimize(
                reactant_conf,
                self.rxn.reactant.paired_bem,
                lot=joint_cfg.lot.lower(),
                maxiter=joint_cfg.maxiter,
            )
            if initial_conf is None:
                # ``progress_yarp`` treats a missing valid output as a normal
                # failed task.  Record the reason and submit a failing stub
                # instead of raising here, which would terminate the manager.
                (self.scratch_dir / "pre_opt.error").write_text(
                    "UFF joint optimization could not produce a product geometry "
                    "with the enumerated product adjacency.\n"
                )
                return

            # Retain the exact UFF-projected starting geometry for inspection.
            initial_conf.type = "joint_uff_from_reactant"
            initial_conf.lot = joint_cfg.lot
            initial_conf.software = "rdkit_or_openbabel"
            self.rxn.product.conformers[initial_conf.type] = initial_conf

        with open(self.scratch_dir / "initial_geom.xyz", "w") as handle:
            handle.write(initial_conf.to_xyz_string())
        self._write_pysis_input(self.scratch_dir / "pre_opt.yaml", "initial_geom.xyz")

    def write_submission_script(self):
        script_path = self.scratch_dir / "run_pysis_preopt.sh"
        env_vars = {
            "OMP_NUM_THREADS": self.config.n_cpus,
            "MKL_NUM_THREADS": self.config.n_cpus,
        }
        with open(script_path, "w") as handle:
            handle.write("#!/bin/bash\n")
            self.write_scheduler_headers(handle)
            if (self.scratch_dir / "pre_opt.error").exists():
                handle.write(f"cat {self.scratch_dir / 'pre_opt.error'} >&2\nexit 1\n")
                script_path.chmod(0o755)
                return script_path
            prefix = self.get_container_prefix(self.image_name, self.scratch_dir, env_vars=env_vars)
            handle.write(f"cd {self.scratch_dir}\n")
            handle.write(f"{prefix} pysis pre_opt.yaml > pre_opt.log 2> pre_opt.err\n")
        script_path.chmod(0o755)
        return script_path

    def check_output(self):
        error_file = self.scratch_dir / "pre_opt.error"
        if error_file.exists():
            print(f"     * Geometry preparation failed: {error_file.read_text().strip()}")
            return False
        log_file = self.scratch_dir / "pre_opt.log"
        xyz_file = self.scratch_dir / "final_geometry.xyz"
        if not (log_file.exists() and xyz_file.exists()):
            return False
        log_text = log_file.read_text(errors="replace")
        return (
            "Wrote final, hopefully optimized, geometry to" in log_text
            and "pysisyphus run took" in log_text
        )

    def scrape_data(self):
        elements, geometry = xyz_parse(self.scratch_dir / "final_geometry.xyz", multiple=False)
        conf = conformer()
        conf.elements = elements
        conf.geo = geometry
        conf.lot = self.config.lot
        conf.software = self.config.software
        conf.type = self._key
        conf.properties["internal_energy_Eh"] = self._parse_energy()

        if self._is_reactant:
            self.rxn.reactant.conformers[conf.type] = conf
        else:
            self.rxn.product.conformers[conf.type] = conf
        return True

    def cleanup(self):
        keep = {
            "pre_opt.yaml", "pre_opt.log", "pre_opt.err", "initial_geom.xyz",
            "final_geometry.xyz", "run_pysis_preopt.sh",
        }
        for item in self.scratch_dir.iterdir():
            if item.name in keep:
                continue
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    def _write_pysis_input(self, input_path: Path, geometry_file: str):
        with open(input_path, "w") as handle:
            handle.write(f"geom:\n type: cart\n fn: {geometry_file}\n")
            handle.write(
                "calc:\n"
                f" type: {self.config.lot}\n"
                f" pal: {self.config.n_cpus}\n"
                f" mem: {self.config.mem_per_cpu}\n"
                f" charge: {self.config.charge}\n"
                f" mult: {self.config.multiplicity}\n"
            )
            handle.write(
                "opt:\n"
                f" type: rfo\n max_cycles: {self.config.max_cycles}\n"
                " overachieve_factor: 3\n do_hess: False\n"
            )

    def _parse_energy(self):
        log_text = (self.scratch_dir / "pre_opt.log").read_text(errors="replace")
        matches = re.findall(r"energy:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+hartree", log_text)
        return float(matches[-1]) if matches else None
