"""Container-backed biased endpoint generation for conformer pair selection.

The host process owns reaction/conformer bookkeeping. The ``joint_opt`` image
owns the OpenBabel and xTB executables and receives a batch of independent
joint-optimization requests over a small JSON protocol.
"""

import copy
import json
import subprocess
from pathlib import Path
from typing import NamedTuple

import numpy as np

from yarp.reaction.external.model_scorer import get_container_prefix
from yarp.yarpecule.lewis.bem_score import return_formals
from yarp.util.properties import el_radii


class XTBConstraint(NamedTuple):
    """One xTB distance constraint using xTB's one-indexed atom numbers."""

    atom_i: int
    atom_j: int
    distance: float


def bem_to_distance_constraints(elements, target_bem):
    """Convert a target BEM into the xTB constraints sent to the container."""
    target_bem = np.asarray(target_bem)
    if target_bem.ndim != 2 or target_bem.shape[0] != target_bem.shape[1]:
        raise ValueError(f"target_bem must be a square matrix, got shape {target_bem.shape}")
    if len(elements) != target_bem.shape[0]:
        raise ValueError(
            f"elements length ({len(elements)}) does not match target_bem size ({target_bem.shape[0]})"
        )

    constraints = []
    for atom_i in range(target_bem.shape[0] - 1):
        for atom_j in range(atom_i + 1, target_bem.shape[1]):
            if target_bem[atom_i, atom_j] <= 0:
                continue

            radius_i = el_radii.get(elements[atom_i], el_radii.get(str(elements[atom_i]).capitalize()))
            radius_j = el_radii.get(elements[atom_j], el_radii.get(str(elements[atom_j]).capitalize()))
            if radius_i is None or radius_j is None:
                raise KeyError(
                    f"Missing covalent radius for constrained pair {elements[atom_i]}-{elements[atom_j]}"
                )
            constraints.append(XTBConstraint(atom_i + 1, atom_j + 1, float(radius_i + radius_j)))

    return constraints


class JointOptimizationEngine:
    """Base interface for biased endpoint generation."""

    def optimize_many(self, conformers, target_bem, labels):
        raise NotImplementedError


class ContainerJointOptimizationEngine(JointOptimizationEngine):
    """Run one batch of OpenBabel or xTB biased optimizations in a container."""

    image_name = "erm42/yarp:joint_opt"

    def __init__(self, job_manager, scratch_dir, config, log=None):
        if job_manager is None:
            raise ValueError("Container joint optimization requires a job_manager configuration")

        self.job_manager = job_manager
        self.scratch_dir = Path(scratch_dir)
        self.config = config
        self.log = log
        self.engine = config.joint_opt_engine
        self.image_name = getattr(config, "joint_opt_image", None) or self.image_name
        self.calls = 0

    def optimize_many(self, conformers, target_bem, labels):
        """Optimize a conformer batch in one container process."""
        if len(conformers) != len(labels):
            raise ValueError("Each joint-optimization conformer must have exactly one label")
        if not conformers:
            return []

        self.calls += 1
        call_dir = self.scratch_dir / f"batch_{self.calls:04d}"
        call_dir.mkdir(parents=True, exist_ok=True)
        input_path = call_dir / "input.json"
        output_path = call_dir / "output.json"
        stdout_path = call_dir / "joint_opt.out"
        stderr_path = call_dir / "joint_opt.err"

        payload = {
            "protocol_version": 1,
            "jobs": [
                self._make_job(conformer, target_bem, label)
                for conformer, label in zip(conformers, labels)
            ],
        }
        input_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        prefix = get_container_prefix(self.job_manager, self.image_name, str(call_dir))
        cmd = " ".join(
            [
                prefix,
                "python",
                "/opt/yarp_joint_opt/joint_opt.py",
                "--input",
                "/work/input.json",
                "--output",
                "/work/output.json",
            ]
        )
        self._log(f"running {len(conformers)} {self.engine} joint optimization(s) in {self.image_name}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        stdout_path.write_text(result.stdout or "", encoding="utf-8")
        stderr_path.write_text(result.stderr or "", encoding="utf-8")

        if result.returncode != 0:
            raise RuntimeError(
                "Joint-optimization container failed. "
                f"See {stdout_path} and {stderr_path}."
            )
        if not output_path.exists():
            raise RuntimeError(f"Joint-optimization container did not write {output_path}")

        output = json.loads(output_path.read_text(encoding="utf-8"))
        returned = output.get("results")
        if not isinstance(returned, list):
            raise RuntimeError(f"Joint-optimization container wrote an invalid result to {output_path}")

        result_by_label = {}
        for item in returned:
            label = item.get("label") if isinstance(item, dict) else None
            if label in result_by_label:
                raise RuntimeError(f"Joint-optimization container returned duplicate label {label!r}")
            result_by_label[label] = item

        expected_labels = set(labels)
        if set(result_by_label) != expected_labels:
            raise RuntimeError(
                "Joint-optimization container response labels did not match its input. "
                f"Expected {sorted(expected_labels)}, got {sorted(result_by_label)}."
            )

        optimized = []
        for conformer, label in zip(conformers, labels):
            item = result_by_label[label]
            if not item.get("success"):
                self._log(f"{label} did not converge: {item.get('error', 'unknown error')}")
                optimized.append(None)
                continue
            optimized.append(self._make_biased_conformer(conformer, item, label))
        return optimized

    def _make_job(self, conformer, target_bem, label):
        job = {
            "label": label,
            "engine": self.engine,
            "elements": list(conformer.elements),
            "geo": np.asarray(conformer.geo, dtype=float).tolist(),
            "target_bem": np.asarray(target_bem, dtype=float).tolist(),
            "formal_charges": [
                int(charge)
                for charge in return_formals(np.asarray(target_bem), conformer.elements)
            ],
            "radical_atoms": [
                atom_index
                for atom_index in range(len(conformer.elements))
                if int(np.asarray(target_bem)[atom_index, atom_index]) % 2 == 1
            ],
            "options": {
                "keep_files": bool(self.config.xtb_joint_keep_files),
            },
        }
        if self.engine == "ob":
            job["options"]["ff_name"] = self.config.bias_lot
        elif self.engine == "xtb":
            job["constraints"] = [
                constraint._asdict()
                for constraint in bem_to_distance_constraints(conformer.elements, target_bem)
            ]
            job["options"].update(
                {
                    "lot": self.config.xtb_joint_lot,
                    "charge": self.config.charge,
                    "multiplicity": self.config.multiplicity,
                    "n_cpus": self.config.n_cpus,
                    "force_constant": self.config.xtb_joint_force_constant,
                    "scf_iters": self.config.xtb_joint_scf_iters,
                }
            )
        else:
            raise ValueError(f"Unsupported joint optimization engine: {self.engine}")
        return job

    def _make_biased_conformer(self, conformer, result, label):
        geometry = np.asarray(result.get("geo"), dtype=float)
        expected_shape = np.asarray(conformer.geo).shape
        if geometry.shape != expected_shape:
            raise RuntimeError(
                f"Joint-optimization result for {label} has geometry shape {geometry.shape}; "
                f"expected {expected_shape}."
            )

        biased = copy.deepcopy(conformer)
        biased.geo = geometry
        if self.engine == "xtb":
            biased.type = f"biased_xtb_{conformer.type}"
            biased.lot = self.config.xtb_joint_lot
            biased.software = "xtb"
        else:
            biased.type = f"biased_{conformer.type}"
        return biased

    def _log(self, message):
        if self.log is not None:
            print(f"[ContainerJointOptimizationEngine] {message}", file=self.log, flush=True)


def make_joint_optimization_engine(config, scratch_dir, job_manager, log=None):
    """Build the container-only joint optimization engine."""
    return ContainerJointOptimizationEngine(job_manager, scratch_dir, config, log=log)
