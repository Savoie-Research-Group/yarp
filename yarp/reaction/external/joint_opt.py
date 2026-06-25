"""
Joint optimization engine adapters used during TS guess pair selection.

These classes are intentionally not AsyncYarpCalculator tasks. Joint
optimization remains an implementation detail of TS guess generation.
"""

from pathlib import Path

from yarp.reaction.conf_sampling.joint_opt import ob_joint_optimize
from yarp.reaction.conf_sampling.joint_opt import xtb_joint_optimize


class JointOptimizationEngine:
    """Base interface for biased endpoint generation."""

    def optimize(self, conformer, target_bem, label):
        raise NotImplementedError


class OpenBabelJointOptimizationEngine(JointOptimizationEngine):
    """OpenBabel force-field joint optimization."""

    def __init__(self, ff_name="uff"):
        self.ff_name = ff_name

    def optimize(self, conformer, target_bem, label=None):
        return ob_joint_optimize(conformer, target_bem, self.ff_name)


class XTBJointOptimizationEngine(JointOptimizationEngine):
    """Constrained xTB joint optimization."""

    def __init__(self, scratch_dir, config):
        self.scratch_dir = Path(scratch_dir)
        self.config = config

    def optimize(self, conformer, target_bem, label):
        return xtb_joint_optimize(
            conformer,
            target_bem,
            self.scratch_dir / label,
            lot=self.config.xtb_joint_lot,
            charge=self.config.charge,
            multiplicity=self.config.multiplicity,
            n_cpus=self.config.n_cpus,
            force_constant=self.config.xtb_joint_force_constant,
            scf_iters=self.config.xtb_joint_scf_iters,
            keep_files=self.config.xtb_joint_keep_files,
        )


def make_joint_optimization_engine(config, scratch_dir):
    """Build the configured joint optimization engine."""
    engine = getattr(config, "joint_opt_engine", "ob")
    if engine == "xtb":
        return XTBJointOptimizationEngine(scratch_dir, config)
    return OpenBabelJointOptimizationEngine(config.bias_lot)
