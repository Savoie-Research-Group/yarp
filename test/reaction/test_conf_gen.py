"""
Tests for the CREST conformer-generation calculator.

`CrestConfCalculator` sits between the xTB pre-optimization and GSM. These tests
pin where it reads its starting geometry from -- the pre-optimized structure,
never the raw yarpecule graph geometry -- and the MD timestep it requests when a
fragment is a free diatomic.
"""
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from yarp.reaction.external.calc_base import CalculatorInputError
from yarp.reaction.external.conf_gen import CrestConfCalculator, DIATOMIC_MD_TIMESTEP_FS
from yarp.reaction.reaction import reaction


def _conf(geo, elements, ctype):
    return SimpleNamespace(geo=geo, elements=elements, type=ctype,
                           to_xyz_string=lambda: "stub")



# =====================================================================
# Handoff from the pre-optimization into conformer generation
# =====================================================================

class TestCrestReadsThePreOptGeometry:
    """
    CREST used to start from `initial_geom`, the raw yarpecule graph geometry.
    For an enumerated product that is the parent's coordinates under the
    product's bonding. If CREST kept reading it, the pre-optimization would run
    and then be thrown away.
    """

    def _calc(self, rxn, task_type):
        task_def = SimpleNamespace(task_type=task_type, config=MagicMock(), task_id=f"s.{task_type}")
        return CrestConfCalculator(task_def, rxn, MagicMock())

    def test_not_ready_before_the_pre_opt_runs(self, khp_reaction):
        assert not self._calc(khp_reaction, "reactant_conformer").has_prerequisites()

    def test_initial_geom_alone_is_not_enough(self, khp_reaction):
        """A fresh reaction already has initial_geom; that must no longer satisfy it."""
        assert "initial_geom" in khp_reaction.reactant.conformers

        assert not self._calc(khp_reaction, "reactant_conformer").has_prerequisites()

    def test_ready_once_its_own_side_is_pre_optimized(self, khp_reaction):
        graph = khp_reaction.reactant.graph
        khp_reaction.reactant.conformers["preopt_xtb_pysisyphus"] = _conf(
            graph.geo.copy(), graph.elements, "preopt_xtb_pysisyphus")

        assert self._calc(khp_reaction, "reactant_conformer").has_prerequisites()

    def test_reactant_does_not_wait_on_the_product_leg(self, khp_reaction):
        """
        The product pre-opt depends on the reactant pre-opt, so it finishes
        later. Requiring both sides here would fail the reactant's conformer
        task the moment its own dependency was met.
        """
        graph = khp_reaction.reactant.graph
        khp_reaction.reactant.conformers["preopt_xtb_pysisyphus"] = _conf(
            graph.geo.copy(), graph.elements, "preopt_xtb_pysisyphus")

        assert self._calc(khp_reaction, "reactant_conformer").has_prerequisites()
        assert not self._calc(khp_reaction, "product_conformer").has_prerequisites()

    def test_writes_the_pre_opt_geometry_not_the_graph_geometry(self, khp_reaction, tmp_path):
        graph = khp_reaction.reactant.graph
        relaxed = _conf(graph.geo.copy(), graph.elements, "preopt_xtb_pysisyphus")
        relaxed.to_xyz_string = lambda: "RELAXED"
        khp_reaction.reactant.conformers["preopt_xtb_pysisyphus"] = relaxed

        calc = self._calc(khp_reaction, "reactant_conformer")
        calc.set_scratch_dir(tmp_path)
        calc.generate_input()

        assert (tmp_path / "input.xyz").read_text() == "RELAXED"

    def test_missing_pre_opt_raises_rather_than_writing_nothing(self, khp_reaction, tmp_path):
        calc = self._calc(khp_reaction, "product_conformer")
        calc.set_scratch_dir(tmp_path)

        with pytest.raises(CalculatorInputError, match="pre-optimization"):
            calc.generate_input()



# =====================================================================
# CREST MD timestep for free diatomics
# =====================================================================

class TestCrestDiatomicTimestep:
    """
    CREST's default 5 fs metadynamics is only stable because SHAKE constrains
    the bonds, and SHAKE can only constrain bonds xtb perceives. An xTB-relaxed
    free H2 sits at 0.7750 A, just outside the ~0.768 A H-H perception cutoff,
    so it is never constrained and the MD diverges. Measured: every --tstep
    variant survives, every 5 fs variant fails, and --shake 1 also fails --
    confirming the issue is topology, not SHAKE mode.
    """

    def _calc(self, rxn, task_type, lot="gfn2", n_cpus=4):
        config = SimpleNamespace(lot=lot, n_cpus=n_cpus, charge=0,
                                 n_unpaired_electrons=0, seed=42, solvent=None)
        task_def = SimpleNamespace(task_type=task_type, config=config,
                                   task_id=f"s.{task_type}")
        return CrestConfCalculator(task_def, rxn, MagicMock())

    def test_noopt_is_gone(self, khp_reaction):
        """
        --noopt was the first attempt at this and does not work: the failure
        reproduces with and without it, and keeping it costs conformers.
        """
        cmd = self._calc(khp_reaction, "reactant_conformer")._get_crest_command()

        assert "--noopt" not in cmd

    def test_no_timestep_flag_without_a_diatomic(self, khp_reaction):
        """The ~70% of systems with no diatomic keep CREST's default 5 fs."""
        calc = self._calc(khp_reaction, "reactant_conformer")

        assert not calc.has_free_diatomic()
        assert "--tstep" not in calc._get_crest_command()

    def test_timestep_flag_when_a_diatomic_is_present(self, khp_parent, khp_products):
        """O=C=CCOO.[H][H] carries a free H2 -- this is the failing case."""
        product = khp_products["O=C=CCOO.[H][H]"]
        rxn = reaction(khp_parent, product)
        calc = self._calc(rxn, "product_conformer")

        assert calc.has_free_diatomic()
        assert f"--tstep {DIATOMIC_MD_TIMESTEP_FS}" in calc._get_crest_command()

    def test_the_reactant_side_of_that_reaction_is_unaffected(self, khp_parent, khp_products):
        """
        The H2 is on the product side only. Flagging the reactant too would
        pay the runtime cost for nothing.
        """
        rxn = reaction(khp_parent, khp_products["O=C=CCOO.[H][H]"])
        calc = self._calc(rxn, "reactant_conformer")

        assert not calc.has_free_diatomic()
        assert "--tstep" not in calc._get_crest_command()

    def test_timestep_is_short_enough_to_integrate_an_H2_stretch(self):
        """
        H2 stretches near 4400 cm-1, a period of ~7.6 fs. Stable integration
        wants roughly ten steps per period; 2.0 fs gives ~3.8 and was measured
        working but marginal, so the default must stay at or below 1.0.
        """
        assert DIATOMIC_MD_TIMESTEP_FS <= 1.0

