"""
Tests for the conformer-generation calculators.

`CrestConfCalculator` and `RdkitConfCalculator` sit between the xTB
pre-optimization and GSM. These tests pin where CREST reads its starting
geometry from -- the pre-optimized structure, never the raw yarpecule graph
geometry -- the MD timestep it requests when a fragment is a free diatomic, and
how the RDKit calculator hands its graph to the container and filters what
comes back.
"""
import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from rdkit import Chem

from yarp.reaction.external.calc_base import CalculatorInputError
from yarp.reaction.external.calc_factory import get_calculator
from yarp.reaction.external.conf_gen import (
    CrestConfCalculator, RdkitConfCalculator, DIATOMIC_MD_TIMESTEP_FS,
)
from yarp.reaction.reaction import reaction
from yarp.util.config import ConformerConfig


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



# =====================================================================
# RDKit as a conf_gen software option
# =====================================================================

class TestRdkitConformerConfig:
    """`software: rdkit` in the conf_gen block, validated like CREST's options."""

    def test_rdkit_with_a_force_field_lot_is_accepted(self):
        cfg = ConformerConfig(software="rdkit", lot="uff", charge=0)

        assert cfg.n_conf == 50 and cfg.prune_rms_thresh == 0.1

    def test_rdkit_does_not_need_n_unpaired_electrons(self):
        """That key exists for CREST's --uhf; RDKit has no use for it."""
        ConformerConfig(software="rdkit", lot="mmff94", charge=0)

    def test_rdkit_rejects_a_crest_lot(self):
        with pytest.raises(ValueError, match="RDKit"):
            ConformerConfig(software="rdkit", lot="gfn2", charge=0)

    def test_crest_rejects_a_force_field_lot(self):
        with pytest.raises(ValueError, match="CREST"):
            ConformerConfig(software="crest", lot="uff", charge=0, n_unpaired_electrons=0)

    def test_unknown_software_is_still_rejected(self):
        with pytest.raises(ValueError, match="Invalid 'software'"):
            ConformerConfig(software="auto3d", lot="uff", charge=0)


def _rdkit_calc(rxn, task_type, lot="uff"):
    config = ConformerConfig(software="rdkit", lot=lot, charge=0, n_cpus=2)
    task_def = SimpleNamespace(task_type=task_type, config=config, task_id=f"s.{task_type}")
    return RdkitConfCalculator(task_def, rxn, MagicMock())


def _add_preopt(state, geo=None):
    graph = state.graph
    geo = graph.geo.copy() if geo is None else geo
    state.conformers["preopt_xtb_pysisyphus"] = _conf(geo, graph.elements, "preopt_xtb_pysisyphus")


def _write_xyz(path, elements, geometries):
    with open(path, "w") as f:
        for geo in geometries:
            f.write(f"{len(elements)}\n\n")
            for el, (x, y, z) in zip(elements, geo):
                f.write(f"{el.capitalize()} {x} {y} {z}\n")


class TestRdkitRouting:

    def test_factory_routes_rdkit_to_the_rdkit_calculator(self, khp_reaction):
        config = ConformerConfig(software="rdkit", lot="uff", charge=0)
        task_def = SimpleNamespace(task_type="reactant_conformer", config=config, task_id="s.r")

        assert isinstance(get_calculator(task_def, khp_reaction, MagicMock()), RdkitConfCalculator)


class TestRdkitGenerateInput:
    """
    classy_yarp wrote its own MOL file and had RDKit re-read it. Here the mol
    comes from yarpecule_to_rdmol, so the handoff file must carry the graph's
    atoms and bonds, index-aligned, with the pre-optimized coordinates.
    """

    def test_mol_file_carries_the_graph_and_the_pre_opt_geometry(self, khp_reaction, tmp_path):
        graph = khp_reaction.reactant.graph
        shifted = graph.geo + 1.5
        _add_preopt(khp_reaction.reactant, geo=shifted)

        calc = _rdkit_calc(khp_reaction, "reactant_conformer")
        calc.set_scratch_dir(tmp_path)
        calc.generate_input()

        mol = Chem.MolFromMolFile(str(tmp_path / "input.mol"), removeHs=False)
        assert [a.GetSymbol().lower() for a in mol.GetAtoms()] == list(graph.elements)
        assert np.array_equal(Chem.GetAdjacencyMatrix(mol), graph.adj_mat)
        assert np.allclose(mol.GetConformer().GetPositions(), shifted, atol=1e-4)

    def test_missing_pre_opt_raises_rather_than_writing_nothing(self, khp_reaction, tmp_path):
        calc = _rdkit_calc(khp_reaction, "product_conformer")
        calc.set_scratch_dir(tmp_path)

        with pytest.raises(CalculatorInputError, match="pre-optimization"):
            calc.generate_input()

    def test_command_passes_the_config_to_the_container(self, khp_reaction, tmp_path):
        calc = _rdkit_calc(khp_reaction, "reactant_conformer", lot="mmff94")
        calc.set_scratch_dir(tmp_path)
        calc.get_container_prefix = MagicMock(return_value="PREFIX")
        calc.write_scheduler_headers = MagicMock()

        script = calc.write_submission_script().read_text()

        # The image's entrypoint is the script, so apptainer must use `run`.
        assert calc.get_container_prefix.call_args.kwargs["apptainer_run"] is True
        assert ("PREFIX input.mol --lot mmff94 --n_conf 50 --prune_rms_thresh 0.1 "
                "--n_threads 2 --seed 42") in script


class TestRdkitScrapeData:
    """
    classy_yarp's connectivity filter: a conformer whose perceived bonding
    differs from the yarpecule graph is dropped, and survivors keep the
    container's energy order.
    """

    def _squashed(self, geo):
        """Pull every atom toward the centroid until bonds form that the graph lacks."""
        centroid = geo.mean(axis=0)
        return centroid + 0.3 * (geo - centroid)

    def test_drops_broken_conformers_and_ranks_the_rest_in_order(self, khp_reaction, tmp_path):
        graph = khp_reaction.reactant.graph
        good, broken = graph.geo, self._squashed(graph.geo)
        _write_xyz(tmp_path / "rdkit_conformers.xyz", graph.elements, [good, broken, good + 2.0])

        calc = _rdkit_calc(khp_reaction, "reactant_conformer")
        calc.set_scratch_dir(tmp_path)

        assert calc.scrape_data()
        confs = khp_reaction.reactant.conformers
        assert {k for k in confs if k.startswith("conf_gen")} == {
            "conf_gen_rank0_uff_rdkit", "conf_gen_rank1_uff_rdkit"}
        # Rank 1 is the third frame -- the broken second frame left no gap.
        assert np.allclose(confs["conf_gen_rank1_uff_rdkit"].geo, good + 2.0)
        assert confs["conf_gen_rank0_uff_rdkit"].software == "rdkit"

    def test_fails_when_no_conformer_keeps_the_bonding(self, khp_reaction, tmp_path):
        graph = khp_reaction.reactant.graph
        _write_xyz(tmp_path / "rdkit_conformers.xyz", graph.elements, [self._squashed(graph.geo)])

        calc = _rdkit_calc(khp_reaction, "reactant_conformer")
        calc.set_scratch_dir(tmp_path)

        assert not calc.scrape_data()
        assert not any(k.startswith("conf_gen") for k in khp_reaction.reactant.conformers)

