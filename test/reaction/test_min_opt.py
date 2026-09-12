"""
Tests for the shared reactant/product minimization calculator.

`MinOptTask` serves four task types: the two xTB pre-optimization legs that run
ahead of conformer generation, and the two refinement legs that run after it.
They are the same operation -- minimize a structure -- so they share one
calculator, and the per-task-type dispatch that used to be copy-pasted through
the module now lives in the base class. These tests pin that dispatch, since
getting it wrong silently writes a geometry onto the wrong state.
"""
import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from yarp.reaction.external.calc_base import CalculatorInputError
from yarp.reaction.external.min_opt import MinOptTask, PysisyphusMinOptCalculator
from yarp.reaction.reaction import reaction
from yarp.util.config import PreOptConfig, RPOptConfig, InitialGeomConfig, GeomSourceConfig


@pytest.fixture
def khp_reaction(khp_parent, khp_products):
    """A real reaction object: KHP -> one of its break-2/form-2 products."""
    product = khp_products["C=COCOO"]
    return reaction(khp_parent, product)


def make_calc(rxn, task_type, config, cls=MinOptTask):
    task_def = SimpleNamespace(task_type=task_type, config=config, task_id=f"s.{task_type}")
    return cls(task_def, rxn, MagicMock())


@pytest.fixture
def preopt_config():
    return PreOptConfig()


@pytest.fixture
def rpopt_config():
    source = GeomSourceConfig(label="conf_gen", lot="gfn2", software="crest")
    cfg = RPOptConfig(software="pysisyphus", lot="xtb", charge=0, multiplicity=1)
    cfg.initial_geom = InitialGeomConfig(reactant=source, product=source, transition_state=
                                         GeomSourceConfig(label="ts_guess", lot="xtb", software="pysisyphus"))
    return cfg


class TestTaskDispatch:
    @pytest.mark.parametrize("task_type, expected", [
        ("reactant_pre_opt", "reactant"),
        ("product_pre_opt", "product"),
        ("reactant_optimization", "reactant"),
        ("product_optimization", "product"),
    ])
    def test_side(self, khp_reaction, preopt_config, task_type, expected):
        assert make_calc(khp_reaction, task_type, preopt_config).side == expected

    @pytest.mark.parametrize("task_type, expected", [
        ("reactant_pre_opt", True),
        ("product_pre_opt", True),
        ("reactant_optimization", False),
        ("product_optimization", False),
    ])
    def test_is_pre_opt(self, khp_reaction, preopt_config, task_type, expected):
        assert make_calc(khp_reaction, task_type, preopt_config).is_pre_opt is expected

    def test_unknown_task_type_is_rejected(self, khp_reaction, preopt_config):
        calc = make_calc(khp_reaction, "transition_state_optimization", preopt_config)

        with pytest.raises(ValueError, match="Unknown task type"):
            calc.side

    def test_node_points_at_the_right_state(self, khp_reaction, preopt_config):
        r_calc = make_calc(khp_reaction, "reactant_pre_opt", preopt_config)
        p_calc = make_calc(khp_reaction, "product_pre_opt", preopt_config)

        assert r_calc.node is khp_reaction.reactant
        assert p_calc.node is khp_reaction.product


class TestConformerKeys:
    def test_pre_opt_writes_its_own_key(self, khp_reaction, preopt_config):
        calc = make_calc(khp_reaction, "reactant_pre_opt", preopt_config)

        assert calc.output_key() == "preopt_xtb_pysisyphus"

    def test_refine_key_is_unchanged(self, khp_reaction, rpopt_config):
        calc = make_calc(khp_reaction, "reactant_optimization", rpopt_config)

        assert calc.output_key() == "rpopt_xtb_pysisyphus"

    def test_reactant_pre_opt_starts_from_the_graph_geometry(self, khp_reaction, preopt_config):
        calc = make_calc(khp_reaction, "reactant_pre_opt", preopt_config)

        assert calc.source_key() == "initial_geom"
        assert calc.source_node() is khp_reaction.reactant

    def test_product_pre_opt_starts_from_the_RELAXED_REACTANT(self, khp_reaction, preopt_config):
        """
        The product leg patches the reactant's optimized geometry onto the
        product bonding, so its source is the reactant state, not the product.
        """
        calc = make_calc(khp_reaction, "product_pre_opt", preopt_config)

        assert calc.source_key() == "preopt_xtb_pysisyphus"
        assert calc.source_node() is khp_reaction.reactant
        assert calc.node is khp_reaction.product


class TestPrerequisites:
    def test_reactant_pre_opt_ready_from_a_fresh_reaction(self, khp_reaction, preopt_config):
        """A brand new reaction already carries 'initial_geom' on both states."""
        calc = make_calc(khp_reaction, "reactant_pre_opt", preopt_config)

        assert calc.has_prerequisites()

    def test_product_pre_opt_waits_for_the_reactant_leg(self, khp_reaction, preopt_config):
        calc = make_calc(khp_reaction, "product_pre_opt", preopt_config)

        assert not calc.has_prerequisites()

    def test_product_pre_opt_ready_once_the_reactant_is_relaxed(self, khp_reaction, preopt_config):
        relaxed = MagicMock()
        relaxed.geo = khp_reaction.reactant.graph.geo.copy()
        khp_reaction.reactant.conformers["preopt_xtb_pysisyphus"] = relaxed

        calc = make_calc(khp_reaction, "product_pre_opt", preopt_config)

        assert calc.has_prerequisites()


class TestProductPatchFailure:
    """
    `generate_input` is the only place a reaction can be discarded between the
    pre-flight check and the job running, and `CalculatorInputError` is the
    channel. Without it a failed patch would either submit a job against a
    stale file or take down the whole progress_yarp invocation.
    """

    def _prepare(self, rxn, preopt_config, geo):
        conf = SimpleNamespace(geo=geo, elements=rxn.reactant.graph.elements,
                               type="preopt_xtb_pysisyphus")
        rxn.reactant.conformers["preopt_xtb_pysisyphus"] = conf
        return make_calc(rxn, "product_pre_opt", preopt_config, cls=PysisyphusMinOptCalculator)

    def test_raises_when_the_patch_cannot_reach_the_product_graph(
        self, khp_reaction, preopt_config, monkeypatch
    ):
        calc = self._prepare(khp_reaction, preopt_config, khp_reaction.reactant.graph.geo.copy())
        monkeypatch.setattr("yarp.reaction.external.min_opt.joint_optimize", lambda *a, **k: None)

        with pytest.raises(CalculatorInputError, match="could not patch"):
            calc._patched_product_geometry()

    def test_raises_when_the_patch_lands_on_the_wrong_graph(
        self, khp_reaction, preopt_config, monkeypatch
    ):
        """
        `joint_optimize` returning a geometry is not sufficient -- it has to be
        a geometry with the product's bonding.
        """
        calc = self._prepare(khp_reaction, preopt_config, khp_reaction.reactant.graph.geo.copy())

        # Hand back an obviously wrong structure: every atom on top of another.
        bogus = SimpleNamespace(
            geo=np.zeros_like(khp_reaction.product.graph.geo),
            elements=khp_reaction.product.graph.elements,
        )
        monkeypatch.setattr("yarp.reaction.external.min_opt.joint_optimize", lambda *a, **k: bogus)

        with pytest.raises(CalculatorInputError, match="did not reproduce the product graph"):
            calc._patched_product_geometry()

    def test_patches_against_the_products_own_bem_not_the_paired_one(
        self, khp_reaction, preopt_config, monkeypatch
    ):
        """
        `product.paired_bem` is the REACTANT's BEM -- that is what the GSM
        machinery wants. Patching against it here would build the reactant
        again instead of the product.
        """
        calc = self._prepare(khp_reaction, preopt_config, khp_reaction.reactant.graph.geo.copy())

        seen = {}

        def spy(conf, target_bem, lot="uff"):
            seen["bem"] = target_bem
            return None

        monkeypatch.setattr("yarp.reaction.external.min_opt.joint_optimize", spy)
        with pytest.raises(CalculatorInputError):
            calc._patched_product_geometry()

        assert np.array_equal(seen["bem"], khp_reaction.product.graph.bond_mats[0])
        assert not np.array_equal(seen["bem"], khp_reaction.product.paired_bem)


class TestPostOptAdjacencyPolicy:
    """
    Three adjacency checks, two of which are gates. The reactant leg rejects
    the reaction if the optimization walked off its graph; the product leg
    keeps going, because an enumerated product that is not a GFN2 minimum is a
    real result and the design calls for handing it to CREST regardless.
    """

    def test_reactant_leg_accepts_a_matching_graph(self, khp_reaction, preopt_config):
        calc = make_calc(khp_reaction, "reactant_pre_opt", preopt_config)
        graph = khp_reaction.reactant.graph

        assert calc._check_preopt_adjacency(graph.elements, graph.geo)

    def test_reactant_leg_rejects_a_changed_graph(self, khp_reaction, preopt_config, capsys):
        calc = make_calc(khp_reaction, "reactant_pre_opt", preopt_config)
        graph = khp_reaction.reactant.graph

        assert not calc._check_preopt_adjacency(graph.elements, np.zeros_like(graph.geo))
        assert "Rejecting reaction" in capsys.readouterr().out

    def test_product_leg_keeps_a_changed_graph(self, khp_reaction, preopt_config, capsys):
        calc = make_calc(khp_reaction, "product_pre_opt", preopt_config)
        graph = khp_reaction.product.graph

        assert calc._check_preopt_adjacency(graph.elements, np.zeros_like(graph.geo))
        assert "Keeping it anyway" in capsys.readouterr().out


class TestPysisInputBlock:
    def _written(self, tmp_path, calc):
        path = tmp_path / "min_opt.yaml"
        calc._write_pysis_rp_opt_input(path, "initial_geom.xyz")
        return path.read_text()

    def test_pre_opt_uses_lbfgs_and_skips_the_hessian(self, tmp_path, khp_reaction, preopt_config):
        calc = make_calc(khp_reaction, "reactant_pre_opt", preopt_config,
                         cls=PysisyphusMinOptCalculator)
        text = self._written(tmp_path, calc)

        assert "type: lbfgs" in text
        assert "do_hess" not in text
        assert "hessian_recalc" not in text

    def test_refine_still_uses_rfo_and_computes_a_hessian(self, tmp_path, khp_reaction, rpopt_config):
        calc = make_calc(khp_reaction, "reactant_optimization", rpopt_config,
                         cls=PysisyphusMinOptCalculator)
        text = self._written(tmp_path, calc)

        assert "type: rfo" in text
        assert "do_hess: True" in text

    def test_optimizer_is_configurable(self, tmp_path, khp_reaction):
        calc = make_calc(khp_reaction, "reactant_pre_opt", PreOptConfig(opt_type="rfo"),
                         cls=PysisyphusMinOptCalculator)

        assert "type: rfo" in self._written(tmp_path, calc)


# =====================================================================
# Handoff from the pre-optimization into conformer generation
# =====================================================================
from yarp.reaction.conf_sampling.select_pairs import select_gsm_pairs
from yarp.reaction.external.conf_gen import CrestConfCalculator


def _conf(geo, elements, ctype):
    return SimpleNamespace(geo=geo, elements=elements, type=ctype,
                           to_xyz_string=lambda: "stub")


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


class TestGsmPoolExcludesNonConformers:
    """
    `select_gsm_pairs` used to take "every conformer except initial_geom".
    The states now also carry the pre-optimization's structure, which is not a
    CREST conformer -- feeding it to GSM would seed the string with the wrong
    geometry.
    """

    def _populate(self, rxn):
        for state in (rxn.reactant, rxn.product):
            graph = state.graph
            state.conformers["preopt_xtb_pysisyphus"] = _conf(
                graph.geo.copy(), graph.elements, "preopt_xtb_pysisyphus")
            state.conformers["rpopt_xtb_pysisyphus"] = _conf(
                graph.geo.copy(), graph.elements, "rpopt_xtb_pysisyphus")
            state.conformers["conf_gen_rank0_gfn2_crest"] = _conf(
                graph.geo.copy(), graph.elements, "conf_gen_rank0_gfn2_crest")

    def _stub_pipeline(self, monkeypatch, seen):
        """
        Neutralize everything downstream of the conformer selection, so the
        test observes which structures `select_gsm_pairs` actually picked up
        rather than re-implementing the filter and proving nothing.
        """
        import yarp.reaction.conf_sampling.select_pairs as sp

        def record(conf, target_bem, lot="uff"):
            seen.append(conf.type)
            return conf

        monkeypatch.setattr(sp, "joint_optimize", record)
        monkeypatch.setattr(sp, "return_indicator", lambda E, RG, PG: [[0.0]])
        monkeypatch.setattr(sp, "align_conformers", lambda conf, biased: biased)
        monkeypatch.setattr(
            sp.pickle, "load",
            lambda f: SimpleNamespace(predict_proba=lambda ind: [[0.0, 1.0]]),
        )

    def test_only_conf_gen_structures_reach_the_pairing(self, khp_reaction, monkeypatch):
        self._populate(khp_reaction)
        seen = []
        self._stub_pipeline(monkeypatch, seen)

        config = SimpleNamespace(bias_lot="uff", joint_opt="dual", n_conf=1, verbose=False)
        select_gsm_pairs(khp_reaction, config)

        # One reactant conformer and one product conformer, both from CREST.
        assert seen == ["conf_gen_rank0_gfn2_crest", "conf_gen_rank0_gfn2_crest"]
        assert "preopt_xtb_pysisyphus" not in seen
        assert "rpopt_xtb_pysisyphus" not in seen

    def test_the_old_exclusion_filter_would_have_swept_them_in(self, khp_reaction):
        """Documents the bug: 'everything but initial_geom' is not the same set."""
        self._populate(khp_reaction)

        old = [k for k in khp_reaction.reactant.conformers if k != "initial_geom"]
        new = [k for k in khp_reaction.reactant.conformers if "conf_gen" in k]

        assert set(old) - set(new) == {"preopt_xtb_pysisyphus", "rpopt_xtb_pysisyphus"}


# =====================================================================
# CREST MD timestep for free diatomics
# =====================================================================
from yarp.reaction.external.conf_gen import DIATOMIC_MD_TIMESTEP_FS


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
