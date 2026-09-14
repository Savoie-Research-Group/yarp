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
from yarp.util.config import PreOptConfig, RPOptConfig, InitialGeomConfig, GeomSourceConfig


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

