"""
Tests for GSM conformer-pair selection.

`select_gsm_pairs` builds the pool GSM is seeded from. These tests pin which
structures may enter that pool (CREST conformers only) and that conformers which
have collapsed onto the other side of the reaction are dropped before pairing.
"""
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from yarp.reaction.conf_sampling.joint_opt import joint_optimize
from yarp.reaction.conf_sampling.select_pairs import select_gsm_pairs
from yarp.reaction.external.calc_base import CalculatorInputError
from yarp.reaction.external.ts_guess import PysisyphusTSGuessCalculator
from yarp.reaction.reaction import reaction
from yarp.yarpecule.graph.adjacency import compare_adjacency


def _conf(geo, elements, ctype):
    return SimpleNamespace(geo=geo, elements=elements, type=ctype,
                           to_xyz_string=lambda: "stub")



@pytest.fixture
def patchable_reaction(khp_parent, khp_products):
    """
    KHP -> CCC(=O)OO, for tests that need a real on-graph product geometry.

    Not `khp_reaction`: its product, C=COCOO, is the one KHP product whose UFF
    patch from the raw reactant coordinates fails. This one patches with RDKit
    alone, which is deterministic.
    """
    return reaction(khp_parent, khp_products["CCC(=O)OO"])


def _on_graph_geometry(rxn, side):
    """
    A geometry that perceives to exactly `side`'s own graph.

    The reactant's graph geometry already does. An enumerated product's does
    not -- it still carries the reactant's coordinates -- so the product's is
    made by patching the reactant onto the product BEM.
    """
    state = rxn.reactant if side == "reactant" else rxn.product
    if side == "reactant":
        geo = rxn.reactant.graph.geo.copy()
    else:
        patched = joint_optimize(rxn.reactant.conformers["initial_geom"], rxn.product.graph.bond_mats[0])
        assert patched is not None, "fixture product must patch from the raw reactant geometry"
        geo = patched.geo
    assert compare_adjacency(state.graph.elements, geo, state.graph.adj_mat)[0]
    return geo


def _stub_gsm_pairing(monkeypatch, seen):
    """
    Neutralize everything downstream of the conformer selection, so a test
    observes which structures `select_gsm_pairs` actually picked up rather
    than re-implementing the filter and proving nothing.
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


class TestGsmPoolExcludesNonConformers:
    """
    `select_gsm_pairs` used to take "every conformer except initial_geom".
    The states now also carry the pre-optimization's structure, which is not a
    CREST conformer -- feeding it to GSM would seed the string with the wrong
    geometry.
    """

    def _populate(self, rxn):
        for side, state in (("reactant", rxn.reactant), ("product", rxn.product)):
            graph = state.graph
            state.conformers["preopt_xtb_pysisyphus"] = _conf(
                graph.geo.copy(), graph.elements, "preopt_xtb_pysisyphus")
            state.conformers["rpopt_xtb_pysisyphus"] = _conf(
                graph.geo.copy(), graph.elements, "rpopt_xtb_pysisyphus")
            # On its own graph, or the collapse check would drop it first.
            state.conformers["conf_gen_rank0_gfn2_crest"] = _conf(
                _on_graph_geometry(rxn, side), graph.elements, "conf_gen_rank0_gfn2_crest")

    def test_only_conf_gen_structures_reach_the_pairing(self, patchable_reaction, monkeypatch):
        self._populate(patchable_reaction)
        seen = []
        _stub_gsm_pairing(monkeypatch, seen)

        config = SimpleNamespace(bias_lot="uff", joint_opt="dual", n_conf=1, verbose=False)
        select_gsm_pairs(patchable_reaction, config)

        # One reactant conformer and one product conformer, both from CREST.
        assert seen == ["conf_gen_rank0_gfn2_crest", "conf_gen_rank0_gfn2_crest"]
        assert "preopt_xtb_pysisyphus" not in seen
        assert "rpopt_xtb_pysisyphus" not in seen

    def test_the_old_exclusion_filter_would_have_swept_them_in(self, patchable_reaction):
        """Documents the bug: 'everything but initial_geom' is not the same set."""
        self._populate(patchable_reaction)

        old = [k for k in patchable_reaction.reactant.conformers if k != "initial_geom"]
        new = [k for k in patchable_reaction.reactant.conformers if "conf_gen" in k]

        assert set(old) - set(new) == {"preopt_xtb_pysisyphus", "rpopt_xtb_pysisyphus"}


class TestCollapsedConformersAreDropped:
    """
    A conformer whose bonds perceive to exactly the OTHER side's graph has
    collapsed, and pairing it gives GSM nothing to do. `select_gsm_pairs` drops
    those before biasing, and discards the reaction if a side has none left.

    A collapsed conformer here is a real geometry of the other side, index
    aligned with this one, rather than a hand-edited adjacency matrix.
    """

    CONFIG = SimpleNamespace(bias_lot="uff", joint_opt="dual", n_conf=5, verbose=False)

    def _set(self, state, geos):
        """Replace a state's CREST conformers with one per {label: geometry}."""
        for key in [k for k in state.conformers if "conf_gen" in k]:
            del state.conformers[key]
        for rank, (label, geo) in enumerate(geos.items()):
            state.conformers[f"conf_gen_rank{rank}_gfn2_crest"] = _conf(geo, state.graph.elements, label)

    def test_a_product_conformer_on_the_reactant_graph_is_dropped(self, patchable_reaction, monkeypatch, capsys):
        rxn = patchable_reaction
        reactant_geo = _on_graph_geometry(rxn, "reactant")
        self._set(rxn.reactant, {"R": reactant_geo})
        self._set(rxn.product, {"P": _on_graph_geometry(rxn, "product"),
                                "P collapsed": reactant_geo.copy()})
        seen = []
        _stub_gsm_pairing(monkeypatch, seen)

        select_gsm_pairs(rxn, self.CONFIG)

        assert seen == ["R", "P"]
        assert "0 reactant, 1 product" in capsys.readouterr().out

    def test_a_reactant_conformer_on_the_product_graph_is_dropped(self, patchable_reaction, monkeypatch, capsys):
        rxn = patchable_reaction
        product_geo = _on_graph_geometry(rxn, "product")
        self._set(rxn.reactant, {"R": _on_graph_geometry(rxn, "reactant"),
                                 "R collapsed": product_geo.copy()})
        self._set(rxn.product, {"P": product_geo})
        seen = []
        _stub_gsm_pairing(monkeypatch, seen)

        select_gsm_pairs(rxn, self.CONFIG)

        assert seen == ["R", "P"]
        assert "1 reactant, 0 product" in capsys.readouterr().out

    @pytest.mark.parametrize("side", ["reactant", "product"])
    def test_a_side_with_nothing_left_discards_the_reaction(self, patchable_reaction, monkeypatch, side):
        rxn = patchable_reaction
        on_graph = {"reactant": _on_graph_geometry(rxn, "reactant"),
                    "product": _on_graph_geometry(rxn, "product")}
        other = {"reactant": "product", "product": "reactant"}
        for s in ("reactant", "product"):
            geo = on_graph[other[s]] if s == side else on_graph[s]
            self._set(getattr(rxn, s), {s: geo.copy()})
        seen = []
        _stub_gsm_pairing(monkeypatch, seen)

        with pytest.raises(CalculatorInputError, match=f"No usable {side} conformers"):
            select_gsm_pairs(rxn, self.CONFIG)
        assert seen == []

    def test_other_graph_changes_are_kept(self, patchable_reaction, monkeypatch):
        """
        Only a collapse onto the other side is dropped. The product pre-opt
        deliberately keeps a product whose graph changed in some other way, and
        this check must not undo that.
        """
        rxn = patchable_reaction
        changed = _on_graph_geometry(rxn, "product")
        changed[0] += 10.0  # pull one atom out of bonding range
        for adj in (rxn.reactant.graph.adj_mat, rxn.product.graph.adj_mat):
            assert not compare_adjacency(rxn.product.graph.elements, changed, adj)[0]

        self._set(rxn.reactant, {"R": _on_graph_geometry(rxn, "reactant")})
        self._set(rxn.product, {"P changed": changed})
        seen = []
        _stub_gsm_pairing(monkeypatch, seen)

        select_gsm_pairs(rxn, self.CONFIG)

        assert seen == ["R", "P changed"]

    def test_the_error_reaches_pass_3_through_generate_input(self, patchable_reaction, monkeypatch):
        """
        PASS 3.2 discards a reaction only for a CalculatorInputError raised by
        generate_input(); anything else would take down the whole pass.
        """
        rxn = patchable_reaction
        reactant_geo = _on_graph_geometry(rxn, "reactant")
        self._set(rxn.reactant, {"R": reactant_geo})
        self._set(rxn.product, {"P collapsed": reactant_geo.copy()})
        _stub_gsm_pairing(monkeypatch, [])

        task_def = SimpleNamespace(task_type="ts_guess", config=self.CONFIG, task_id="s.ts_guess")
        calc = PysisyphusTSGuessCalculator(task_def, rxn, MagicMock())

        with pytest.raises(CalculatorInputError, match="No usable product conformers"):
            calc.generate_input()

