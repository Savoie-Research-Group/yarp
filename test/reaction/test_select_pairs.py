"""
Tests for GSM conformer-pair selection.

`select_gsm_pairs` builds the pool GSM is seeded from. These tests pin which
structures may enter that pool (CREST conformers only), that conformers which
have collapsed onto the other side of the reaction are dropped before pairing,
and that geometries beyond the paper's hard cutoffs are discarded before the RF
model scores them.
"""
import pandas as pd
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from yarp.reaction.conf_sampling.joint_opt import joint_optimize
from yarp.reaction.conf_sampling.select_pairs import (
    select_gsm_pairs, passes_hard_cutoffs, HARD_CUTOFFS,
)
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


FEATURES = ["RMSD", "max_dis", "max_Hdis", "min_cross_dis", "min_Hcross_dis", "path_cross"]


def _indicators(**values):
    """A one-row indicator frame, as return_indicator builds it, within every hard cutoff by default."""
    row = {"RMSD": 1.0, "max_dis": 2.0, "max_Hdis": 2.0, "min_cross_dis": 1.0,
           "min_Hcross_dis": 1.0, "path_cross": 1.0}
    row.update(values)
    return pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)


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
    monkeypatch.setattr(sp, "return_indicator", lambda E, RG, PG: _indicators())
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

    CONFIG = SimpleNamespace(bias_lot="uff", joint_opt="dual", n_conf=5, verbose=False,
                             gsm_lot="xtb")  # the GSM calculator picks its image from gsm_lot

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


class TestHardCutoffs:
    """
    Zhao, Hsu & Savoie (JCTC 2022, Table 1) discard reaction geometries whose
    indicators exceed RMSD 1.8, Dis_max 4.8, Cross_min 0.03 or Path_min 0.20 A
    before applying the RF model. Each sampled conformer has two candidate
    geometries, unaligned and aligned; each is checked on its own, the
    tournament runs between those that pass, and a conformer whose two
    geometries both fail is dropped (ERM, 2026-11-17). If every conformer is
    dropped, the reaction is discarded.
    """

    CONFIG = SimpleNamespace(bias_lot="uff", joint_opt="dual", n_conf=5, verbose=False)

    def test_thresholds_are_table_1(self):
        assert HARD_CUTOFFS == (("RMSD", "max", 1.8), ("max_dis", "max", 4.8),
                                ("min_cross_dis", "min", 0.03), ("path_cross", "min", 0.20))

    @pytest.mark.parametrize("name,at,beyond", [("RMSD", 1.8, 1.81), ("max_dis", 4.8, 4.81),
                                                ("min_cross_dis", 0.03, 0.029), ("path_cross", 0.20, 0.19)])
    def test_a_value_at_the_threshold_passes_and_beyond_it_fails(self, name, at, beyond):
        assert passes_hard_cutoffs(_indicators(**{name: at}))
        assert not passes_hard_cutoffs(_indicators(**{name: beyond}))

    def test_the_hydrogen_indicators_have_no_cutoff(self):
        assert passes_hard_cutoffs(_indicators(max_Hdis=50.0, min_Hcross_dis=0.0))

    def _run(self, rxn, monkeypatch, verdicts):
        """
        One reactant and one product conformer, each on its own graph. The
        stubs score each geometry with the p in `verdicts[(side, aligned)]` and
        give it indicators that pass or fail the cutoffs as that entry says:
        verdicts maps (side, aligned) -> (p, passes). Returns the selected pairs.
        """
        import yarp.reaction.conf_sampling.select_pairs as sp

        r_geo, p_geo = _on_graph_geometry(rxn, "reactant"), _on_graph_geometry(rxn, "product")
        for state, geo in ((rxn.reactant, r_geo), (rxn.product, p_geo)):
            for key in [k for k in state.conformers if "conf_gen" in k]:
                del state.conformers[key]
            state.conformers["conf_gen_rank0_gfn2_crest"] = _conf(geo, state.graph.elements, "sampled")

        def indicators(E, RG, PG):
            side = "R" if RG is r_geo else "P"
            p, ok = verdicts[(side, isinstance(PG, str))]
            # p travels in max_Hdis, which has no cutoff, so the model stub can read it back
            return _indicators(max_Hdis=p, **({} if ok else {"RMSD": 2.5}))

        monkeypatch.setattr(sp, "joint_optimize", lambda conf, bem, lot="uff": _conf(conf.geo, conf.elements, "biased"))
        monkeypatch.setattr(sp, "align_conformers", lambda conf, biased: _conf("ALIGNED", conf.elements, "aligned"))
        monkeypatch.setattr(sp, "return_indicator", indicators)
        monkeypatch.setattr(sp.pickle, "load", lambda f: SimpleNamespace(
            predict_proba=lambda ind: [[1 - ind["max_Hdis"].iloc[0], ind["max_Hdis"].iloc[0]]]))
        return select_gsm_pairs(rxn, self.CONFIG)

    def test_a_geometry_beyond_the_cutoffs_cannot_win_the_tournament(self, patchable_reaction, monkeypatch):
        """
        The unaligned geometry scores higher (0.9) but fails the cutoffs; the
        aligned one (0.3) passes. The pair keeps the aligned geometry, at 0.3.
        """
        pairs = self._run(patchable_reaction, monkeypatch, {
            ("R", False): (0.9, False), ("R", True): (0.3, True),
            ("P", False): (0.2, True), ("P", True): (0.1, True)})

        r_side = [p for p in pairs if p["r_conf"].type == "sampled"]
        assert len(r_side) == 1
        assert r_side[0]["p_conf"].type == "aligned"
        assert r_side[0]["score"] == pytest.approx(0.3)

    def test_a_conformer_with_both_geometries_beyond_the_cutoffs_is_dropped(self, patchable_reaction, monkeypatch):
        pairs = self._run(patchable_reaction, monkeypatch, {
            ("R", False): (0.9, False), ("R", True): (0.8, False),
            ("P", False): (0.2, True), ("P", True): (0.1, True)})

        assert [p["p_conf"].type for p in pairs] == ["sampled"]   # only the product conformer's pair
        assert pairs[0]["score"] == pytest.approx(0.2)

    def test_every_conformer_beyond_the_cutoffs_discards_the_reaction(self, patchable_reaction, monkeypatch):
        with pytest.raises(CalculatorInputError, match="hard cutoffs"):
            self._run(patchable_reaction, monkeypatch, {
                ("R", False): (0.9, False), ("R", True): (0.8, False),
                ("P", False): (0.2, False), ("P", True): (0.1, False)})
