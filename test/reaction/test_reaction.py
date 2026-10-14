"""
Tests for the reaction class (yarp/reaction/reaction.py).

Per-TS IRC results: IRC records each TS's outcome and R->P barriers on the TS conformer itself.
`reaction.ts_results` lists the TSs at one level that have a result, and
`reaction.min_barrier` gives the lowest barrier among intended TSs by default.
"""
import pytest

from yarp.reaction.conformer import conformer


def ts(rxn, key, outcome, forward):
    conf = conformer()
    if outcome is not None:
        conf.properties["irc_outcome"] = outcome
        conf.properties["forward_barrier_kcal_per_mol"] = forward
        conf.properties["reverse_barrier_kcal_per_mol"] = forward + 1.0
    rxn.ts_geom[key] = conf
    return conf


@pytest.fixture
def rxn(khp_reaction):
    ts(khp_reaction, "ts_guess_1_xtb_pysisyphus", None, None)
    ts(khp_reaction, "1_tsopt_xtb_pysisyphus", "unintended", 10.0)
    ts(khp_reaction, "2_tsopt_xtb_pysisyphus", "inverse_intended", 30.0)
    ts(khp_reaction, "3_tsopt_xtb_pysisyphus", "intended", 40.0)
    ts(khp_reaction, "4_tsopt_xtb_pysisyphus", None, None)            # no IRC result (non-saddle)
    ts(khp_reaction, "1_tsopt_PBE D3BJ def2-SVP_orca", "intended", 5.0)  # another level
    return khp_reaction


class TestTSResults:
    def test_only_this_level_with_an_irc_result(self, rxn):
        assert list(rxn.ts_results("xtb_pysisyphus")) == [
            "1_tsopt_xtb_pysisyphus", "2_tsopt_xtb_pysisyphus", "3_tsopt_xtb_pysisyphus",
        ]

    def test_unknown_level_is_empty(self, rxn):
        assert rxn.ts_results("B3LYP_orca") == {}


class TestMinBarrier:
    def test_default_is_lowest_intended_including_inverse(self, rxn):
        # The unintended TS (10) is lower but doesn't count; the other level's (5) neither.
        assert rxn.min_barrier("xtb_pysisyphus") == 30.0

    def test_other_labels_on_request(self, rxn):
        assert rxn.min_barrier("xtb_pysisyphus", labels=("unintended",)) == 10.0

    def test_none_without_an_intended_ts(self, khp_reaction):
        ts(khp_reaction, "1_tsopt_xtb_pysisyphus", "unintended", 10.0)

        assert khp_reaction.min_barrier("xtb_pysisyphus") is None

    def test_none_without_any_result(self, khp_reaction):
        assert khp_reaction.min_barrier("xtb_pysisyphus") is None
