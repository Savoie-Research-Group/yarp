"""
Tests for where a TS optimization takes its starting structures from.

A refinement stage whose `initial_geom.transition_state` is `label: ts_opt`
used to start from the single TS that the source level's IRC validated. It now
starts from every converged TS optimization at the source level that is a valid
saddle point: low-level IRC labels are poor predictors of high-level IRC
outcomes, so dropping the other candidates loses TSs the high level could have
found. `label: ts_guess` is unchanged.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from yarp.reaction.conformer import conformer
from yarp.reaction.external.ts_opt import OrcaTSOptCalculator, PysisyphusTSOptCalculator
from yarp.util.config import TSOptConfig, InitialGeomConfig, GeomSourceConfig

ELEMENTS = ["H", "H", "H"]
HL_LOT = "PBE D3BJ def2-SVP"
ONE_IMAG = np.array([-500.0, 100.0, 200.0])
NO_IMAG = np.array([50.0, 100.0, 200.0])
TWO_IMAG = np.array([-500.0, -80.0, 200.0])


def ts_conf(tag, key, freqs):
    """A TS conformer identified by x(atom 0) = 10*tag."""
    conf = conformer()
    conf.elements = list(ELEMENTS)
    conf.geo = np.array([[10.0 * tag, 0, 0], [1.5, 0, 0], [2.5, 0, 0]])
    conf.vibrational_freqs = None if freqs is None else freqs.copy()
    conf.type = key
    return conf


def make_calc(cls, ts_geom, ts_source, lot=HL_LOT, software="orca"):
    src = GeomSourceConfig(label="rp_opt", lot="xtb", software="pysisyphus")
    cfg = TSOptConfig(software=software, lot=lot, charge=0, multiplicity=1)
    cfg.initial_geom = InitialGeomConfig(reactant=src, product=src, transition_state=ts_source)
    return cls(SimpleNamespace(config=cfg), SimpleNamespace(ts_geom=ts_geom), MagicMock(container="docker"))


def written_tags(calc, tmp_path, prefix="ts_guess"):
    """generate_input, then read back which conformer each tsopt_run{i} got."""
    calc.set_scratch_dir(tmp_path)
    calc.generate_input()
    tags = []
    for i in range(1, calc._get_num_runs() + 1):
        geo = np.loadtxt(tmp_path / f"tsopt_run{i}" / f"{prefix}_{i}.xyz", skiprows=2, usecols=(1, 2, 3))
        tags.append(int(round(geo[0][0] / 10.0)))
    return tags


def xtb_refined_rxn():
    """
    ts_geom as it stands after an xTB refine stage: TS-opt run 2 failed (gap),
    run 4 converged to a minimum, run 5 to a second-order saddle; IRC validated
    run 3. Plus GSM guesses and another level's TS opt that must not be picked up.
    """
    confs = [
        ts_conf(91, "ts_guess_1_xtb_pysisyphus", None),
        ts_conf(92, "ts_guess_2_xtb_pysisyphus", None),
        ts_conf(1, "1_tsopt_xtb_pysisyphus", ONE_IMAG),
        ts_conf(3, "3_tsopt_xtb_pysisyphus", ONE_IMAG),
        ts_conf(4, "4_tsopt_xtb_pysisyphus", NO_IMAG),
        ts_conf(5, "5_tsopt_xtb_pysisyphus", TWO_IMAG),
        ts_conf(6, "6_tsopt_xtb_pysisyphus", ONE_IMAG),
        ts_conf(70, "1_tsopt_B3LYP def2-SVP_orca", ONE_IMAG),
    ]
    ts_geom = {c.type: c for c in confs}
    ts_geom["validated_ts_xtb_pysisyphus"] = ts_geom["3_tsopt_xtb_pysisyphus"]  # IRC stores the same object
    return ts_geom


XTB_TS_OPT = GeomSourceConfig(label="ts_opt", lot="xtb", software="pysisyphus")
XTB_TS_GUESS = GeomSourceConfig(label="ts_guess", lot="xtb", software="pysisyphus")


class TestTSOptSource:
    def test_starts_from_every_valid_source_ts(self, tmp_path):
        calc = make_calc(OrcaTSOptCalculator, xtb_refined_rxn(), XTB_TS_OPT)

        # 1, 3 and 6 are first-order saddles at xtb/pysisyphus, in insertion
        # order; 4 and 5 aren't saddles; 70 is another level; 91/92 are guesses.
        assert written_tags(calc, tmp_path) == [1, 3, 6]

    def test_not_only_the_validated_ts(self, tmp_path):
        calc = make_calc(OrcaTSOptCalculator, xtb_refined_rxn(), XTB_TS_OPT)

        assert len(written_tags(calc, tmp_path)) > 1

    def test_ready_when_one_valid_ts_exists(self):
        ts_geom = {"2_tsopt_xtb_pysisyphus": ts_conf(2, "2_tsopt_xtb_pysisyphus", ONE_IMAG)}

        assert make_calc(OrcaTSOptCalculator, ts_geom, XTB_TS_OPT).has_prerequisites()

    def test_not_ready_without_a_valid_ts(self):
        ts_geom = {
            "1_tsopt_xtb_pysisyphus": ts_conf(1, "1_tsopt_xtb_pysisyphus", NO_IMAG),
            "2_tsopt_xtb_pysisyphus": ts_conf(2, "2_tsopt_xtb_pysisyphus", TWO_IMAG),
            "1_tsopt_B3LYP def2-SVP_orca": ts_conf(70, "1_tsopt_B3LYP def2-SVP_orca", ONE_IMAG),
        }
        # IRC should never be performed on a TS that is not a 1st order saddle point
        ts_geom["validated_ts_xtb_pysisyphus"] = ts_geom["1_tsopt_xtb_pysisyphus"]

        assert not make_calc(OrcaTSOptCalculator, ts_geom, XTB_TS_OPT).has_prerequisites()


class TestTSGuessSourceUnchanged:
    def test_starts_from_every_guess(self, tmp_path):
        calc = make_calc(PysisyphusTSOptCalculator, xtb_refined_rxn(), XTB_TS_GUESS,
                         lot="xtb", software="pysisyphus")

        assert written_tags(calc, tmp_path) == [91, 92]

    def test_ready_with_a_guess_geometry(self):
        ts_geom = {"ts_guess_1_xtb_pysisyphus": ts_conf(91, "ts_guess_1_xtb_pysisyphus", None)}

        assert make_calc(PysisyphusTSOptCalculator, ts_geom, XTB_TS_GUESS,
                         lot="xtb", software="pysisyphus").has_prerequisites()

    def test_not_ready_when_guess_has_no_geometry(self):
        guess = ts_conf(91, "ts_guess_1_xtb_pysisyphus", None)
        guess.geo = None

        assert not make_calc(PysisyphusTSOptCalculator, {guess.type: guess}, XTB_TS_GUESS,
                             lot="xtb", software="pysisyphus").has_prerequisites()
