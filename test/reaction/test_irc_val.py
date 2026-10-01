"""
Tests for the IRC validation calculators.

TS optimization names its results `{run}_tsopt_<lot>_<sw>` by TS-opt run index,
so a failed TS-opt run leaves a gap in those keys. IRC writes its own runs as
irc_run1..N. These tests pin that each IRC result is attributed to the TS
conformer that run was actually started from, gaps or not. Getting it wrong
silently pairs one TS geometry with another TS's barrier and outcome label (or,
on the ORCA path, crashes on a missing conformer).
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from yarp.reaction.conformer import conformer
from yarp.reaction.external.irc_val import PysisyphusIRCValCalculator, OrcaIRCValCalculator
from yarp.util.config import IRCValConfig
from yarp.util.constants import Constants
from yarp.yarpecule.graph.adjacency import table_generator

# Toy H + H2 -> H2 + H. Only connectivity matters to the IRC labeler.
ELEMENTS = ["H", "H", "H"]
R_GEO = np.array([[0.0, 0, 0], [0.74, 0, 0], [3.74, 0, 0]])
P_GEO = np.array([[0.0, 0, 0], [3.0, 0, 0], [3.74, 0, 0]])


ONE_IMAG = np.array([-500.0, 100.0, 200.0])
NO_IMAG = np.array([50.0, 100.0, 200.0])
TWO_IMAG = np.array([-500.0, -80.0, 200.0])


def make_rxn(lot, software, tsopt_runs, freqs=None):
    """Stub reaction holding TS-opt conformers for the given TS-opt run indices.
    Conformer k is tagged by x(atom 0) = 10*k and G(TS) = k kcal/mol. Each is a
    valid saddle (one imaginary freq) unless `freqs` gives it other ones."""
    def species(geo):
        conf = conformer()
        conf.elements, conf.geo = list(ELEMENTS), geo.copy()
        conf.properties["gibbs_free_energy_kcal_per_mol"] = 0.0
        return SimpleNamespace(graph=SimpleNamespace(adj_mat=table_generator(ELEMENTS, geo)),
                               conformers={f"rpopt_{lot}_{software}": conf})

    rxn = SimpleNamespace(reactant=species(R_GEO), product=species(P_GEO), ts_geom={},
                          outcome_label={}, barrier={}, reverse_barrier={}, dg_rxn={})
    for k in tsopt_runs:
        conf = conformer()
        conf.elements = list(ELEMENTS)
        conf.geo = np.array([[10.0 * k, 0, 0], [1.5, 0, 0], [2.5, 0, 0]])
        conf.vibrational_freqs = (freqs or {}).get(k, ONE_IMAG).copy()
        conf.properties["gibbs_free_energy_kcal_per_mol"] = float(k)
        conf.type = f"{k}_tsopt_{lot}_{software}"
        rxn.ts_geom[conf.type] = conf
    return rxn


def write_xyz(path, geo):
    lines = [str(len(ELEMENTS)), ""] + [f"{e} {x} {y} {z}" for e, (x, y, z) in zip(ELEMENTS, geo)]
    Path(path).write_text("\n".join(lines) + "\n")


def tag_of(geo):
    return int(round(geo[0][0] / 10.0))


def run_irc(cls, lot, software, tsopt_runs, write_outputs, tmp_path, freqs=None):
    """generate_input -> fabricated successful outputs -> scrape_data.
    Returns the reaction and, per IRC run, the TS-opt index it was fed."""
    rxn = make_rxn(lot, software, tsopt_runs, freqs)
    cfg = IRCValConfig(software=software, lot=lot, charge=0, multiplicity=1)
    calc = cls(SimpleNamespace(config=cfg), rxn, MagicMock(container="docker"))
    calc.set_scratch_dir(tmp_path)
    calc.generate_input()

    fed = {}
    for i in range(1, calc._get_num_runs() + 1):
        run_dir = tmp_path / f"irc_run{i}"
        fed[i] = tag_of(np.loadtxt(run_dir / f"ts_opt_{i}.xyz", skiprows=2, usecols=(1, 2, 3)))
        write_outputs(run_dir, i)

    assert calc.scrape_data()
    return rxn, fed


def pysis_outputs(run_dir, i):
    """Every run intended; forward barrier 10*i kJ/mol, so irc_run1 is the lowest."""
    write_xyz(run_dir / "forward_end_opt.xyz", P_GEO)
    write_xyz(run_dir / "backward_end_opt.xyz", R_GEO)
    (run_dir / f"irc_{i}.log").write_text(
        "Minimum energy of 0.0 at 'Left'.\n"
        f"    Left: 0.0 kJ mol\n    TS: {10.0 * i} kJ mol\n    Right: 5.0 kJ mol\n"
        "Wrote optimized end-geometries and TS to x\npysisyphus run took 1 s\n")


def orca_outputs(run_dir, i):
    """Every run intended; barrier is G(TS) - G(R) = TS-opt index of the conformer."""
    write_xyz(run_dir / f"irc_{i}_IRC_F.xyz", P_GEO)
    write_xyz(run_dir / f"irc_{i}_IRC_B.xyz", R_GEO)
    (run_dir / f"irc_{i}.out").write_text("THE IRC HAS CONVERGED\nORCA TERMINATED NORMALLY\n")


TSOPT_RUN_PATTERNS = [
    pytest.param([1, 2, 3], id="no_gap"),
    pytest.param([1, 3], id="middle_run_failed"),
    pytest.param([2, 3], id="first_run_failed"),
]


class TestIRCPairsResultWithItsTS:
    @pytest.mark.parametrize("tsopt_runs", TSOPT_RUN_PATTERNS)
    def test_pysisyphus(self, tmp_path, tsopt_runs):
        rxn, fed = run_irc(PysisyphusIRCValCalculator, "xtb", "pysisyphus",
                           tsopt_runs, pysis_outputs, tmp_path)

        # irc_run1 has the lowest barrier, so its TS and its barrier must win.
        validated = rxn.ts_geom["validated_ts_xtb_pysisyphus"]
        assert validated is not None
        assert tag_of(validated.geo) == fed[1]
        assert rxn.barrier["xtb_pysisyphus"] == pytest.approx(10.0 / Constants.kcal2kJ)

    @pytest.mark.parametrize("tsopt_runs", TSOPT_RUN_PATTERNS)
    def test_orca(self, tmp_path, tsopt_runs):
        lot = "PBE D3BJ def2-SVP"
        rxn, fed = run_irc(OrcaIRCValCalculator, lot, "orca", tsopt_runs, orca_outputs, tmp_path)

        # Barrier is G(TS) - G(R), so it identifies which conformer was used.
        validated = rxn.ts_geom[f"validated_ts_{lot}_orca"]
        assert tag_of(validated.geo) == fed[1]
        assert rxn.barrier[f"{lot}_orca"] == pytest.approx(float(fed[1]))

    @pytest.mark.parametrize("tsopt_runs", TSOPT_RUN_PATTERNS)
    def test_every_run_is_scraped_against_its_own_ts(self, tmp_path, mocker, tsopt_runs):
        # Checking only the winner misses a gap after irc_run1: the later run is
        # dropped silently and the winner is still right.
        rxn, fed = run_irc(PysisyphusIRCValCalculator, "xtb", "pysisyphus",
                           tsopt_runs, pysis_outputs, tmp_path)
        # Re-scrape with a spy on the selection step to see each run's pairing.
        spy = mocker.spy(PysisyphusIRCValCalculator, "_get_final_results")
        cfg = IRCValConfig(software="pysisyphus", lot="xtb", charge=0, multiplicity=1)
        calc = PysisyphusIRCValCalculator(SimpleNamespace(config=cfg), rxn, MagicMock())
        calc.set_scratch_dir(tmp_path)
        calc.scrape_data()

        irc_runs = spy.call_args.args[1]
        assert {i: tag_of(d["ts_geom"].geo) for i, d in irc_runs.items()} == fed


class TestIRCSkipsNonSaddles:
    """
    IRC validates only TS-opt results that are first-order saddle points. It
    used to need just one valid TS to start, then ran every TS-opt result --
    so a minimum or a higher-order saddle could be run and even become the
    validated TS.
    """

    def test_only_valid_saddles_are_run(self, tmp_path):
        rxn, fed = run_irc(PysisyphusIRCValCalculator, "xtb", "pysisyphus", [1, 2, 3],
                           pysis_outputs, tmp_path, freqs={1: NO_IMAG, 3: TWO_IMAG})

        assert list(fed.values()) == [2]

    def test_non_saddle_is_never_the_validated_ts(self, tmp_path):
        # TS-opt result 1 is a minimum. Run through IRC it would be irc_run1,
        # whose fabricated barrier is the lowest, so it would win.
        rxn, fed = run_irc(PysisyphusIRCValCalculator, "xtb", "pysisyphus", [1, 2],
                           pysis_outputs, tmp_path, freqs={1: NO_IMAG})

        assert tag_of(rxn.ts_geom["validated_ts_xtb_pysisyphus"].geo) == 2
