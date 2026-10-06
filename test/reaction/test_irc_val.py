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

        # Each TS carries the barrier of the run started from it (10*i kJ/mol);
        # irc_run1's is the lowest, so it is the reaction's barrier.
        for i, k in fed.items():
            props = rxn.ts_geom[f"{k}_tsopt_xtb_pysisyphus"].properties
            assert props["forward_barrier_kcal_per_mol"] == pytest.approx(10.0 * i / Constants.kcal2kJ)
        assert rxn.barrier["xtb_pysisyphus"] == pytest.approx(10.0 / Constants.kcal2kJ)

    @pytest.mark.parametrize("tsopt_runs", TSOPT_RUN_PATTERNS)
    def test_orca(self, tmp_path, tsopt_runs):
        lot = "PBE D3BJ def2-SVP"
        rxn, fed = run_irc(OrcaIRCValCalculator, lot, "orca", tsopt_runs, orca_outputs, tmp_path)

        # Barrier is G(TS) - G(R) = k for conformer k, so it identifies which
        # conformer each result was computed from.
        for k in fed.values():
            props = rxn.ts_geom[f"{k}_tsopt_{lot}_orca"].properties
            assert props["forward_barrier_kcal_per_mol"] == pytest.approx(float(k))
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

    def test_non_saddle_gets_no_irc_result(self, tmp_path):
        # TS-opt result 1 is a minimum. Run through IRC it would be irc_run1,
        # whose fabricated barrier is the lowest, so it would win.
        rxn, fed = run_irc(PysisyphusIRCValCalculator, "xtb", "pysisyphus", [1, 2],
                           pysis_outputs, tmp_path, freqs={1: NO_IMAG})

        assert "irc_outcome" not in rxn.ts_geom["1_tsopt_xtb_pysisyphus"].properties
        assert rxn.ts_geom["2_tsopt_xtb_pysisyphus"].properties["irc_outcome"] == "intended"


# Endpoints for an IRC that reproduces neither side: forward has an H0-H2 bond,
# backward has no bonds at all.
U_FWD_GEO = np.array([[0.0, 0, 0], [3.0, 0, 0], [0.74, 0.0, 0]])
U_BWD_GEO = np.array([[0.0, 0, 0], [3.0, 0, 0], [6.0, 0, 0]])


def pysis_outputs_by_label(labels):
    """
    Like pysis_outputs, with irc_run{i} ending as labels[i]: 'intended',
    'inverse_intended' (IRC sides swapped) or 'unintended'. The log always says
    Left 0, TS 10*i, Right 5 kJ/mol, so the R->P forward barrier is 10*i for an
    intended run and 10*i - 5 for an inverse one.
    """
    def write(run_dir, i):
        fwd, bwd = {"intended": (P_GEO, R_GEO), "inverse_intended": (R_GEO, P_GEO),
                    "unintended": (U_FWD_GEO, U_BWD_GEO)}[labels[i]]
        write_xyz(run_dir / "forward_end_opt.xyz", fwd)
        write_xyz(run_dir / "backward_end_opt.xyz", bwd)
        (run_dir / f"irc_{i}.log").write_text(
            "Minimum energy of 0.0 at 'Left'.\n"
            f"    Left: 0.0 kJ mol\n    TS: {10.0 * i} kJ mol\n    Right: 5.0 kJ mol\n"
            "Wrote optimized end-geometries and TS to x\npysisyphus run took 1 s\n")
    return write


def kcal(kj):
    return pytest.approx(kj / Constants.kcal2kJ)


class TestPerTSResults:
    """
    Every IRC run is recorded on the TS conformer it started from. The
    reaction-level entries summarize the lowest intended TS, and are None when
    no TS is intended. No separate validated_ts copy is written.
    """

    def _run(self, tmp_path, labels):
        return run_irc(PysisyphusIRCValCalculator, "xtb", "pysisyphus", list(labels),
                       pysis_outputs_by_label(labels), tmp_path)

    def test_every_ts_gets_its_label_and_oriented_barriers(self, tmp_path):
        rxn, _ = self._run(tmp_path, {1: "unintended", 2: "inverse_intended", 3: "intended"})
        props = {k: rxn.ts_geom[f"{k}_tsopt_xtb_pysisyphus"].properties for k in (1, 2, 3)}

        assert {k: p["irc_outcome"] for k, p in props.items()} == \
            {1: "unintended", 2: "inverse_intended", 3: "intended"}
        # Intended: forward = TS - Left, reverse = TS - Right.
        assert props[3]["forward_barrier_kcal_per_mol"] == kcal(30.0)
        assert props[3]["reverse_barrier_kcal_per_mol"] == kcal(25.0)
        # Inverse: the IRC ran P -> R, so forward (R -> P) = TS - Right.
        assert props[2]["forward_barrier_kcal_per_mol"] == kcal(15.0)
        assert props[2]["reverse_barrier_kcal_per_mol"] == kcal(20.0)
        # Unintended runs keep their barriers too.
        assert props[1]["forward_barrier_kcal_per_mol"] == kcal(10.0)

    def test_summary_is_lowest_intended_including_inverse(self, tmp_path):
        # The unintended TS has the lowest barrier of all; it must not win.
        rxn, _ = self._run(tmp_path, {1: "unintended", 2: "inverse_intended", 3: "intended"})

        assert rxn.outcome_label["xtb_pysisyphus"] == "inverse_intended"
        assert rxn.barrier["xtb_pysisyphus"] == kcal(15.0)
        assert rxn.reverse_barrier["xtb_pysisyphus"] == kcal(20.0)
        assert rxn.dg_rxn["xtb_pysisyphus"] == kcal(5.0)

    def test_summary_is_none_without_an_intended_ts(self, tmp_path):
        rxn, _ = self._run(tmp_path, {1: "unintended", 2: "unintended"})

        assert rxn.outcome_label["xtb_pysisyphus"] == "unintended"
        assert rxn.barrier["xtb_pysisyphus"] is None
        assert rxn.reverse_barrier["xtb_pysisyphus"] is None
        assert rxn.dg_rxn["xtb_pysisyphus"] is None

    def test_no_validated_ts_copy(self, tmp_path):
        rxn, _ = self._run(tmp_path, {1: "intended", 2: "unintended"})

        assert not any("validated_ts" in k for k in rxn.ts_geom)
