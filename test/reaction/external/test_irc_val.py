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
from yarp.util.config import IRCValConfig, JobManagerConfig
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


def run_irc(cls, lot, software, tsopt_runs, write_outputs, tmp_path, freqs=None, setup=None):
    """generate_input -> fabricated successful outputs -> scrape_data.
    Returns the reaction and, per IRC run, the TS-opt index it was fed.
    `setup(rxn)`, if given, edits the stub reaction first."""
    rxn = make_rxn(lot, software, tsopt_runs, freqs)
    if setup is not None:
        setup(rxn)
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


def pysis_irc_log(i, forward="rms(grad) converged!", backward="Integrator indicated convergence!",
                  end_opts=("Converged!", "Converged!")):
    """
    The lines of a pysisyphus IRC log that YARP reads, in order: how each IRC
    direction stopped, how each end optimization ended, the barriers (Left 0,
    TS 10*i, Right 5 kJ/mol) and the closing lines. Pysisyphus writes the end
    geometries and these closing lines whether or not anything converged.
    """
    fwd_end, bwd_end = end_opts
    return (f"        {forward}\n"
            f"        {backward}\n"
            f"        {fwd_end}\n"
            f"        {bwd_end}\n"
            "Minimum energy of 0.0 at 'Left'.\n"
            f"    Left: 0.0 kJ mol\n    TS: {10.0 * i} kJ mol\n    Right: 5.0 kJ mol\n"
            "Wrote optimized end-geometries and TS to x\npysisyphus run took 1 s\n")


def pysis_outputs(run_dir, i):
    """Every run intended; forward barrier 10*i kJ/mol, so irc_run1 is the lowest."""
    write_xyz(run_dir / "forward_end_opt.xyz", P_GEO)
    write_xyz(run_dir / "backward_end_opt.xyz", R_GEO)
    (run_dir / f"irc_{i}.log").write_text(pysis_irc_log(i))


# ORCA prints the first line once per IRC direction; a direction stopped at
# MaxIter prints the second instead, and the run still terminates normally.
ORCA_IRC_CONVERGED = "THE IRC HAS CONVERGED\n"
ORCA_IRC_MAXITER = "MAXIMUM NUMBER OF ITERATIONS REACHED - STOPPING IRC RUN\n"
ORCA_IRC_LOG = 2 * ORCA_IRC_CONVERGED + "ORCA TERMINATED NORMALLY\n"


def orca_outputs(run_dir, i):
    """Every run intended; barrier is G(TS) - G(R) = TS-opt index of the conformer."""
    write_xyz(run_dir / f"irc_{i}_IRC_F.xyz", P_GEO)
    write_xyz(run_dir / f"irc_{i}_IRC_B.xyz", R_GEO)
    (run_dir / f"irc_{i}.out").write_text(ORCA_IRC_LOG)


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


# A soft saddle (ts_handoff check 18: KHP 16610336 TS2, -35.5 cm-1), and a
# stiff TS with a small extra imaginary mode (KHP 17011678 TS3: -628, -17.5).
SOFT_IMAG = np.array([-35.0, 100.0, 200.0])
STIFF_PLUS_NOISE = np.array([-628.0, -17.5, 200.0])


class TestMinImagFreq:
    """
    irc_val.min_imag_freq (cm-1) sets which TSs an IRC validates: exactly one
    mode below -min_imag_freq. Smaller imaginary modes are ignored as noise.
    The default, 0.01, is the old rule.
    """

    def _calc(self, freqs, min_imag_freq=None):
        extra = {} if min_imag_freq is None else {"min_imag_freq": min_imag_freq}
        cfg = IRCValConfig(software="pysisyphus", lot="xtb", charge=0, multiplicity=1, **extra)
        rxn = make_rxn("xtb", "pysisyphus", list(freqs), freqs)
        return PysisyphusIRCValCalculator(SimpleNamespace(config=cfg), rxn, MagicMock())

    FREQS = {1: ONE_IMAG, 2: SOFT_IMAG, 3: STIFF_PLUS_NOISE}

    def test_default_keeps_the_old_rule(self):
        assert [tag_of(c.geo) for c in self._calc(self.FREQS)._initial_guesses()] == [1, 2]

    def test_cutoff_drops_the_soft_ts_and_ignores_a_small_extra_mode(self):
        assert [tag_of(c.geo) for c in self._calc(self.FREQS, 50.0)._initial_guesses()] == [1, 3]

    def test_not_ready_when_every_ts_is_soft(self):
        assert self._calc({1: SOFT_IMAG}).has_prerequisites()
        assert not self._calc({1: SOFT_IMAG}, 50.0).has_prerequisites()


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
        (run_dir / f"irc_{i}.log").write_text(pysis_irc_log(i))
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


class TestIRCSkipsDuplicates:
    """
    TSs that are the same structure (conformer.is_duplicate_of) get one IRC,
    run from the lowest-energy copy; the others are marked duplicate_of and get
    no IRC result. Each run's result still lands on the TS it started from.
    """

    @staticmethod
    def _duplicates(rxn):
        # TS 3 is TS 1 again (same geometry, tag 1), 1e-6 Eh lower; TS 2 is distinct.
        ts1, ts2, ts3 = (rxn.ts_geom[f"{k}_tsopt_xtb_pysisyphus"] for k in (1, 2, 3))
        ts3.geo = ts1.geo.copy()
        for conf, energy in ((ts1, -1.0), (ts2, -2.0), (ts3, -1.0 - 1e-6)):
            conf.properties["internal_energy_Eh"] = energy

    def test_one_irc_per_structure_from_the_lowest_copy(self, tmp_path):
        rxn, fed = run_irc(PysisyphusIRCValCalculator, "xtb", "pysisyphus", [1, 2, 3],
                           pysis_outputs, tmp_path, setup=self._duplicates)
        ts1, ts2, ts3 = (rxn.ts_geom[f"{k}_tsopt_xtb_pysisyphus"].properties for k in (1, 2, 3))

        assert fed == {1: 2, 2: 1}
        assert ts1["duplicate_of"] == "3_tsopt_xtb_pysisyphus"
        assert "irc_outcome" not in ts1
        # irc_run{i} has a forward barrier of 10*i kJ/mol.
        assert ts2["forward_barrier_kcal_per_mol"] == kcal(10.0)
        assert ts3["forward_barrier_kcal_per_mol"] == kcal(20.0)


class TestPysisRunConvergence:
    """
    Pysisyphus writes both end geometries even when an IRC direction or an end
    optimization stopped unconverged, and those runs used to be labeled and
    given barriers. In the ts_handoff step 0.3 runs, 12 of 37 end opts hit
    max_cycles; one gave a -268 kcal/mol barrier. Now a run counts only if both
    directions stop on a converged message and both end opts converge.
    """

    def _run(self, tmp_path, **log_kwargs):
        """irc_run1 is the run under test; irc_run2 is a converged control."""
        def write(run_dir, i):
            write_xyz(run_dir / "forward_end_opt.xyz", P_GEO)
            write_xyz(run_dir / "backward_end_opt.xyz", R_GEO)
            (run_dir / f"irc_{i}.log").write_text(pysis_irc_log(i, **(log_kwargs if i == 1 else {})))
        return run_irc(PysisyphusIRCValCalculator, "xtb", "pysisyphus", [1, 2], write, tmp_path)

    @pytest.mark.parametrize("forward, backward", [
        ("rms(grad) converged!", "Integrator indicated convergence!"),
        ("Energy converged!", "rms(grad) converged!"),
    ])
    def test_converged_run_counts(self, tmp_path, forward, backward):
        rxn, _ = self._run(tmp_path, forward=forward, backward=backward)

        assert rxn.ts_geom["1_tsopt_xtb_pysisyphus"].properties["irc_outcome"] == "intended"

    @pytest.mark.parametrize("log_kwargs", [
        pytest.param(dict(end_opts=("Converged!", "Number of cycles exceeded!")), id="end_opt_at_max_cycles"),
        pytest.param(dict(end_opts=("Number of cycles exceeded!",) * 2), id="both_end_opts_at_max_cycles"),
        pytest.param(dict(forward="Energy increased!"), id="irc_energy_increased"),
        pytest.param(dict(backward="IRC steps exceeded. Stopping."), id="irc_steps_exceeded"),
        pytest.param(dict(forward="rms(grad) below hard threshold."), id="irc_below_hard_threshold"),
    ])
    def test_unconverged_run_is_dropped(self, tmp_path, log_kwargs):
        rxn, _ = self._run(tmp_path, **log_kwargs)

        # The dropped run's TS gets no IRC result; irc_run2's TS (barrier 20
        # kJ/mol) becomes the reaction's, not irc_run1's lower 10.
        assert "irc_outcome" not in rxn.ts_geom["1_tsopt_xtb_pysisyphus"].properties
        assert rxn.ts_geom["2_tsopt_xtb_pysisyphus"].properties["irc_outcome"] == "intended"
        assert rxn.barrier["xtb_pysisyphus"] == kcal(20.0)


class TestOrcaRunConvergence:
    """
    The ORCA IRC counted as converged once "THE IRC HAS CONVERGED" appeared,
    but ORCA prints it once per direction. In the ts_handoff E3 trial
    (13453614 TS2) the backward direction stopped at MaxIter 200 while still
    falling (max|G| 0.049), and the run was labeled and given barriers. Now
    both directions must converge.
    """
    LOT = "PBE D3BJ def2-SVP"

    def _run(self, tmp_path, log):
        """irc_run1 is the run under test; irc_run2 is a converged control."""
        def write(run_dir, i):
            orca_outputs(run_dir, i)
            if i == 1:
                (run_dir / "irc_1.out").write_text(log)
        return run_irc(OrcaIRCValCalculator, self.LOT, "orca", [1, 2], write, tmp_path)

    def test_converged_run_counts(self, tmp_path):
        rxn, _ = self._run(tmp_path, ORCA_IRC_LOG)

        assert rxn.ts_geom[f"1_tsopt_{self.LOT}_orca"].properties["irc_outcome"] == "intended"

    @pytest.mark.parametrize("log", [
        pytest.param(ORCA_IRC_CONVERGED + ORCA_IRC_MAXITER + "ORCA TERMINATED NORMALLY\n", id="backward_at_max_iter"),
        pytest.param(ORCA_IRC_MAXITER + ORCA_IRC_CONVERGED + "ORCA TERMINATED NORMALLY\n", id="forward_at_max_iter"),
    ])
    def test_half_converged_run_is_dropped(self, tmp_path, log):
        rxn, _ = self._run(tmp_path, log)

        # The dropped run's TS gets no IRC result; irc_run2's TS (barrier 2)
        # becomes the reaction's, not irc_run1's lower 1.
        assert "irc_outcome" not in rxn.ts_geom[f"1_tsopt_{self.LOT}_orca"].properties
        assert rxn.ts_geom[f"2_tsopt_{self.LOT}_orca"].properties["irc_outcome"] == "intended"
        assert rxn.barrier[f"{self.LOT}_orca"] == pytest.approx(2.0)


class TestPysisMaxCycles:
    """
    Check that irc.max_cycles is used in all parts of IRC calculation setup
    """

    @pytest.mark.parametrize("max_cycles", [None, 100])
    def test_caps_irc_steps_and_end_opts(self, tmp_path, max_cycles):
        extra = {} if max_cycles is None else {"max_cycles": max_cycles}
        cfg = IRCValConfig(software="pysisyphus", lot="xtb", charge=0, multiplicity=1, **extra)
        calc = PysisyphusIRCValCalculator(SimpleNamespace(config=cfg), make_rxn("xtb", "pysisyphus", [1]), MagicMock())
        path = tmp_path / "irc_1_input.yaml"
        calc._write_pysis_irc_input(path, "ts_opt_1.xyz")
        text = path.read_text()

        expected = cfg.max_cycles
        irc_block = text.split("irc:\n")[1].split("endopt:\n")[0]
        endopt_block = text.split("endopt:\n")[1]
        assert f" max_cycles: {expected}\n" in irc_block
        assert f" max_cycles: {expected}\n" in endopt_block
        assert expected == (300 if max_cycles is None else max_cycles)


class TestOrcaIRCHessian:
    """
    The ORCA IRC computed two analytic Hessians at the TS: one for `! Freq`,
    which nothing reads, and the IRC's own (`InitHess calc_anfreq`). Without
    Freq, the ts_handoff check (PBE/def2-SVP, 17311254 TS1) gave the same
    endpoints with one Hessian.
    """

    def test_input_requests_one_hessian(self, tmp_path):
        lot = "PBE D3BJ def2-SVP"
        cfg = IRCValConfig(software="orca", lot=lot, charge=0, multiplicity=1)
        calc = OrcaIRCValCalculator(SimpleNamespace(config=cfg), make_rxn(lot, "orca", [1]), MagicMock())
        path = tmp_path / "irc_1.inp"
        calc._write_orca_irc_input(path, "ts_opt_1.xyz")
        text = path.read_text()

        assert "! IRC\n" in text
        assert "Freq" not in text
        assert "  InitHess calc_anfreq\n" in text


# Per software: a finished irc_run1 and a failed irc_run2, as {name: text}
# (a name ending in "/" is a dir), and what cleanup leaves in a run dir it cleans.
IRC_RUNS = {
    "pysisyphus": dict(
        cls=PysisyphusIRCValCalculator, lot="xtb",
        ok={"irc_1_input.yaml": "", "irc_1.log": pysis_irc_log(1), "forward_end_opt.xyz": "",
            "backward_end_opt.xyz": "", "irc_1.err": "", "ts_opt_1.xyz": ""},
        failed={"irc_2_input.yaml": "", "irc_2.log": "Crashed input:\n", "irc_2.err": "Traceback ...\n",
                "crashed_calculator_000/": "", "crashed_calculator_000/xtb.out": "SCF not converged\n"},
        cleaned={1: ["backward_end_opt.xyz", "forward_end_opt.xyz", "irc_1.log", "irc_1_input.yaml"],
                 2: ["irc_2.log", "irc_2_input.yaml"]},
    ),
    "orca": dict(
        cls=OrcaIRCValCalculator, lot="B3LYP def2-SVP",
        ok={"irc_1.inp": "", "irc_1.out": ORCA_IRC_LOG,
            "irc_1_IRC_F.xyz": "", "irc_1_IRC_B.xyz": "", "irc_1.gbw": "", "irc_1.err": ""},
        failed={"irc_2.inp": "", "irc_2.out": "ORCA finished by error termination in SCF\n",
                "irc_2.err": "", "irc_2.gbw": ""},
        cleaned={1: ["irc_1.inp", "irc_1.out", "irc_1_IRC_B.xyz", "irc_1_IRC_F.xyz"],
                 2: ["irc_2.inp", "irc_2.out"]},
    ),
}


def write_files(run_dir, files):
    for name, text in files.items():
        if name.endswith("/"):
            (run_dir / name).mkdir(parents=True, exist_ok=True)
        else:
            (run_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (run_dir / name).write_text(text)


def listing(run_dir):
    return sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*"))


@pytest.mark.parametrize("software", IRC_RUNS)
class TestKeepCrashLogs:
    """
    IRC validation is cleaned up when at least one run succeeded, which used to
    delete the failed runs' stderr and crash dirs too. With keep_crash_logs on,
    a failed run's dir is left whole; the successful runs are cleaned as before.
    """

    def _cleanup(self, tmp_path, software, keep_crash_logs):
        spec = IRC_RUNS[software]
        cfg = IRCValConfig(software=software, lot=spec["lot"], charge=0, multiplicity=1,
                           keep_crash_logs=keep_crash_logs)
        calc = spec["cls"](SimpleNamespace(config=cfg), make_rxn(spec["lot"], software, [1, 2]), MagicMock())
        calc.set_scratch_dir(tmp_path)
        write_files(tmp_path / "irc_run1", spec["ok"])
        write_files(tmp_path / "irc_run2", spec["failed"])
        before = listing(tmp_path / "irc_run2")
        calc.cleanup()
        return spec, before

    def test_on_keeps_the_failed_run_whole(self, tmp_path, software):
        spec, before = self._cleanup(tmp_path, software, keep_crash_logs=True)

        assert listing(tmp_path / "irc_run2") == before
        assert listing(tmp_path / "irc_run1") == spec["cleaned"][1]

    def test_off_cleans_every_run(self, tmp_path, software):
        spec, _ = self._cleanup(tmp_path, software, keep_crash_logs=False)

        assert listing(tmp_path / "irc_run2") == spec["cleaned"][2]
        assert listing(tmp_path / "irc_run1") == spec["cleaned"][1]


IRC_INNER_SCRIPTS = [
    pytest.param(PysisyphusIRCValCalculator, "pysisyphus", "xtb", "erm42_yarp_pysis_xtb.sif", "run_pysis_inner.sh",
                 "pysis irc_{i}_input.yaml > {log}irc_{i}.log 2> {log}irc_{i}.err", id="pysis"),
    pytest.param(OrcaIRCValCalculator, "orca", "B3LYP def2-SVP", "orca_6.0.1.sif", "run_orca_inner.sh",
                 "$orca irc_{i}.inp > {log}irc_{i}.out 2> {log}irc_{i}.err", id="orca"),
]


class TestInnerScriptLogs:
    """A staged IRC writes each run's log straight to the NFS run dir; unstaged is unchanged."""

    @pytest.mark.parametrize("staged", [False, True])
    @pytest.mark.parametrize("cls, software, lot, sif_name, script, line", IRC_INNER_SCRIPTS)
    def test_log_lines(self, tmp_path, staged, cls, software, lot, sif_name, script, line):
        sif_dir = tmp_path / "sifs"
        sif_dir.mkdir()
        (sif_dir / sif_name).touch()
        job_config = JobManagerConfig(scheduler="sge", container="apptainer", queue="long",
                                      sif_location=str(sif_dir))
        cfg = IRCValConfig(software=software, lot=lot, charge=0, multiplicity=1, use_local_scratch=staged)
        calc = cls(SimpleNamespace(config=cfg), make_rxn(lot, software, [1, 2]), job_config)
        calc.set_scratch_dir(tmp_path / "SCRATCH" / "abc123_refine.irc_validation")
        calc.generate_input()
        calc.write_submission_script()

        lines = (calc.scratch_dir / script).read_text().splitlines()
        for i in (1, 2):
            log = f"/nfs/irc_run{i}/" if staged else ""
            assert line.format(i=i, log=log) in lines
