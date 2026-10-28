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
from yarp.util.config import TSOptConfig, InitialGeomConfig, GeomSourceConfig, JobManagerConfig
from yarp.util.constants import Constants

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


def make_calc(cls, ts_geom, ts_source, lot=HL_LOT, software="orca", job_config=None):
    src = GeomSourceConfig(label="rp_opt", lot="xtb", software="pysisyphus")
    cfg = TSOptConfig(software=software, lot=lot, charge=0, multiplicity=1)
    cfg.initial_geom = InitialGeomConfig(reactant=src, product=src, transition_state=ts_source)
    return cls(SimpleNamespace(config=cfg), SimpleNamespace(ts_geom=ts_geom),
               job_config or MagicMock(container="docker"))


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
    run 4 converged to a minimum, run 5 to a second-order saddle. Plus GSM
    guesses and another level's TS opt that must not be picked up.
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
    return {c.type: c for c in confs}


XTB_TS_OPT = GeomSourceConfig(label="ts_opt", lot="xtb", software="pysisyphus")
XTB_TS_GUESS = GeomSourceConfig(label="ts_guess", lot="xtb", software="pysisyphus")


class TestTSOptSource:
    def test_starts_from_every_valid_source_ts(self, tmp_path):
        calc = make_calc(OrcaTSOptCalculator, xtb_refined_rxn(), XTB_TS_OPT)

        # 1, 3 and 6 are first-order saddles at xtb/pysisyphus, in insertion
        # order; 4 and 5 aren't saddles; 70 is another level; 91/92 are guesses.
        assert written_tags(calc, tmp_path) == [1, 3, 6]

    def test_ready_when_one_valid_ts_exists(self):
        ts_geom = {"2_tsopt_xtb_pysisyphus": ts_conf(2, "2_tsopt_xtb_pysisyphus", ONE_IMAG)}

        assert make_calc(OrcaTSOptCalculator, ts_geom, XTB_TS_OPT).has_prerequisites()

    def test_not_ready_without_a_valid_ts(self):
        ts_geom = {
            "1_tsopt_xtb_pysisyphus": ts_conf(1, "1_tsopt_xtb_pysisyphus", NO_IMAG),
            "2_tsopt_xtb_pysisyphus": ts_conf(2, "2_tsopt_xtb_pysisyphus", TWO_IMAG),
            "1_tsopt_B3LYP def2-SVP_orca": ts_conf(70, "1_tsopt_B3LYP def2-SVP_orca", ONE_IMAG),
        }

        assert not make_calc(OrcaTSOptCalculator, ts_geom, XTB_TS_OPT).has_prerequisites()


# A soft saddle (ts_handoff check 18: KHP 16610336 TS2, -35.5 cm-1, became a
# -25 cm-1 DFT "TS"), and a stiff TS with a small extra imaginary mode (KHP
# 17011678 TS3: -628 and -17.5 cm-1).
SOFT_IMAG = np.array([-35.0, 100.0, 200.0])
STIFF_PLUS_NOISE = np.array([-628.0, -17.5, 200.0])


class TestMinImagFreq:
    """
    ts_opt.min_imag_freq (cm-1) sets which source-level TSs a TS optimization
    starts from: exactly one mode below -min_imag_freq. Smaller imaginary modes
    are ignored as noise. The default, 0.01, is the old rule.
    """

    def _calc(self, min_imag_freq=None):
        ts_geom = {c.type: c for c in (ts_conf(1, "1_tsopt_xtb_pysisyphus", ONE_IMAG),
                                       ts_conf(2, "2_tsopt_xtb_pysisyphus", SOFT_IMAG),
                                       ts_conf(3, "3_tsopt_xtb_pysisyphus", STIFF_PLUS_NOISE))}
        calc = make_calc(OrcaTSOptCalculator, ts_geom, XTB_TS_OPT)
        if min_imag_freq is not None:
            calc.config.min_imag_freq = min_imag_freq
        return calc

    def test_default_keeps_the_old_rule(self, tmp_path):
        assert written_tags(self._calc(), tmp_path) == [1, 2]

    def test_cutoff_drops_the_soft_ts_and_ignores_a_small_extra_mode(self, tmp_path):
        assert written_tags(self._calc(50.0), tmp_path) == [1, 3]

    def test_not_ready_when_every_source_ts_is_soft(self):
        ts_geom = {"1_tsopt_xtb_pysisyphus": ts_conf(1, "1_tsopt_xtb_pysisyphus", SOFT_IMAG)}
        calc = make_calc(OrcaTSOptCalculator, ts_geom, XTB_TS_OPT)
        calc.config.min_imag_freq = 50.0

        assert not calc.has_prerequisites()


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


class TestOrcaJobScripts:
    """
    The two scripts an ORCA TS optimization writes under SGE + apptainer: an
    outer script that carries the scheduler header and launches the container,
    and an inner script that runs each tsopt_run{i} in turn inside it.
    """

    def _write(self, tmp_path, staged=False):
        sif_dir = tmp_path / "sifs"
        sif_dir.mkdir()
        sif = sif_dir / "orca_6.0.1.sif"
        sif.touch()
        job_config = JobManagerConfig(scheduler="sge", container="apptainer", queue="long",
                                      sif_location=str(sif_dir))
        ts_geom = {key: ts_conf(tag, key, ONE_IMAG)
                   for tag, key in ((1, "1_tsopt_xtb_pysisyphus"), (2, "2_tsopt_xtb_pysisyphus"))}

        calc = make_calc(OrcaTSOptCalculator, ts_geom, XTB_TS_OPT, job_config=job_config)
        calc.config.use_local_scratch = staged
        scratch = tmp_path / "SCRATCH" / "abc123_refine.transition_state_optimization"
        calc.set_scratch_dir(scratch)
        calc.generate_input()
        return calc.write_submission_script(), scratch, sif

    def test_outer_script(self, tmp_path):
        submit, scratch, sif = self._write(tmp_path)

        assert submit == str(scratch / "submit_tsopt.sh")
        assert (scratch / "submit_tsopt.sh").read_text() == (
            "#!/bin/bash\n"
            "#$ -N yarp\n"
            "#$ -q long\n"
            "#$ -pe smp 1\n"
            "#$ -l h_vmem=4000M\n"
            "#$ -l h_rt=01:00:00\n"
            "#$ -notify\n"
            "#$ -o /dev/null\n"
            "#$ -e /dev/null\n"
            "\n"
            f"apptainer exec -e  --bind {scratch}:/work --pwd /work {sif} bash /work/run_orca_inner.sh\n"
        )

    def test_inner_script(self, tmp_path):
        _, scratch, _ = self._write(tmp_path)

        assert (scratch / "run_orca_inner.sh").read_text() == ORCA_INNER_SCRIPT

    def test_staged_outer_script(self, tmp_path):
        """Staging adds the staging block and changes the container line; nothing else."""
        _, scratch, sif = self._write(tmp_path, staged=True)

        assert (scratch / "submit_tsopt.sh").read_text() == (
            "#!/bin/bash\n"
            "#$ -N yarp\n"
            "#$ -q long\n"
            "#$ -pe smp 1\n"
            "#$ -l h_vmem=4000M\n"
            "#$ -l h_rt=01:00:00\n"
            "#$ -notify\n"
            "#$ -o /dev/null\n"
            "#$ -e /dev/null\n"
            "\n"
            "# Stage this task in node-local scratch (use_local_scratch)\n"
            f'YARP_NFS_DIR="{scratch}"\n'
            'yarp_staging_error() { echo "$(date \'+%F %T\') $(hostname): $*" >> "$YARP_NFS_DIR/local_scratch.err"; }\n'
            'YARP_LOCAL_DIR=$(mktemp -d "$TMPDIR/yarp.XXXXXX" 2>&1) || '
            '{ yarp_staging_error "could not create the local scratch dir: $YARP_LOCAL_DIR"; exit 1; }\n'
            "yarp_copy_back() {\n"
            "    local out\n"
            '    out=$(rsync -a "$YARP_LOCAL_DIR"/ "$YARP_NFS_DIR"/ 2>&1) && return 0\n'
            '    yarp_staging_error "copy-back failed, local copy kept at $YARP_LOCAL_DIR: $out"\n'
            "    return 1\n"
            "}\n"
            'trap \'yarp_copy_back && rm -rf "$YARP_LOCAL_DIR"\' EXIT\n'
            "trap 'yarp_copy_back' USR1\n"
            "trap 'exit 140' USR2\n"
            'YARP_OUT=$(cp -a "$YARP_NFS_DIR"/. "$YARP_LOCAL_DIR"/ 2>&1) || '
            '{ yarp_staging_error "copy-in failed: $YARP_OUT"; exit 1; }\n'
            "yarp_bg() {\n"
            '    "$@" &\n'
            "    local pid=$!\n"
            "    while kill -0 $pid 2>/dev/null; do wait $pid; done\n"
            "    wait $pid\n"
            "}\n"
            "\n"
            f'yarp_bg apptainer exec -e  --bind "$YARP_LOCAL_DIR":/work --bind "$YARP_NFS_DIR":/nfs '
            f"--pwd /work {sif} bash /work/run_orca_inner.sh\n"
        )

    def test_staged_inner_script_logs_to_nfs(self, tmp_path):
        """ORCA's .out/.err go straight to the NFS run dir, so they can be followed during the run."""
        _, scratch, _ = self._write(tmp_path, staged=True)

        assert (scratch / "run_orca_inner.sh").read_text() == (
            ORCA_INNER_SCRIPT
            .replace("> tsopt_1.out 2> tsopt_1.err", "> /nfs/tsopt_run1/tsopt_1.out 2> /nfs/tsopt_run1/tsopt_1.err")
            .replace("> tsopt_2.out 2> tsopt_2.err", "> /nfs/tsopt_run2/tsopt_2.out 2> /nfs/tsopt_run2/tsopt_2.err")
        )


class TestPysisInnerScriptLogs:
    @pytest.mark.parametrize("staged", [False, True])
    def test_log_lines(self, tmp_path, staged):
        sif_dir = tmp_path / "sifs"
        sif_dir.mkdir()
        (sif_dir / "erm42_yarp_pysis_xtb.sif").touch()
        job_config = JobManagerConfig(scheduler="sge", container="apptainer", queue="long",
                                      sif_location=str(sif_dir))
        calc = make_calc(PysisyphusTSOptCalculator, xtb_refined_rxn(), XTB_TS_GUESS,
                         lot="xtb", software="pysisyphus", job_config=job_config)
        calc.config.use_local_scratch = staged
        calc.set_scratch_dir(tmp_path / "SCRATCH" / "abc123_refine.transition_state_optimization")
        calc.generate_input()
        calc.write_submission_script()

        lines = (calc.scratch_dir / "run_pysis_inner.sh").read_text().splitlines()
        for i in (1, 2):
            log = f"/nfs/tsopt_run{i}/" if staged else ""
            assert f"pysis tsopt_{i}_input.yaml > {log}tsopt_{i}.log 2> {log}tsopt_{i}.err" in lines


def pysis_opt_log(outcome):
    """
    The lines of a pysisyphus TS-opt log that YARP reads, in order. `outcome`
    is the optimizer's last word: "Converged!" or "Number of cycles exceeded!".
    Pysisyphus writes the final geometry and its Hessian either way.
    """
    return (f"        {outcome}\n"
            "        \tenergy: -21.61755730 hartree\n"
            "        Wrote final, hopefully optimized, geometry to 'ts_final_geometry.xyz'\n"
            "pysisyphus run took 0:00:30 h.\n")


class TestPysisRunConvergence:
    """
    A pysisyphus TS optimization that hit max_cycles used to count as a
    finished run. It was scraped as a TS, and a single imaginary mode made it
    "valid" for the IRC and the next level (17413990 TS2 in the ts_handoff
    step 0.3 xtb run: -86 cm-1, below both minima). Only converged runs count.
    """

    def _calc(self, tmp_path, outcomes):
        calc = make_calc(PysisyphusTSOptCalculator, {}, XTB_TS_GUESS, lot="xtb", software="pysisyphus")
        calc.set_scratch_dir(tmp_path)
        for i, outcome in enumerate(outcomes, start=1):
            run_dir = tmp_path / f"tsopt_run{i}"
            run_dir.mkdir()
            (run_dir / "ts_final_geometry.xyz").write_text("3\n\nH 0 0 0\nH 1.5 0 0\nH 2.5 0 0\n")
            (run_dir / "ts_final_hessian.h5").touch()
            (run_dir / f"tsopt_{i}.log").write_text(pysis_opt_log(outcome))
        return calc

    def test_converged_run_succeeds(self, tmp_path):
        assert self._calc(tmp_path, ["Converged!"])._is_run_successful(1)

    def test_run_at_max_cycles_fails(self, tmp_path):
        assert not self._calc(tmp_path, ["Number of cycles exceeded!"])._is_run_successful(1)

    def test_unconverged_run_is_not_scraped(self, tmp_path, monkeypatch):
        calc = self._calc(tmp_path, ["Converged!", "Number of cycles exceeded!"])
        monkeypatch.setattr(calc, "_parse_hessian_freq", lambda hess_file: (None, ONE_IMAG.copy()))

        assert calc.check_output()
        calc.scrape_data()

        assert list(calc.rxn.ts_geom) == ["1_tsopt_xtb_pysisyphus"]


# Per software: a finished tsopt_run1 and a failed tsopt_run2, as {name: text}
# (a name ending in "/" is a dir), and what cleanup leaves in a run dir it cleans.
TSOPT_RUNS = {
    "pysisyphus": dict(
        cls=PysisyphusTSOptCalculator, lot="xtb",
        ok={"tsopt_1.log": pysis_opt_log("Converged!"), "ts_final_geometry.xyz": "", "ts_final_hessian.h5": "",
            "tsopt_1.err": "", "calculator_000.out": ""},
        failed={"tsopt_2.log": "Crashed input:\n", "tsopt_2.err": "Traceback ...\n",
                "crashed_calculator_000/": "", "crashed_calculator_000/xtb.out": "SCF not converged\n"},
        cleaned={1: ["ts_final_geometry.xyz", "tsopt_1.log"], 2: ["tsopt_2.log"]},
    ),
    "orca": dict(
        cls=OrcaTSOptCalculator, lot=HL_LOT,
        ok={"tsopt_1.out": "THE OPTIMIZATION HAS CONVERGED\nORCA TERMINATED NORMALLY\n",
            "tsopt_1.xyz": "", "tsopt_1.hess": "", "tsopt_1.gbw": "", "tsopt_1.err": ""},
        failed={"tsopt_2.out": "ORCA finished by error termination in SCF\n", "tsopt_2.err": "",
                "tsopt_2.gbw": ""},
        cleaned={1: ["tsopt_1.out", "tsopt_1.xyz"], 2: ["tsopt_2.out"]},
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


@pytest.mark.parametrize("software", TSOPT_RUNS)
class TestKeepCrashLogs:
    """
    A TS optimization is cleaned up when at least one run succeeded, which
    used to delete the failed runs' stderr and crash dirs too. With
    keep_crash_logs on, a failed run's dir is left whole; the successful runs
    are cleaned as before.
    """

    def _cleanup(self, tmp_path, software, keep_crash_logs):
        spec = TSOPT_RUNS[software]
        calc = make_calc(spec["cls"], {}, XTB_TS_OPT, lot=spec["lot"], software=software)
        calc.config.keep_crash_logs = keep_crash_logs
        calc.set_scratch_dir(tmp_path)
        write_files(tmp_path / "tsopt_run1", spec["ok"])
        write_files(tmp_path / "tsopt_run2", spec["failed"])
        before = listing(tmp_path / "tsopt_run2")
        calc.cleanup()
        return spec, before

    def test_on_keeps_the_failed_run_whole(self, tmp_path, software):
        spec, before = self._cleanup(tmp_path, software, keep_crash_logs=True)

        assert listing(tmp_path / "tsopt_run2") == before
        assert listing(tmp_path / "tsopt_run1") == spec["cleaned"][1]

    def test_off_cleans_every_run(self, tmp_path, software):
        spec, _ = self._cleanup(tmp_path, software, keep_crash_logs=False)

        assert listing(tmp_path / "tsopt_run2") == spec["cleaned"][2]
        assert listing(tmp_path / "tsopt_run1") == spec["cleaned"][1]


def orca_thermo_block(h, s, g):
    """One ORCA thermochemistry block's GIBBS FREE ENERGY section, in Eh, as ORCA prints it."""
    return ("GIBBS FREE ENERGY\n-------------------\n\nThe Gibbs free energy is G = H - T*S\n\n"
            f"Total enthalpy                    ...   {h:.8f} Eh \n"
            f"Total entropy correction          ...     {s:.8f} Eh    -21.63 kcal/mol\n"
            "-----------------------------------------------------------------------\n"
            f"Final Gibbs free energy         ...   {g:.8f} Eh\n")


# An ORCA opt with Calc_Hess + Recalc_Hess prints one thermochemistry block per
# exact Hessian, the starting geometry's first; with Freq, the final geometry's
# last. (H, -T*S, G) per block, from checks/12 of ts_handoff (PBE/def2-SVP TS
# opt with Freq: Hessians at cycles 1, 4, 7, then the final geometry).
OPT_THERMO_BLOCKS = [(-342.67569344, -0.03447304, -342.71016648),
                     (-342.68076681, -0.03853593, -342.71930274),
                     (-342.68085923, -0.03841772, -342.71927695),
                     (-342.68092217, -0.03841495, -342.71933711)]


class TestOrcaFinalHessian:
    """
    The ORCA TS opt used to read the Hessian and Gibbs energy of the wrong
    geometry. Its .hess held the last Recalc_Hess Hessian (up to N - 1
    cycles stale), and _parse_orca_thermo returned the first thermochemistry
    block, the starting geometry's: +5.7 kcal/mol in the ts_handoff check
    (PBE/def2-SVP, 17311254 TS1). Freq adds a Hessian at the final geometry,
    and the parser returns the last block.
    """

    def test_input_requests_a_final_hessian(self, tmp_path):
        calc = make_calc(OrcaTSOptCalculator, {}, XTB_TS_OPT)
        calc._write_orca_ts_opt_input(tmp_path / "tsopt_1.inp", "ts_guess_1.xyz")

        assert "! OptTS Freq" in (tmp_path / "tsopt_1.inp").read_text().splitlines()

    def test_gibbs_is_the_last_thermochemistry_block(self, tmp_path):
        out = tmp_path / "tsopt_1.out"
        out.write_text("".join(orca_thermo_block(*b) + "\n...\n" for b in OPT_THERMO_BLOCKS))
        calc = make_calc(OrcaTSOptCalculator, {}, XTB_TS_OPT)

        enthalpy, entropy, gibbs = calc._parse_orca_thermo(out)

        h, s, g = OPT_THERMO_BLOCKS[-1]
        assert (enthalpy, entropy, gibbs) == pytest.approx(
            (h * Constants.ha_to_kcalmol, s * Constants.ha_to_kcalmol, g * Constants.ha_to_kcalmol), abs=1e-6)


ORCA_INNER_SCRIPT = (
    "#!/bin/bash\n"
    "\n"
    "echo 'Starting YARP serial TSOPT execution...'\n"
    "\n"
    "echo '--- Running TSOPT 1 ---'\n"
    "cd /work/tsopt_run1\n"
    "orca=$(which orca)\n"
    "$orca tsopt_1.inp > tsopt_1.out 2> tsopt_1.err\n"
    "echo '--- Finished TSOPT 1 ---'\n"
    "\n"
    "echo '--- Running TSOPT 2 ---'\n"
    "cd /work/tsopt_run2\n"
    "orca=$(which orca)\n"
    "$orca tsopt_2.inp > tsopt_2.out 2> tsopt_2.err\n"
    "echo '--- Finished TSOPT 2 ---'\n"
    "\n"
)
