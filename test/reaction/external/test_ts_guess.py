"""
Tests for the GSM (TS guess) job scripts and cleanup.
"""
from types import SimpleNamespace

import pytest

from yarp.reaction.external.ts_guess import PysisyphusTSGuessCalculator
from yarp.util.config import JobManagerConfig, TSGuessConfig


class TestInnerScriptLogs:
    """A staged GSM writes each pair's log straight to the NFS run dir; unstaged is unchanged."""

    @pytest.mark.parametrize("staged", [False, True])
    def test_log_lines(self, tmp_path, staged):
        sif_dir = tmp_path / "sifs"
        sif_dir.mkdir()
        (sif_dir / "erm42_yarp_pysis_xtb.sif").touch()
        job_config = JobManagerConfig(scheduler="sge", container="apptainer", queue="long",
                                      sif_location=str(sif_dir))
        cfg = TSGuessConfig(software="pysisyphus", gsm_lot="xtb", charge=0, multiplicity=1,
                            use_local_scratch=staged)
        calc = PysisyphusTSGuessCalculator(SimpleNamespace(config=cfg), SimpleNamespace(), job_config)
        calc.set_scratch_dir(tmp_path / "SCRATCH" / "abc123_path.ts_guess")
        # write_submission_script only counts the pairs; selecting them needs real conformers.
        calc.pairs_to_run = [None, None]
        calc.write_submission_script()

        lines = (calc.scratch_dir / "run_pysis_inner.sh").read_text().splitlines()
        for i in (1, 2):
            log = f"/nfs/gsm_run{i}/" if staged else ""
            assert f"pysis gsm_{i}_input.yaml > {log}gsm_{i}.log 2> {log}gsm_{i}.err" in lines


def gsm_runs(scratch):
    """
    gsm_run1 finished; gsm_run2 crashed the way 5 GSM runs did in the
    ts_handoff step 0.3 runs: no splined HEI, and pysisyphus copied the
    crashed image's calculation into crashed_image_028/. Each run also has a
    stderr file and an xTB calc dir that cleanup normally deletes.
    """
    run1, run2 = scratch / "gsm_run1", scratch / "gsm_run2"
    for run in (run1, run2):
        (run / "image_000").mkdir(parents=True)
    (run1 / "gsm_1.log").write_text("Wrote splined HEI to 'splined_hei.xyz'\npysisyphus run took 0:03:00 h.\n")
    (run1 / "final_geometries.trj").touch()
    (run1 / "splined_hei.xyz").touch()
    (run1 / "gsm_1.err").touch()
    (run2 / "gsm_2.log").write_text("Crashed input:\n...\n'/work/gsm_run2/crashed_image_028'.\n")
    (run2 / "gsm_2.err").write_text("Traceback ...\n")
    (run2 / "crashed_image_028").mkdir()
    (run2 / "crashed_image_028" / "xtb.out").write_text("SCF not converged\n")
    return run1, run2


def listing(run_dir):
    return sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*"))


class TestKeepCrashLogs:
    """
    GSM is cleaned up when at least one run succeeded, and that used to delete
    the crashed runs' stderr and crashed_image_* dirs too, so GSM crashes
    couldn't be diagnosed. With keep_crash_logs on, a failed run's dir is left
    whole; the successful runs are cleaned as before.
    """

    def _cleanup(self, tmp_path, keep_crash_logs):
        cfg = TSGuessConfig(software="pysisyphus", gsm_lot="xtb", charge=0, multiplicity=1,
                            keep_crash_logs=keep_crash_logs)
        calc = PysisyphusTSGuessCalculator(SimpleNamespace(config=cfg), SimpleNamespace(),
                                           JobManagerConfig())
        calc.set_scratch_dir(tmp_path)
        run1, run2 = gsm_runs(tmp_path)
        before = listing(run2)
        calc.cleanup()
        return run1, run2, before

    def test_on_keeps_the_failed_run_whole(self, tmp_path):
        run1, run2, before = self._cleanup(tmp_path, keep_crash_logs=True)

        assert listing(run2) == before
        assert listing(run1) == ["final_geometries.trj", "gsm_1.log", "splined_hei.xyz"]

    def test_off_cleans_every_run(self, tmp_path):
        run1, run2, _ = self._cleanup(tmp_path, keep_crash_logs=False)

        assert listing(run2) == ["gsm_2.log"]
        assert listing(run1) == ["final_geometries.trj", "gsm_1.log", "splined_hei.xyz"]


THREE_H = "3\n\nH 0 0 0\nH 0.74 0 0\nH 3.0 0 0\n"


class TestScrapeSkipsFailedRuns:
    """
    scrape_data used to run its own looser check than check_output: a log with
    "Wrote splined HEI" but no "pysisyphus run took" (pysisyphus died after
    writing the HEI) was scraped as a TS guess, though check_output counted the
    run as failed. Both now use _is_run_successful.
    """

    def test_only_runs_that_check_output_accepts_are_scraped(self, tmp_path, capsys):
        cfg = TSGuessConfig(software="pysisyphus", gsm_lot="xtb", charge=0, multiplicity=1)
        rxn = SimpleNamespace(ts_geom={}, reactant=SimpleNamespace(conformers={}),
                              product=SimpleNamespace(conformers={}))
        calc = PysisyphusTSGuessCalculator(SimpleNamespace(config=cfg), rxn, JobManagerConfig())
        calc.set_scratch_dir(tmp_path)
        logs = {1: "Wrote splined HEI to 'splined_hei.xyz'\npysisyphus run took 0:03:00 h.\n",
                2: "Wrote splined HEI to 'splined_hei.xyz'\n"}
        for i, log in logs.items():
            run_dir = tmp_path / f"gsm_run{i}"
            run_dir.mkdir()
            (run_dir / f"gsm_{i}.log").write_text(log)
            (run_dir / "splined_hei.xyz").write_text(THREE_H)
            (run_dir / "final_geometries.trj").write_text(THREE_H + THREE_H)

        assert calc.check_output()
        assert "Run 2 failed" in capsys.readouterr().out
        calc.scrape_data()

        assert list(rxn.ts_geom) == ["ts_guess_1_xtb_pysisyphus"]
        assert list(rxn.reactant.conformers) == ["guess_conf_1_xtb_pysisyphus"]
        assert list(rxn.product.conformers) == ["guess_conf_1_xtb_pysisyphus"]
