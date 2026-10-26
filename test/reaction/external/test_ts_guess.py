"""
Tests for the GSM (TS guess) job scripts.
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
