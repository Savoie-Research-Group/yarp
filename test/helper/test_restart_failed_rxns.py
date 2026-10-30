"""
Tests for helper/restart_failed_rxns.py (yarp-restart), which puts failed
reactions back into a YARP run so their failed step is tried again.

Each test builds a small run directory by hand -- STATUS.json, the run's
pickle, failed_rxns.pkl and failed_status.json, as progress_yarp leaves them
when a reaction fails -- with a real KHP reaction object, then restarts it.
"""
import json
import pickle
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from helper.restart_failed_rxns import restart_failed, clear_task_results
from yarp.util.run_lock import run_lock, RunLockedError

XTB = SimpleNamespace(lot="xtb", software="pysisyphus", seed=42)


def _task(task_type, config=XTB):
    return SimpleNamespace(task_type=task_type, config=config)


CONFIG = SimpleNamespace(
    job_manager=SimpleNamespace(scheduler="sge"),
    pipeline_tasks={
        "ll_path.reactant_conformer": _task("reactant_conformer", SimpleNamespace(lot="uff", software="rdkit", seed=42)),
        "ll_path.ts_guess": _task("ts_guess", SimpleNamespace(gsm_lot="xtb", software="pysisyphus", seed=42)),
        "ll_refine.ts_opt": _task("transition_state_optimization"),
        "ll_refine.irc": _task("irc_validation"),
    })


def _t(status, job_id=None, scratch_dir=None, error_log=None):
    meta = {"status": status, "job_id": job_id, "scratch_dir": scratch_dir}
    if error_log:
        meta["error_log"] = error_log
    return meta


def _make_run(tmp_path, failed_tasks, rxn, rxn_hash="rxn_1", other_active=None):
    """A run directory with `rxn` failed (STATUS entry marked failed, object in failed_rxns.pkl)."""
    status = {"input_config": {}, "status_output_file": "STATUS.json",
              "reaction_output_file": "rxns.pkl", "global_tasks": {},
              "reactions": {rxn_hash: {"failed": True, "tasks": failed_tasks}}}
    active = dict(other_active or {})
    for h in active:
        status["reactions"][h] = {"tasks": {"ll_refine.irc": _t("terminated_normally")}}
    (tmp_path / "STATUS.json").write_text(json.dumps(status, indent=4))
    (tmp_path / "rxns.pkl").write_bytes(pickle.dumps(active))
    (tmp_path / "failed_rxns.pkl").write_bytes(pickle.dumps({rxn_hash: rxn}))
    (tmp_path / "failed_status.json").write_text(json.dumps(
        {rxn_hash: {tid: {"error_log": m.get("error_log"), "scratch_dir": m.get("scratch_dir")}
                    for tid, m in failed_tasks.items() if m["status"] in ("finished_with_error", "filtered_out")}}))


def _load(tmp_path):
    status = json.loads((tmp_path / "STATUS.json").read_text())
    active = pickle.loads((tmp_path / "rxns.pkl").read_bytes())
    failed = pickle.loads((tmp_path / "failed_rxns.pkl").read_bytes())
    failed_status = json.loads((tmp_path / "failed_status.json").read_text())
    return status, active, failed, failed_status


def _jm(running=()):
    """A job manager reporting the given job ids as still in the queue."""
    jm = MagicMock()
    jm.is_running.side_effect = lambda job_id: job_id in running
    return jm


def _tsopt_failed_run(tmp_path, khp_reaction):
    """TS optimization failed; GSM finished and stored a guess; the scrape left one partial TS."""
    khp_reaction.ts_geom["ts_guess_1_xtb_pysisyphus"] = "guess"
    khp_reaction.ts_geom["1_tsopt_xtb_pysisyphus"] = "partial"
    scratch = tmp_path / "SCRATCH" / "rxn_1_ll_refine.ts_opt"
    scratch.mkdir(parents=True)
    (scratch / "run_pysis.sh").write_text("#!/bin/bash\n")
    scratch.joinpath("run_pysis.sh").chmod(0o555)   # read-only, as left by a container job
    _make_run(tmp_path, {
        "ll_path.ts_guess": _t("terminated_normally", "1", str(tmp_path / "SCRATCH" / "rxn_1_ll_path.ts_guess")),
        "ll_refine.ts_opt": _t("finished_with_error", "2", str(scratch), "Output validation failed."),
        "ll_refine.irc": _t("pending"),
    }, khp_reaction)
    return scratch


class TestRestartRestoresTheFailedStep:

    def test_reaction_returns_to_the_run_with_only_the_failed_step_reset(self, tmp_path, khp_reaction):
        """
        After a restart: the reaction object is back in the run's pickle and
        gone from failed_rxns.pkl and failed_status.json; its `failed` mark is
        gone; the failed TS-opt is pending again, GSM stays finished (its
        results are kept and will be fast-forwarded), and the IRC waits as before.
        """
        _tsopt_failed_run(tmp_path, khp_reaction)

        result = restart_failed(tmp_path, all_failed=True, config=CONFIG, job_manager=_jm())

        status, active, failed, failed_status = _load(tmp_path)
        entry = status["reactions"]["rxn_1"]
        assert [r["hash"] for r in result["restarted"]] == ["rxn_1"]
        assert "rxn_1" in active and "rxn_1" not in failed and "rxn_1" not in failed_status
        assert "failed" not in entry
        assert {k: v["status"] for k, v in entry["tasks"].items()} == {
            "ll_path.ts_guess": "terminated_normally", "ll_refine.ts_opt": "pending", "ll_refine.irc": "pending"}

    def test_failed_attempt_is_recorded_and_its_scratch_folder_kept_aside(self, tmp_path, khp_reaction):
        """
        The failed attempt's folder is renamed to `.attempt1` with its contents
        intact (here a read-only script, as container jobs can leave behind),
        so the retry gets a fresh folder and nothing has to be overwritten. The
        task records the attempt: its job id, error and where its folder went.
        """
        scratch = _tsopt_failed_run(tmp_path, khp_reaction)

        restart_failed(tmp_path, all_failed=True, config=CONFIG, job_manager=_jm())

        kept = scratch.with_name(scratch.name + ".attempt1")
        assert not scratch.exists() and (kept / "run_pysis.sh").exists()
        meta = _load(tmp_path)[0]["reactions"]["rxn_1"]["tasks"]["ll_refine.ts_opt"]
        assert meta["attempt"] == 2 and meta["job_id"] is None and meta["scratch_dir"] is None
        assert "error_log" not in meta
        assert meta["attempts"] == [{"attempt": 1, "status": "finished_with_error", "job_id": "2",
                                     "error_log": "Output validation failed.", "scratch_dir": str(kept)}]

    def test_partial_results_of_the_failed_step_are_cleared_others_kept(self, tmp_path, khp_reaction):
        """
        The failed TS-opt's scrape left one optimized TS behind; the progress
        fast-forward would take that as "TS-opt done" and skip the retry. It is
        removed. GSM's guess, from the step that finished, is kept.
        """
        _tsopt_failed_run(tmp_path, khp_reaction)

        restart_failed(tmp_path, all_failed=True, config=CONFIG, job_manager=_jm())

        rxn = _load(tmp_path)[1]["rxn_1"]
        assert "1_tsopt_xtb_pysisyphus" not in rxn.ts_geom
        assert rxn.ts_geom["ts_guess_1_xtb_pysisyphus"] == "guess"

    def test_backups_are_written_before_anything_changes(self, tmp_path, khp_reaction):
        _tsopt_failed_run(tmp_path, khp_reaction)
        before = {f: (tmp_path / f).read_bytes() for f in ("STATUS.json", "rxns.pkl", "failed_rxns.pkl", "failed_status.json")}

        restart_failed(tmp_path, all_failed=True, config=CONFIG, job_manager=_jm())

        (backup,) = (tmp_path / "restart_backups").iterdir()
        assert {f: (backup / f).read_bytes() for f in before} == before


class TestWhatIsNotRestarted:

    def test_reaction_with_a_job_still_in_the_queue_is_left_alone(self, tmp_path, khp_reaction):
        """
        The reactant branch failed while the TS-opt job (id 7) is still queued:
        that reaction is skipped and nothing about the run changes.
        """
        _make_run(tmp_path, {
            "ll_path.reactant_conformer": _t("finished_with_error", "6", None, "Output validation failed."),
            "ll_refine.ts_opt": _t("submitted", "7", "/s/tsopt")}, khp_reaction)
        before = (tmp_path / "STATUS.json").read_text()

        result = restart_failed(tmp_path, all_failed=True, config=CONFIG, job_manager=_jm(running={"7"}))

        assert "job 7" in result["skipped"]["rxn_1"]
        assert (tmp_path / "STATUS.json").read_text() == before

    def test_a_finished_but_unchecked_job_stays_submitted(self, tmp_path, khp_reaction):
        """
        Same, but job 7 has ended: the reaction is restarted, and the TS-opt
        task stays "submitted" with its job id, so the next progress pass checks
        and keeps its output like any finished job, instead of re-running it.
        """
        _make_run(tmp_path, {
            "ll_path.reactant_conformer": _t("finished_with_error", "6", None, "Output validation failed."),
            "ll_refine.ts_opt": _t("submitted", "7", "/s/tsopt")}, khp_reaction)

        restart_failed(tmp_path, all_failed=True, config=CONFIG, job_manager=_jm())

        tasks = _load(tmp_path)[0]["reactions"]["rxn_1"]["tasks"]
        assert tasks["ll_refine.ts_opt"] == _t("submitted", "7", "/s/tsopt")
        assert tasks["ll_path.reactant_conformer"]["status"] == "pending"

    def test_filtered_reactions_need_include_filtered(self, tmp_path, khp_reaction):
        """An IRC 'unintended' diversion is skipped by --all, and restarted only with --include-filtered."""
        _make_run(tmp_path, {"ll_refine.irc": _t("filtered_out", "8", "/s/irc", "IRC validation failed: Outcome was 'unintended'.")},
                  khp_reaction)

        result = restart_failed(tmp_path, all_failed=True, config=CONFIG, job_manager=_jm())
        assert "include-filtered" in result["skipped"]["rxn_1"]

        restart_failed(tmp_path, all_failed=True, include_filtered=True, config=CONFIG, job_manager=_jm())
        assert _load(tmp_path)[0]["reactions"]["rxn_1"]["tasks"]["ll_refine.irc"]["status"] == "pending"

    def test_dry_run_changes_nothing(self, tmp_path, khp_reaction):
        """--dry-run reports the plan (including what it would clear) and leaves every file, folder and backup alone."""
        scratch = _tsopt_failed_run(tmp_path, khp_reaction)
        before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}

        result = restart_failed(tmp_path, all_failed=True, dry_run=True, config=CONFIG, job_manager=_jm())

        assert result["restarted"][0]["tasks"][0]["removed_results"] == ["ts_geom:1_tsopt_xtb_pysisyphus"]
        assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before
        assert scratch.exists() and not (tmp_path / "restart_backups").exists()

    def test_hash_selects_only_the_named_reactions(self, tmp_path, khp_reaction):
        _tsopt_failed_run(tmp_path, khp_reaction)

        result = restart_failed(tmp_path, hashes=["rxn_other"], config=CONFIG, job_manager=_jm())

        assert result["restarted"] == [] and result["skipped"] == {"rxn_other": "not in this run"}
        assert _load(tmp_path)[0]["reactions"]["rxn_1"]["failed"] is True

    def test_refuses_while_a_progress_pass_holds_the_lock(self, tmp_path, khp_reaction):
        """A progress pass rewrites STATUS.json at its end; restarting during one would be lost."""
        _tsopt_failed_run(tmp_path, khp_reaction)

        with run_lock(tmp_path, "yarp-progress"):
            with pytest.raises(RunLockedError, match="yarp-progress"):
                restart_failed(tmp_path, all_failed=True, config=CONFIG, job_manager=_jm())


class TestNewSeed:

    def test_seeded_conformer_retry_gets_seed_plus_attempt(self, tmp_path, khp_reaction):
        """
        An RDKit conformer step failed. With --new-seed its retry carries seed
        42 + 1 = 43 on the task (the stage's shared config keeps 42); without
        --new-seed no seed is set and the retry reuses 42.
        """
        tasks = {"ll_path.reactant_conformer": _t("finished_with_error", "3", None, "Output validation failed.")}
        _make_run(tmp_path, tasks, khp_reaction)
        restart_failed(tmp_path, all_failed=True, new_seed=True, config=CONFIG, job_manager=_jm())
        assert _load(tmp_path)[0]["reactions"]["rxn_1"]["tasks"]["ll_path.reactant_conformer"]["seed"] == 43

        _make_run(tmp_path, tasks, khp_reaction)
        restart_failed(tmp_path, all_failed=True, config=CONFIG, job_manager=_jm())
        assert "seed" not in _load(tmp_path)[0]["reactions"]["rxn_1"]["tasks"]["ll_path.reactant_conformer"]

    def test_unseeded_steps_get_no_seed(self, tmp_path, khp_reaction):
        """--new-seed only applies to seeded conformer software; a failed TS-opt retries unchanged."""
        _tsopt_failed_run(tmp_path, khp_reaction)

        restart_failed(tmp_path, all_failed=True, new_seed=True, config=CONFIG, job_manager=_jm())

        assert "seed" not in _load(tmp_path)[0]["reactions"]["rxn_1"]["tasks"]["ll_refine.ts_opt"]


class TestClearTaskResults:
    """Which stored results each task type owns, at its own level of theory only."""

    def test_only_this_level_of_theory_is_cleared(self, khp_reaction):
        """An IRC failure at xtb clears the xtb barrier, not a DFT one stored for the same reaction."""
        khp_reaction.barrier.update({"xtb_pysisyphus": 30.0, "b3lyp_orca": 25.0})

        removed = clear_task_results(khp_reaction, "irc_validation", XTB)

        assert khp_reaction.barrier == {"b3lyp_orca": 25.0}
        assert removed == ["barrier:xtb_pysisyphus"]

    def test_gsm_clears_its_guesses_and_guess_conformers(self, khp_reaction):
        khp_reaction.ts_geom["ts_guess_1_xtb_pysisyphus"] = "g"
        khp_reaction.reactant.conformers["guess_conf_1_xtb_pysisyphus"] = "r"
        khp_reaction.reactant.conformers["conf_gen_rank0_uff_rdkit"] = "keep"

        clear_task_results(khp_reaction, "ts_guess", SimpleNamespace(gsm_lot="xtb", software="pysisyphus"))

        assert "ts_guess_1_xtb_pysisyphus" not in khp_reaction.ts_geom
        assert "guess_conf_1_xtb_pysisyphus" not in khp_reaction.reactant.conformers
        assert khp_reaction.reactant.conformers["conf_gen_rank0_uff_rdkit"] == "keep"
