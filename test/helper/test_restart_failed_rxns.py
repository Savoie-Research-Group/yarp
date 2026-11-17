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

import helper.restart_failed_rxns as restart_mod
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


def _fail_again(tmp_path, scratch, job_id="3"):
    """The retry failed too: progress marks it failed in a new folder at the original path."""
    scratch.mkdir(parents=True)
    (scratch / "tsopt.log").write_text("failed again\n")
    status, active, failed, failed_status = _load(tmp_path)
    entry = status["reactions"]["rxn_1"]
    entry["failed"] = True
    entry["tasks"]["ll_refine.ts_opt"].update(status="finished_with_error", job_id=job_id,
                                              scratch_dir=str(scratch), error_log="Output validation failed.")
    failed["rxn_1"] = active.pop("rxn_1")
    (tmp_path / "STATUS.json").write_text(json.dumps(status, indent=4))
    (tmp_path / "rxns.pkl").write_bytes(pickle.dumps(active))
    (tmp_path / "failed_rxns.pkl").write_bytes(pickle.dumps(failed))


class TestFailedAttemptFolder:
    """
    By default the failed attempt's scratch folder is deleted, so the retry
    starts in a fresh one; --keep-failed renames it to `.attempt<N>` instead.
    Either way the attempt (job id, error) is recorded in the task's `attempts`.
    """

    def test_failed_attempt_folder_is_deleted_by_default(self, tmp_path, khp_reaction):
        """Deleted even with a read-only file inside, as container jobs leave behind; no `.attempt1` appears."""
        scratch = _tsopt_failed_run(tmp_path, khp_reaction)

        result = restart_failed(tmp_path, all_failed=True, config=CONFIG, job_manager=_jm())

        assert not scratch.exists()
        assert not scratch.with_name(scratch.name + ".attempt1").exists()
        meta = _load(tmp_path)[0]["reactions"]["rxn_1"]["tasks"]["ll_refine.ts_opt"]
        assert meta["attempt"] == 2 and meta["job_id"] is None and meta["scratch_dir"] is None
        assert "error_log" not in meta
        assert meta["attempts"] == [{"attempt": 1, "status": "finished_with_error", "job_id": "2",
                                     "error_log": "Output validation failed.", "scratch_dir": None,
                                     "scratch_deleted": True}]
        assert result["restarted"][0]["tasks"][0]["scratch_deleted"] == str(scratch)

    def test_read_only_directories_inside_are_deleted_too(self, tmp_path, khp_reaction):
        """A read-only subfolder (its files cannot be unlinked until it is made writable) is still deleted."""
        scratch = _tsopt_failed_run(tmp_path, khp_reaction)
        sub = scratch / "tsopt_run1" / "qm_calcs"
        sub.mkdir(parents=True)
        (sub / "image_000.xtb.out").write_text("x")
        sub.chmod(0o555)
        (scratch / "tsopt_run1").chmod(0o555)

        result = restart_failed(tmp_path, all_failed=True, config=CONFIG, job_manager=_jm())

        assert not scratch.exists() and not scratch.with_name(scratch.name + ".attempt1").exists()
        assert result["restarted"][0]["tasks"][0]["delete_failed"] is False

    def test_keep_failed_renames_the_folder_with_its_contents(self, tmp_path, khp_reaction):
        """--keep-failed: the folder becomes `.attempt1`, untouched, and the attempt records where it went."""
        scratch = _tsopt_failed_run(tmp_path, khp_reaction)

        restart_failed(tmp_path, all_failed=True, keep_failed=True, config=CONFIG, job_manager=_jm())

        kept = scratch.with_name(scratch.name + ".attempt1")
        assert not scratch.exists() and (kept / "run_pysis.sh").exists()
        meta = _load(tmp_path)[0]["reactions"]["rxn_1"]["tasks"]["ll_refine.ts_opt"]
        assert meta["attempt"] == 2 and meta["scratch_dir"] is None
        assert meta["attempts"] == [{"attempt": 1, "status": "finished_with_error", "job_id": "2",
                                     "error_log": "Output validation failed.", "scratch_dir": str(kept)}]

    def test_folder_that_cannot_be_deleted_is_set_aside_with_a_warning(self, tmp_path, khp_reaction, monkeypatch, capsys):
        """If the delete is incomplete, the rest is renamed to `.attempt1` so the retry still starts clean."""
        scratch = _tsopt_failed_run(tmp_path, khp_reaction)
        monkeypatch.setattr(restart_mod, "_delete_folder", lambda path: False)

        result = restart_failed(tmp_path, all_failed=True, config=CONFIG, job_manager=_jm())
        restart_mod._print_report(result)

        kept = scratch.with_name(scratch.name + ".attempt1")
        assert not scratch.exists() and kept.exists()
        meta = _load(tmp_path)[0]["reactions"]["rxn_1"]["tasks"]["ll_refine.ts_opt"]
        assert meta["attempts"][0]["scratch_dir"] == str(kept) and "scratch_deleted" not in meta["attempts"][0]
        assert "WARNING: old scratch could not be fully deleted" in capsys.readouterr().out

    def test_second_restart_deletes_the_second_failure_and_keeps_both_records(self, tmp_path, khp_reaction):
        scratch = _tsopt_failed_run(tmp_path, khp_reaction)
        restart_failed(tmp_path, all_failed=True, config=CONFIG, job_manager=_jm())
        _fail_again(tmp_path, scratch)

        restart_failed(tmp_path, all_failed=True, config=CONFIG, job_manager=_jm())

        assert not scratch.exists() and not list(scratch.parent.glob(scratch.name + ".attempt*"))
        meta = _load(tmp_path)[0]["reactions"]["rxn_1"]["tasks"]["ll_refine.ts_opt"]
        assert meta["attempt"] == 3
        assert [(a["attempt"], a["job_id"], a.get("scratch_deleted")) for a in meta["attempts"]] == [
            (1, "2", True), (2, "3", True)]

    def test_second_restart_with_keep_failed_keeps_attempt2(self, tmp_path, khp_reaction):
        scratch = _tsopt_failed_run(tmp_path, khp_reaction)
        restart_failed(tmp_path, all_failed=True, keep_failed=True, config=CONFIG, job_manager=_jm())
        _fail_again(tmp_path, scratch)

        restart_failed(tmp_path, all_failed=True, keep_failed=True, config=CONFIG, job_manager=_jm())

        a1, a2 = (scratch.with_name(f"{scratch.name}.attempt{n}") for n in (1, 2))
        assert not scratch.exists() and (a1 / "run_pysis.sh").exists() and (a2 / "tsopt.log").exists()
        meta = _load(tmp_path)[0]["reactions"]["rxn_1"]["tasks"]["ll_refine.ts_opt"]
        assert [a["scratch_dir"] for a in meta["attempts"]] == [str(a1), str(a2)]

    def test_cli_passes_keep_failed(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(restart_mod, "restart_failed",
                            lambda *a, **kw: seen.update(kw) or {"restarted": [], "skipped": {}, "dry_run": False})
        for argv, expected in ((["yarp-restart", str(tmp_path), "--all"], False),
                               (["yarp-restart", str(tmp_path), "--all", "--keep-failed"], True)):
            monkeypatch.setattr("sys.argv", argv)
            restart_mod.cli()
            assert seen["keep_failed"] is expected


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
        assert result["restarted"][0]["tasks"][0]["scratch_deleted"] == str(scratch)
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
