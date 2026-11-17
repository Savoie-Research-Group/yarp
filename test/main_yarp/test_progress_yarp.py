# test_progress_yarp.py
import pytest
import json
from unittest.mock import MagicMock, patch
from pathlib import Path
from yarp.progress_yarp import progress_yarp, save_state, _matches_lot
from yarp.util.config import IRCValConfig

@pytest.fixture
def mock_filesystem(mocker):
    """Mocks load_state and save_state to keep everything in memory."""
    mocker.patch('yarp.progress_yarp.Path.mkdir')
    mocker.patch('yarp.progress_yarp.save_state')
    
@pytest.fixture
def mock_calculators(mocker):
    """Mocks the calculator outputs so we don't need real SCRATCH files."""
    calc_mock = MagicMock()
    calc_mock.write_submission_script.return_value = "submit.sh"
    calc_mock.check_output.return_value = True
    mocker.patch('yarp.progress_yarp.get_calculator', return_value=calc_mock)
    return calc_mock

# =====================================================================
# BATCH 1: IDENTIFICATION OF FAILED REACTIONS
# =====================================================================

import json
from unittest.mock import MagicMock
from yarp.progress_yarp import progress_yarp

def test_failure_scenario_1_global_task_failed(tmp_path, mocker):
    """
    Scenario 1: The global task failed on the cluster.
    Validates that the global task transitions to 'finished_with_error'
    and downstream pipeline tasks remain 'pending'.
    """
    status_tracker = {
        "input_config": {},
        "status_output_file": "STATUS.json",
        "reaction_output_file": "YARP_RXNS.pkl",
        "global_tasks": {
            "egat.ml_predict": {
                "status": "submitted",  # Must be submitted so PASS 1.1 checks it
                "job_id": "global_job_123",
                "scratch_dir": None
            }
        },
        "reactions": {
            "rxn_1": {
                "tasks": {
                    "ll_path.reactant_conformer": {
                        "status": "pending",
                        "job_id": None,
                        "scratch_dir": None
                    }
                }
            }
        }
    }
    reactions = {"rxn_1": MagicMock()}

    # 1. Patch load_state to inject our test state
    mocker.patch('yarp.progress_yarp.load_state', return_value=(status_tracker, reactions))

    # 2. Mock InputParser config definitions
    inp_mock = MagicMock()
    inp_mock.job_manager.scheduler = "slurm"
    inp_mock.job_manager.container = "docker"
    inp_mock.job_manager.max_active_jobs = 10
    inp_mock.global_tasks = {"egat.ml_predict": MagicMock()}
    inp_mock.pipeline_tasks = {
        "ll_path.reactant_conformer": MagicMock(depends_on=["egat.ml_predict"])
    }
    mocker.patch('yarp.progress_yarp.InputParser', return_value=inp_mock)

    # 3. Mock Job Manager to report that the cluster job completed
    jm_mock = MagicMock()
    jm_mock.is_running.return_value = False
    mocker.patch('yarp.progress_yarp.get_job_manager', return_value=jm_mock)

    # 4. Mock Calculator to report that output validation failed
    calc_mock = MagicMock()
    calc_mock.check_output.return_value = False
    mocker.patch('yarp.progress_yarp.get_calculator', return_value=calc_mock)

    # 5. Patch pickle operations to prevent serialization errors across mocks
    mocker.patch('yarp.progress_yarp.pickle.dump')
    mocker.patch('yarp.progress_yarp.pickle.load', return_value={})

    # --- Run the execution tick ---
    progress_yarp(tmp_path)

    # Verify global task caught the error
    assert status_tracker["global_tasks"]["egat.ml_predict"]["status"] == "finished_with_error"

    # Verify downstream tasks stay blocked and pending
    assert status_tracker["reactions"]["rxn_1"]["tasks"]["ll_path.reactant_conformer"]["status"] == "pending"


def test_failure_scenario_2_pipeline_task_failed(tmp_path, mocker):
    """
    Scenario 2: A reaction-specific pipeline task fails output validation.
    Validates that the reaction leaves the active pickle but keeps its STATUS
    history (marked failed), and human-readable logs are extracted into
    failed_status.json.
    """
    status_tracker = {
        "input_config": {},
        "status_output_file": "STATUS.json",
        "reaction_output_file": "YARP_RXNS.pkl",
        "reactions": {
            "rxn_1": {
                "tasks": {
                    "ll_path.reactant_conformer": {
                        "status": "submitted",  # Must be submitted so PASS 1.2 checks it
                        "job_id": "pipeline_job_789",
                        "scratch_dir": "/mock/scratch/dir"
                    }
                }
            }
        }
    }
    reactions = {"rxn_1": MagicMock()}

    # 1. Patch load_state to inject state
    mocker.patch('yarp.progress_yarp.load_state', return_value=(status_tracker, reactions))

    # 2. Mock InputParser config definitions
    inp_mock = MagicMock()
    inp_mock.job_manager.scheduler = "slurm"
    inp_mock.job_manager.container = "docker"
    inp_mock.job_manager.max_active_jobs = 10
    inp_mock.pipeline_tasks = {
        "ll_path.reactant_conformer": MagicMock(task_type="reactant_conformer")
    }
    mocker.patch('yarp.progress_yarp.InputParser', return_value=inp_mock)

    # 3. Mock Job Manager to report the job finished running
    jm_mock = MagicMock()
    jm_mock.is_running.return_value = False
    mocker.patch('yarp.progress_yarp.get_job_manager', return_value=jm_mock)

    # 4. Mock Calculator to simulate failing output validation
    calc_mock = MagicMock()
    calc_mock.check_output.return_value = False
    mocker.patch('yarp.progress_yarp.get_calculator', return_value=calc_mock)

    # 5. Patch pickle operations
    mocker.patch('yarp.progress_yarp.pickle.dump')
    mocker.patch('yarp.progress_yarp.pickle.load', return_value={})

    # --- Run the execution tick ---
    progress_yarp(tmp_path)

    # The reaction object leaves the active pool; its STATUS entry stays, marked
    # failed, with the failed task's history, so it can be restarted later.
    assert "rxn_1" not in reactions
    assert status_tracker["reactions"]["rxn_1"]["failed"] is True
    assert status_tracker["reactions"]["rxn_1"]["tasks"]["ll_path.reactant_conformer"]["status"] == "finished_with_error"

    # Verify that human-readable error reasons were successfully written out
    fail_json_file = tmp_path / "failed_status.json"
    assert fail_json_file.exists()

    with open(fail_json_file, "r") as f:
        failed_status = json.load(f)
        assert "rxn_1" in failed_status
        assert failed_status["rxn_1"]["ll_path.reactant_conformer"]["error_log"] == "Output validation failed."
        assert failed_status["rxn_1"]["ll_path.reactant_conformer"]["scratch_dir"] == "/mock/scratch/dir"


def test_failure_scenario_3_irc_unintended(tmp_path, mocker):
    """
    Scenario 3: A reaction completes tasks, but IRC filters it out as "unintended",
    because the user opted in with `divert_unintended: true`.
    """
    status_tracker = {
        "input_config": {},
        "status_output_file": "STATUS.json",
        "reaction_output_file": "YARP_RXNS.pkl",
        "reactions": {
            "rxn_unintended": {"tasks": {"stage1.irc": {"status": "submitted", "job_id": "201", "scratch_dir": "/tmp/scratch"}}},
            "rxn_intended": {"tasks": {"stage1.irc": {"status": "submitted", "job_id": "202", "scratch_dir": "/tmp/scratch"}}}
        }
    }
    rxn_unintended = MagicMock()
    rxn_unintended.outcome_label = {"b3lyp_orca": "unintended"}
    rxn_intended = MagicMock()
    rxn_intended.outcome_label = {"b3lyp_orca": "intended"}
    reactions = {"rxn_unintended": rxn_unintended, "rxn_intended": rxn_intended}

    mocker.patch('yarp.progress_yarp.load_state', return_value=(status_tracker, reactions))

    jm_mock = MagicMock()
    jm_mock.is_running.return_value = False
    mocker.patch('yarp.progress_yarp.get_job_manager', return_value=jm_mock)

    calc_mock = MagicMock()
    calc_mock.check_output.return_value = True
    calc_mock.scrape_data.return_value = True
    mocker.patch('yarp.progress_yarp.get_calculator', return_value=calc_mock)

    inp_mock = MagicMock()
    inp_mock.job_manager.scheduler = "slurm"
    inp_mock.job_manager.container = "docker"
    inp_mock.job_manager.max_active_jobs = 10
    inp_mock.global_tasks = {}

    # A real config, not a MagicMock: a mocked `divert_unintended` would be a
    # truthy MagicMock and divert regardless of what the code does with it.
    task_mock = MagicMock(task_type="irc_validation", parent_stage="stage1", depends_on=[])
    task_mock.config = IRCValConfig(software="orca", lot="b3lyp", charge=0, multiplicity=1,
                                    divert_unintended=True)
    inp_mock.pipeline_tasks = {"stage1.irc": task_mock}
    mocker.patch('yarp.progress_yarp.InputParser', return_value=inp_mock)

    # Patch pickle operations to prevent MagicMock pickling errors
    mocker.patch('yarp.progress_yarp.pickle.dump')
    mocker.patch('yarp.progress_yarp.pickle.load', return_value={})

    progress_yarp(tmp_path)

    assert "rxn_unintended" not in reactions
    assert status_tracker["reactions"]["rxn_unintended"]["failed"] is True

    assert "rxn_intended" in status_tracker["reactions"]
    assert "rxn_intended" in reactions
    assert status_tracker["reactions"]["rxn_intended"]["tasks"]["stage1.irc"]["status"] == "terminated_normally"

    assert (tmp_path / "failed_rxns.pkl").exists()

    # Verify that human-readable error reasons were successfully written out
    fail_json_file = tmp_path / "failed_status.json"
    assert fail_json_file.exists()

    with open(fail_json_file, "r") as f:
        failed_status = json.load(f)
        assert "rxn_unintended" in failed_status
        assert failed_status["rxn_unintended"]["stage1.irc"]["error_log"] == "IRC validation failed: Outcome was 'unintended'."
        assert failed_status["rxn_unintended"]["stage1.irc"]["scratch_dir"] == "/tmp/scratch"


NON_INTENDED_OUTCOMES = ["unintended", "reactant_unintended", "product_unintended", "no_adjmat_change", None]


class TestIRCDivertFlag:
    """
    `irc_val: divert_unintended` decides whether a reaction whose IRC outcome is
    anything but intended/inverse_intended is routed to failed_rxns. It defaults
    to off: xTB IRC labels are poor predictors of DFT IRC outcomes, so the
    reaction should carry on to the next refinement stage.
    """

    def _run(self, tmp_path, mocker, outcome, **irc_kwargs):
        """One tick: stage1.irc has just finished with `outcome`; stage2.tsopt waits on it."""
        status_tracker = {
            "input_config": {},
            "status_output_file": "STATUS.json",
            "reaction_output_file": "YARP_RXNS.pkl",
            "reactions": {
                "rxn_1": {"tasks": {
                    "stage1.irc": {"status": "submitted", "job_id": "201", "scratch_dir": "/tmp/scratch"},
                    "stage2.tsopt": {"status": "pending", "job_id": None, "scratch_dir": None},
                }},
            },
        }
        rxn = MagicMock()
        rxn.outcome_label = {"xtb_pysisyphus": outcome}
        reactions = {"rxn_1": rxn}
        mocker.patch('yarp.progress_yarp.load_state', return_value=(status_tracker, reactions))

        jm_mock = MagicMock()
        jm_mock.is_running.return_value = False
        jm_mock.submit.return_value = "301"
        mocker.patch('yarp.progress_yarp.get_job_manager', return_value=jm_mock)

        calc_mock = MagicMock()
        calc_mock.check_output.return_value = True
        calc_mock.scrape_data.return_value = True
        calc_mock.has_prerequisites.return_value = True
        mocker.patch('yarp.progress_yarp.get_calculator', return_value=calc_mock)

        inp_mock = MagicMock()
        inp_mock.job_manager.scheduler = "local"
        inp_mock.job_manager.container = "docker"
        inp_mock.job_manager.max_active_jobs = 10
        inp_mock.global_tasks = {}
        irc_task = MagicMock(task_type="irc_validation", parent_stage="stage1", depends_on=[])
        irc_task.config = IRCValConfig(software="pysisyphus", lot="xtb", charge=0, multiplicity=1,
                                       **irc_kwargs)
        tsopt_task = MagicMock(task_type="transition_state_optimization", parent_stage="stage2",
                               depends_on=["stage1.irc"])
        inp_mock.pipeline_tasks = {"stage1.irc": irc_task, "stage2.tsopt": tsopt_task}
        mocker.patch('yarp.progress_yarp.InputParser', return_value=inp_mock)

        mocker.patch('yarp.progress_yarp.pickle.dump')
        mocker.patch('yarp.progress_yarp.pickle.load', return_value={})

        progress_yarp(tmp_path)
        return status_tracker, reactions

    def test_default_is_off(self):
        assert IRCValConfig(software="pysisyphus", lot="xtb", charge=0, multiplicity=1).divert_unintended is False

    @pytest.mark.parametrize("outcome", NON_INTENDED_OUTCOMES)
    def test_flag_omitted_keeps_reaction_and_unblocks_next_stage(self, tmp_path, mocker, outcome):
        status_tracker, reactions = self._run(tmp_path, mocker, outcome)

        assert "rxn_1" in reactions
        tasks = status_tracker["reactions"]["rxn_1"]["tasks"]
        assert tasks["stage1.irc"]["status"] == "terminated_normally"
        assert tasks["stage2.tsopt"]["status"] == "submitted"
        assert not (tmp_path / "failed_rxns.pkl").exists()

    @pytest.mark.parametrize("outcome", NON_INTENDED_OUTCOMES)
    def test_flag_false_keeps_reaction(self, tmp_path, mocker, outcome):
        status_tracker, reactions = self._run(tmp_path, mocker, outcome, divert_unintended=False)

        assert "rxn_1" in reactions
        assert status_tracker["reactions"]["rxn_1"]["tasks"]["stage1.irc"]["status"] == "terminated_normally"

    @pytest.mark.parametrize("outcome", NON_INTENDED_OUTCOMES)
    def test_flag_true_diverts_reaction(self, tmp_path, mocker, outcome):
        status_tracker, reactions = self._run(tmp_path, mocker, outcome, divert_unintended=True)

        assert "rxn_1" not in reactions
        assert status_tracker["reactions"]["rxn_1"]["failed"] is True
        assert (tmp_path / "failed_rxns.pkl").exists()

    @pytest.mark.parametrize("outcome", ["intended", "inverse_intended"])
    @pytest.mark.parametrize("divert", [True, False])
    def test_intended_always_kept(self, tmp_path, mocker, outcome, divert):
        status_tracker, reactions = self._run(tmp_path, mocker, outcome, divert_unintended=divert)

        assert "rxn_1" in reactions
        assert status_tracker["reactions"]["rxn_1"]["tasks"]["stage2.tsopt"]["status"] == "submitted"


class TestIRCFastForward:
    """
    PASS 0.2 skips an IRC task whose results are already on the reaction (e.g.
    a reaction carried over from an earlier run). IRC no longer writes a
    validated_ts entry, so the barrier keys alone must be enough -- including a
    None barrier, which is what a reaction with no intended TS carries.
    """

    def _run(self, tmp_path, mocker, barrier, reverse_barrier):
        status_tracker = {
            "input_config": {},
            "status_output_file": "STATUS.json",
            "reaction_output_file": "YARP_RXNS.pkl",
            "reactions": {"rxn_1": {"tasks": {
                "stage1.irc": {"status": "pending", "job_id": None, "scratch_dir": None},
            }}},
        }
        rxn = MagicMock()
        rxn.ts_geom = {}
        rxn.barrier = barrier
        rxn.reverse_barrier = reverse_barrier
        mocker.patch('yarp.progress_yarp.load_state', return_value=(status_tracker, {"rxn_1": rxn}))
        mocker.patch('yarp.progress_yarp.get_job_manager', return_value=MagicMock())
        mocker.patch('yarp.progress_yarp.get_calculator', return_value=MagicMock())

        inp_mock = MagicMock()
        inp_mock.job_manager.max_active_jobs = 0  # submit nothing: only PASS 0.2 matters here
        inp_mock.global_tasks = {}
        irc_task = MagicMock(task_type="irc_validation", parent_stage="stage1", depends_on=[])
        irc_task.config = IRCValConfig(software="pysisyphus", lot="xtb", charge=0, multiplicity=1)
        inp_mock.pipeline_tasks = {"stage1.irc": irc_task}
        mocker.patch('yarp.progress_yarp.InputParser', return_value=inp_mock)
        mocker.patch('yarp.progress_yarp.pickle.dump')

        progress_yarp(tmp_path)
        return status_tracker["reactions"]["rxn_1"]["tasks"]["stage1.irc"]["status"]

    @pytest.mark.parametrize("value", [12.3, None])
    def test_barrier_keys_are_enough(self, tmp_path, mocker, value):
        status = self._run(tmp_path, mocker, {"xtb_pysisyphus": value}, {"xtb_pysisyphus": value})

        assert status == "terminated_normally"

    def test_not_skipped_without_results(self, tmp_path, mocker):
        status = self._run(tmp_path, mocker, {"egat_rgd1": 50.0}, {"egat_rgd1": 60.0})

        assert status != "terminated_normally"


class TestMatchesLot:
    """
    Fast-forward finds a stage's stored results by level of theory. Since
    "xtb_pysisyphus" is a substring of "gxtb_pysisyphus", check that xtb and
    gxtb results never stand in for each other, for every stage that can run
    at either level.
    """

    # The key each stage stores its result under; {lot} is xtb or gxtb.
    STAGE_KEYS = {
        "pre_opt": "preopt_{lot}_pysisyphus",
        "ts_guess": "ts_guess_0_{lot}_pysisyphus",
        "rp_opt": "rpopt_{lot}_pysisyphus",
        "ts_opt": "0_tsopt_{lot}_pysisyphus",
        "irc_barrier": "{lot}_pysisyphus",
    }

    @pytest.mark.parametrize("stage", STAGE_KEYS)
    def test_xtb_and_gxtb_do_not_overlap(self, stage):
        xtb_key = self.STAGE_KEYS[stage].format(lot="xtb")
        gxtb_key = self.STAGE_KEYS[stage].format(lot="gxtb")

        assert _matches_lot(xtb_key, "xtb_pysisyphus")
        assert _matches_lot(gxtb_key, "gxtb_pysisyphus")
        assert not _matches_lot(gxtb_key, "xtb_pysisyphus")
        assert not _matches_lot(xtb_key, "gxtb_pysisyphus")


# =====================================================================
# BATCH 2: PIPELINE PROGRESSION & DAG HAND-OFFS
# =====================================================================

def test_handoff_scenario_1_global_to_pipeline(tmp_path, mock_calculators, mocker):
    """
    Scenario 1: Global task (ml_rxn_prop) -> Pipeline task (init_rxn_path).
    Validates that downstream pipeline remains pending while global is active,
    and upgrades to 'ready' once the global task marks 'terminated_normally'.
    """
    status_tracker = {
        "input_config": {},
        "status_output_file": "STATUS.json",
        "reaction_output_file": "YARP_RXNS.pkl",
        "pipeline_tasks": [
            "ll_path.reactant_conformer",
            "ll_path.product_conformer",
            "ll_path.ts_guess"
        ],
        "global_tasks_list": [
            "egat.ml_predict"
        ],
        "global_tasks": {
            "egat.ml_predict": {
                "status": "ready",
                "job_id": None,
                "scratch_dir": None
            }
        },
        "reactions": {
            "rxn_1": {
                "tasks": {
                    "ll_path.reactant_conformer": {
                        "status": "pending",
                        "job_id": None,
                        "scratch_dir": None
                    },
                    "ll_path.product_conformer": {
                        "status": "pending",
                        "job_id": None,
                        "scratch_dir": None
                    },
                    "ll_path.ts_guess": {
                        "status": "pending",
                        "job_id": None,
                        "scratch_dir": None
                    }
                }
            }
        }
    }
    reactions = {"rxn_1": MagicMock()}
    mocker.patch('yarp.progress_yarp.load_state', return_value=(status_tracker, reactions))

    jm_mock = MagicMock()
    jm_mock.submit.return_value = "12345"
    jm_mock.is_running.return_value = True # Global job is still busy on the cluster
    mocker.patch('yarp.progress_yarp.get_job_manager', return_value=jm_mock)

    inp_mock = MagicMock()
    inp_mock.job_manager.scheduler = "slurm"
    inp_mock.job_manager.container = "docker"
    inp_mock.job_manager.max_active_jobs = 10
    inp_mock.global_tasks = {"egat.ml_predict": MagicMock(depends_on=[])}
    inp_mock.pipeline_tasks = {
        "ll_path.reactant_conformer": MagicMock(task_type="reactant_conformer", parent_stage="ll_path", depends_on=["egat.ml_predict"]),
        "ll_path.product_conformer": MagicMock(task_type="product_conformer", parent_stage="ll_path", depends_on=["egat.ml_predict"]),
        "ll_path.ts_guess": MagicMock(task_type="ts_guess", parent_stage="ll_path", depends_on=["ll_path.reactant_conformer", "ll_path.product_conformer"])
    }
    inp_mock.stage_filters = {}
    mocker.patch('yarp.progress_yarp.InputParser', return_value=inp_mock)

    # Patch pickle operations to prevent MagicMock pickling errors across both ticks
    mocker.patch('yarp.progress_yarp.pickle.dump')
    mocker.patch('yarp.progress_yarp.pickle.load', return_value={})

    # --- TICK 1: Global Task is still active ---
    progress_yarp(tmp_path)
    assert status_tracker["global_tasks"]["egat.ml_predict"]["status"] == "submitted"
    assert status_tracker["reactions"]["rxn_1"]["tasks"]["ll_path.reactant_conformer"]["status"] == "pending"
    assert status_tracker["reactions"]["rxn_1"]["tasks"]["ll_path.product_conformer"]["status"] == "pending"
    assert status_tracker["reactions"]["rxn_1"]["tasks"]["ll_path.ts_guess"]["status"] == "pending"

    # --- TICK 2: Global Task finishes ---
    jm_mock.is_running.return_value = False
    mock_calculators.check_output.return_value = True
    mock_calculators.scrape_data.return_value = True

    # Lock max active submissions to 0 so we catch it in the "ready" state before Pass 3 submits it
    inp_mock.job_manager.max_active_jobs = 0

    progress_yarp(tmp_path)
    assert status_tracker["global_tasks"]["egat.ml_predict"]["status"] == "terminated_normally"
    assert status_tracker["reactions"]["rxn_1"]["tasks"]["ll_path.reactant_conformer"]["status"] == "ready"
    assert status_tracker["reactions"]["rxn_1"]["tasks"]["ll_path.product_conformer"]["status"] == "ready"
    assert status_tracker["reactions"]["rxn_1"]["tasks"]["ll_path.ts_guess"]["status"] == "pending"


def test_handoff_scenario_2_init_rxn_path_only(tmp_path, mocker):
    """
    Scenario 2: Single stage initialization & execution across multiple reactions.
    Ensures only reactions that satisfy prerequisites advance, leaving busy ones intact.
    """
    status_tracker = {
        "input_config": {},
        "status_output_file": "STATUS.json",
        "reaction_output_file": "YARP_RXNS.pkl",
        "pipeline_tasks": [
            "ll_path.reactant_conformer",
            "ll_path.product_conformer",
            "ll_path.ts_guess"
        ],
        "reactions": {
            "rxn_ready": {
                "tasks": {
                    "ll_path.reactant_conformer": {"status": "ready", "job_id": None, "scratch_dir": None},
                    "ll_path.product_conformer": {"status": "ready", "job_id": None, "scratch_dir": None},
                    "ll_path.ts_guess": {"status": "pending", "job_id": None, "scratch_dir": None}
                }
            },
            "rxn_waiting": {
                "tasks": {
                    "ll_path.reactant_conformer": {"status": "submitted", "job_id": "888", "scratch_dir": "/tmp/scratch"},
                    "ll_path.product_conformer": {"status": "submitted", "job_id": "888", "scratch_dir": "/tmp/scratch"},
                    "ll_path.ts_guess": {"status": "pending", "job_id": None, "scratch_dir": None}
                }
            }
        }
    }
    reactions = {"rxn_ready": MagicMock(), "rxn_waiting": MagicMock()}
    mocker.patch('yarp.progress_yarp.load_state', return_value=(status_tracker, reactions))

    jm_mock = MagicMock()
    jm_mock.submit.return_value = "888"
    jm_mock.is_running.side_effect = lambda job_id: job_id == "888" # Keep rxn_waiting job running
    mocker.patch('yarp.progress_yarp.get_job_manager', return_value=jm_mock)

    calc_mock = MagicMock()
    calc_mock.write_submission_script.return_value = "submit.sh"
    calc_mock.has_prerequisites.return_value = True
    mocker.patch('yarp.progress_yarp.get_calculator', return_value=calc_mock)

    inp_mock = MagicMock()
    inp_mock.job_manager.scheduler = "slurm"
    inp_mock.job_manager.container = "docker"
    inp_mock.job_manager.max_active_jobs = 10
    inp_mock.global_tasks = {}
    inp_mock.pipeline_tasks = {
        "ll_path.reactant_conformer": MagicMock(task_type="reactant_conformer", parent_stage="ll_path", depends_on=[]),
        "ll_path.product_conformer": MagicMock(task_type="product_conformer", parent_stage="ll_path", depends_on=[]),
        "ll_path.ts_guess": MagicMock(task_type="ts_guess", parent_stage="ll_path", depends_on=["ll_path.reactant_conformer", "ll_path.product_conformer"])
    }
    mocker.patch('yarp.progress_yarp.InputParser', return_value=inp_mock)

    # Patch pickle operations to prevent MagicMock pickling errors across both ticks
    mocker.patch('yarp.progress_yarp.pickle.dump')
    mocker.patch('yarp.progress_yarp.pickle.load', return_value={})

    # --- TICK 1: Initial Launch ---
    progress_yarp(tmp_path)

    assert status_tracker["reactions"]["rxn_ready"]["tasks"]["ll_path.reactant_conformer"]["status"] == "submitted"
    assert status_tracker["reactions"]["rxn_ready"]["tasks"]["ll_path.product_conformer"]["status"] == "submitted"
    assert status_tracker["reactions"]["rxn_ready"]["tasks"]["ll_path.ts_guess"]["status"] == "pending"
    assert status_tracker["reactions"]["rxn_waiting"]["tasks"]["ll_path.reactant_conformer"]["status"] == "submitted"
    assert status_tracker["reactions"]["rxn_waiting"]["tasks"]["ll_path.product_conformer"]["status"] == "submitted"
    assert status_tracker["reactions"]["rxn_waiting"]["tasks"]["ll_path.ts_guess"]["status"] == "pending"

    # --- TICK 2: Progress Only the Finished Reaction ---
    # Manually pass rxn_ready.conf to simulate completion on a subsequent tick loop
    status_tracker["reactions"]["rxn_ready"]["tasks"]["ll_path.reactant_conformer"]["status"] = "terminated_normally"
    status_tracker["reactions"]["rxn_ready"]["tasks"]["ll_path.product_conformer"]["status"] = "terminated_normally"

    progress_yarp(tmp_path)

    # rxn_ready should kick forward to submit ts_guess
    assert status_tracker["reactions"]["rxn_ready"]["tasks"]["ll_path.ts_guess"]["status"] == "submitted"
    # rxn_waiting must still remain pending since its parent conf calculation hasn't dropped out of the scheduler
    assert status_tracker["reactions"]["rxn_waiting"]["tasks"]["ll_path.ts_guess"]["status"] == "pending"


def test_handoff_scenario_3_init_to_refine_pipeline(tmp_path, mocker):
    """
    Scenario 3: init_rxn_path followed by refine_rxn_path.
    Tests branch dependencies: rp_opt submits for both (since both finished conf_gen), 
    but ts_opt only fires for the reaction that finished ts_guess.
    """
    status_tracker = {
        "input_config": {},
        "status_output_file": "STATUS.json",
        "reaction_output_file": "YARP_RXNS.pkl",
        "pipeline_tasks": [
            "ll_path.reactant_conformer",
            "ll_path.product_conformer",
            "ll_path.ts_guess",
            "ll_refine.reactant_optimization",
            "ll_refine.product_optimization",
            "ll_refine.transition_state_optimization",
            "ll_refine.irc_validation"
        ],
        "reactions": {
            "rxn_partial": {
                "tasks": {
                    "ll_path.reactant_conformer": {"status": "terminated_normally", "job_id": None, "scratch_dir": None},
                    "ll_path.product_conformer": {"status": "terminated_normally", "job_id": None, "scratch_dir": None},
                    "ll_path.ts_guess": {"status": "submitted", "job_id": "555", "scratch_dir": None},
                    "ll_refine.reactant_optimization": {"status": "pending", "job_id": None, "scratch_dir": None},
                    "ll_refine.product_optimization": {"status": "pending", "job_id": None, "scratch_dir": None},
                    "ll_refine.transition_state_optimization": {"status": "pending", "job_id": None, "scratch_dir": None},
                    "ll_refine.irc_validation": {"status": "pending", "job_id": None, "scratch_dir": None}
                }
            },
            "rxn_complete": {
                "tasks": {
                    "ll_path.reactant_conformer": {"status": "terminated_normally", "job_id": None, "scratch_dir": None},
                    "ll_path.product_conformer": {"status": "terminated_normally", "job_id": None, "scratch_dir": None},
                    "ll_path.ts_guess": {"status": "terminated_normally", "job_id": None, "scratch_dir": None},
                    "ll_refine.reactant_optimization": {"status": "pending", "job_id": None, "scratch_dir": None},
                    "ll_refine.product_optimization": {"status": "pending", "job_id": None, "scratch_dir": None},
                    "ll_refine.transition_state_optimization": {"status": "pending", "job_id": None, "scratch_dir": None},
                    "ll_refine.irc_validation": {"status": "pending", "job_id": None, "scratch_dir": None}
                }
            }
        }
    }
    reactions = {"rxn_partial": MagicMock(), "rxn_complete": MagicMock()}
    mocker.patch('yarp.progress_yarp.load_state', return_value=(status_tracker, reactions))

    jm_mock = MagicMock()
    jm_mock.submit.return_value = "555"
    jm_mock.is_running.return_value = True
    mocker.patch('yarp.progress_yarp.get_job_manager', return_value=jm_mock)

    calc_mock = MagicMock()
    calc_mock.write_submission_script.return_value = "submit.sh"
    calc_mock.has_prerequisites.return_value = True
    mocker.patch('yarp.progress_yarp.get_calculator', return_value=calc_mock)

    inp_mock = MagicMock()
    inp_mock.job_manager.scheduler = "slurm"
    inp_mock.job_manager.container = "docker"
    inp_mock.job_manager.max_active_jobs = 10
    inp_mock.global_tasks = {}
    inp_mock.pipeline_tasks = {
        "ll_path.reactant_conformer": MagicMock(task_type="reactant_conformer", parent_stage="ll_path", depends_on=[]),
        "ll_path.product_conformer": MagicMock(task_type="product_conformer", parent_stage="ll_path", depends_on=[]),
        "ll_path.ts_guess": MagicMock(task_type="ts_guess", parent_stage="ll_path", depends_on=["ll_path.reactant_conformer", "ll_path.product_conformer"]),
        "ll_refine.reactant_optimization": MagicMock(task_type="reactant_optimization", parent_stage="ll_refine", depends_on=["ll_path.reactant_conformer"]),
        "ll_refine.product_optimization": MagicMock(task_type="product_optimization", parent_stage="ll_refine", depends_on=["ll_path.product_conformer"]),
        "ll_refine.transition_state_optimization": MagicMock(task_type="transition_state_optimization", parent_stage="ll_refine", depends_on=["ll_path.ts_guess"]),
        "ll_refine.irc_validation": MagicMock(task_type="irc", parent_stage="ll_refine", depends_on=["ll_refine.transition_state_optimization"])
    }
    mocker.patch('yarp.progress_yarp.InputParser', return_value=inp_mock)

    # Patch pickle operations to prevent MagicMock pickling errors
    mocker.patch('yarp.progress_yarp.pickle.dump')
    mocker.patch('yarp.progress_yarp.pickle.load', return_value={})

    progress_yarp(tmp_path)

    # Assertions for rxn_partial (only conf branch satisfies dependencies)
    assert status_tracker["reactions"]["rxn_partial"]["tasks"]["ll_refine.reactant_optimization"]["status"] == "submitted"
    assert status_tracker["reactions"]["rxn_partial"]["tasks"]["ll_refine.product_optimization"]["status"] == "submitted"
    assert status_tracker["reactions"]["rxn_partial"]["tasks"]["ll_refine.transition_state_optimization"]["status"] == "pending"

    # Assertions for rxn_complete (both branches satisfy dependencies)
    assert status_tracker["reactions"]["rxn_complete"]["tasks"]["ll_refine.reactant_optimization"]["status"] == "submitted"
    assert status_tracker["reactions"]["rxn_complete"]["tasks"]["ll_refine.product_optimization"]["status"] == "submitted"
    assert status_tracker["reactions"]["rxn_complete"]["tasks"]["ll_refine.transition_state_optimization"]["status"] == "submitted"

# =====================================================================
# BATCH 4: PRE-OPTIMIZATION KEYING
# =====================================================================
from types import SimpleNamespace

import numpy as np

from yarp.progress_yarp import species_registry_key, shareable_conformers, synchronize_conformers


def _state(identity):
    """A stand-in state carrying just the attributes the keying logic reads."""
    return SimpleNamespace(identity=identity, conformers={})


def _rxn(reactant_identity, product_identity):
    return SimpleNamespace(reactant=_state(reactant_identity), product=_state(product_identity))


class TestSpeciesRegistryKey:
    """
    The redundancy blocker lets two reactions share one single-species job.
    That is sound for every task whose result depends only on the species it
    acts on -- and unsound for the product pre-optimization, whose starting
    structure is a UFF patch of *this reaction's* relaxed reactant.
    """

    @pytest.mark.parametrize("task_type", [
        "reactant_conformer", "reactant_optimization", "reactant_pre_opt",
    ])
    def test_reactant_tasks_key_on_the_reactant_alone(self, task_type):
        rxn = _rxn(("rhash", 1.0), ("phash", 2.0))

        assert species_registry_key(rxn, task_type, "s.t") == (("rhash", 1.0), "s.t")

    @pytest.mark.parametrize("task_type", ["product_conformer", "product_optimization"])
    def test_most_product_tasks_key_on_the_product_alone(self, task_type):
        rxn = _rxn(("rhash", 1.0), ("phash", 2.0))

        assert species_registry_key(rxn, task_type, "s.t") == (("phash", 2.0), "s.t")

    def test_product_pre_opt_also_keys_on_the_reactant(self):
        rxn = _rxn(("rhash", 1.0), ("phash", 2.0))

        assert species_registry_key(rxn, "product_pre_opt", "s.t") == (
            ("rhash", 1.0), ("phash", 2.0), "s.t",
        )

    def test_same_product_different_reactants_are_not_deduplicated(self):
        """
        The bug this guards: two reactions reaching the same product from
        different reactants produce genuinely different patched geometries, so
        one must not claim the other's job.
        """
        a = _rxn(("reactant_A", 1.0), ("shared_product", 9.0))
        b = _rxn(("reactant_B", 2.0), ("shared_product", 9.0))

        assert species_registry_key(a, "product_pre_opt", "s.t") != \
               species_registry_key(b, "product_pre_opt", "s.t")

    def test_same_product_same_reactant_is_deduplicated(self):
        """Genuine duplicates must still collapse, or dedup buys us nothing."""
        a = _rxn(("reactant_A", 1.0), ("shared_product", 9.0))
        b = _rxn(("reactant_A", 1.0), ("shared_product", 9.0))

        assert species_registry_key(a, "product_pre_opt", "s.t") == \
               species_registry_key(b, "product_pre_opt", "s.t")

    @pytest.mark.parametrize("task_type", ["ts_guess", "irc_validation", "ml_predict"])
    def test_reaction_path_tasks_are_not_deduplicated(self, task_type):
        rxn = _rxn(("rhash", 1.0), ("phash", 2.0))

        assert species_registry_key(rxn, task_type, "s.t") is None


class TestShareableConformers:
    """
    PASS 0.1 pools conformers across states of the same identity. Both pre-opt
    legs write the same conformer key, but the product's is reaction-specific,
    so pooling it would hand one reaction's structure to another.
    """

    def test_reactant_shares_everything_including_its_pre_opt(self):
        species = _state(("h", 1.0))
        species.conformers = {"initial_geom": "a", "preopt_xtb_pysisyphus": "b",
                              "conf_gen_rank0_gfn2_crest": "c"}

        assert set(shareable_conformers(species, "reactant")) == set(species.conformers)

    def test_product_withholds_its_pre_opt(self):
        species = _state(("h", 1.0))
        species.conformers = {"initial_geom": "a", "preopt_xtb_pysisyphus": "b",
                              "conf_gen_rank0_gfn2_crest": "c"}

        shared = shareable_conformers(species, "product")

        assert "preopt_xtb_pysisyphus" not in shared
        assert set(shared) == {"initial_geom", "conf_gen_rank0_gfn2_crest"}

    def test_conformer_generation_results_still_pool(self):
        """Withholding the pre-opt must not break CREST deduplication."""
        species = _state(("h", 1.0))
        species.conformers = {"conf_gen_rank0_gfn2_crest": "c", "rpopt_xtb_pysisyphus": "d"}

        assert set(shareable_conformers(species, "product")) == set(species.conformers)

    def test_returns_a_copy_not_the_live_dict(self):
        species = _state(("h", 1.0))
        species.conformers = {"initial_geom": "a"}

        shareable_conformers(species, "reactant")["injected"] = "x"

        assert "injected" not in species.conformers


class TestSynchronizeWritesReactantGeometry:
    """
    One reactant pre-opt job serves every reaction sharing that reactant; the
    others receive its conformer only through PASS 0.1. Their graph geometry
    has to be written back there, or only the reaction that ran the job would
    carry relaxed coordinates.
    """

    def _two_reactions_sharing_a_reactant(self, khp_parent, khp_products):
        from yarp.reaction.reaction import reaction
        a = reaction(khp_parent, khp_products["CCC(=O)OO"])
        b = reaction(khp_parent, khp_products["C=COCOO"])
        assert a.reactant.identity == b.reactant.identity
        return a, b

    def _preopt(self, graph, shift):
        from yarp.reaction.conformer import conformer
        conf = conformer()
        conf.elements = graph.elements
        conf.geo = graph.geo + shift
        conf.properties["internal_energy_Eh"] = -20.0
        return conf

    def test_pooled_reactant_takes_the_shared_pre_opt(self, khp_parent, khp_products):
        a, b = self._two_reactions_sharing_a_reactant(khp_parent, khp_products)
        a.reactant.conformers["preopt_xtb_pysisyphus"] = self._preopt(a.reactant.graph, 1.0)

        synchronize_conformers({"a": a, "b": b})

        assert np.allclose(b.reactant.graph.geo, a.reactant.conformers["preopt_xtb_pysisyphus"].geo)

    def test_product_pre_opt_is_not_written_back_here(self, khp_parent, khp_products):
        """A product's pre-opt may be off its graph; only scrape_data decides what to store."""
        a, b = self._two_reactions_sharing_a_reactant(khp_parent, khp_products)
        before = a.product.graph.geo.copy()
        a.product.conformers["preopt_xtb_pysisyphus"] = self._preopt(a.product.graph, 1.0)

        synchronize_conformers({"a": a, "b": b})

        assert np.array_equal(a.product.graph.geo, before)


# =====================================================================
# FAILED REACTIONS KEEP THEIR HISTORY IN STATUS.json
# =====================================================================

class TestFailedReactionsKeepHistory:
    """
    A failed reaction's data object still leaves the active pickle (it goes to
    failed_rxns.pkl), but its STATUS.json entry now stays, marked
    `"failed": true`, with every task's status, so a later restart can rebuild
    it. Every pass must then skip it: a failed reaction's leftover pending or
    submitted tasks must not hold up or crash the rest of the run.
    """

    @staticmethod
    def _task(task_type, depends_on=(), parent_stage="ll_path"):
        return MagicMock(task_type=task_type, depends_on=list(depends_on), parent_stage=parent_stage)

    @staticmethod
    def _run(tmp_path, mocker, status_tracker, reactions, pipeline_tasks,
             global_tasks=None, stage_filters=None, check_output=True):
        """One progress pass, with the scheduler, calculators and pickles mocked."""
        mocker.patch('yarp.progress_yarp.load_state', return_value=(status_tracker, reactions))
        inp = MagicMock()
        inp.job_manager.scheduler = "sge"
        inp.job_manager.container = "apptainer"
        inp.job_manager.max_active_jobs = 10
        inp.pipeline_tasks = pipeline_tasks
        inp.global_tasks = global_tasks or {}
        inp.stage_filters = stage_filters or {}
        mocker.patch('yarp.progress_yarp.InputParser', return_value=inp)
        jm = MagicMock()
        jm.is_running.return_value = False
        jm.submit.return_value = "job_new"
        mocker.patch('yarp.progress_yarp.get_job_manager', return_value=jm)
        calc = MagicMock()
        calc.check_output.return_value = check_output
        calc.scrape_data.return_value = True
        calc.has_prerequisites.return_value = True
        mocker.patch('yarp.progress_yarp.get_calculator', return_value=calc)
        mocker.patch('yarp.progress_yarp.pickle.dump')
        mocker.patch('yarp.progress_yarp.pickle.load', return_value={})
        progress_yarp(tmp_path)
        return jm

    @staticmethod
    def _status(reactions, global_tasks=None):
        return {"input_config": {}, "status_output_file": "STATUS.json",
                "reaction_output_file": "YARP_RXNS.pkl",
                "global_tasks": global_tasks or {}, "reactions": reactions}

    @staticmethod
    def _t(status, job_id=None, scratch_dir=None):
        return {"status": status, "job_id": job_id, "scratch_dir": scratch_dir}

    def test_full_task_history_is_kept_when_a_task_fails(self, tmp_path, mocker):
        """
        A reaction whose TS optimization fails: its earlier finished tasks, the
        failed task (with error and scratch dir) and its still-pending IRC all
        stay in STATUS.json, and the entry is marked failed.
        """
        st = self._status({"rxn_1": {"tasks": {
            "ll_path.ts_guess": self._t("terminated_normally", "1", "/s/tsg"),
            "ll_refine.ts_opt": self._t("submitted", "2", "/s/tsopt"),
            "ll_refine.irc": self._t("pending"),
        }}})
        reactions = {"rxn_1": MagicMock()}
        tasks = {"ll_path.ts_guess": self._task("ts_guess"),
                 "ll_refine.ts_opt": self._task("transition_state_optimization", ["ll_path.ts_guess"], "ll_refine"),
                 "ll_refine.irc": self._task("irc_validation", ["ll_refine.ts_opt"], "ll_refine")}
        self._run(tmp_path, mocker, st, reactions, tasks, check_output=False)

        entry = st["reactions"]["rxn_1"]
        assert entry["failed"] is True
        assert "rxn_1" not in reactions
        assert {k: v["status"] for k, v in entry["tasks"].items()} == {
            "ll_path.ts_guess": "terminated_normally",
            "ll_refine.ts_opt": "finished_with_error",
            "ll_refine.irc": "pending"}
        assert entry["tasks"]["ll_refine.ts_opt"]["scratch_dir"] == "/s/tsopt"

    def test_failed_reaction_does_not_block_the_quiescent_state(self, tmp_path, mocker, capsys):
        """
        The loop exits when no task is active, pending or ready. A failed
        reaction keeps its pending tasks, but they will never run, so they must
        not count -- otherwise the loop would never finish on its own.
        """
        st = self._status({"rxn_1": {"failed": True, "tasks": {
            "ll_path.reactant_conformer": self._t("finished_with_error"),
            "ll_path.ts_guess": self._t("pending")}}})
        tasks = {"ll_path.reactant_conformer": self._task("reactant_conformer"),
                 "ll_path.ts_guess": self._task("ts_guess", ["ll_path.reactant_conformer"])}
        self._run(tmp_path, mocker, st, {}, tasks)

        assert "YARP has reached a quiescent state" in capsys.readouterr().out

    def test_failed_reactions_pending_tasks_are_not_promoted(self, tmp_path, mocker):
        """
        A failed reaction's pending task whose dependency did finish (the
        product side of a reaction whose reactant side failed) must stay
        pending, not be marked ready. With a stage filter set, promoting it
        would also look the reaction object up in the active pickle, where it
        no longer is (KeyError, crashing the whole pass).
        """
        st = self._status({"rxn_1": {"failed": True, "tasks": {
            "ll_path.reactant_conformer": self._t("finished_with_error"),
            "ll_path.product_pre_opt": self._t("terminated_normally"),
            "ll_path.product_conformer": self._t("pending")}}})
        tasks = {"ll_path.reactant_conformer": self._task("reactant_conformer"),
                 "ll_path.product_pre_opt": self._task("product_pre_opt"),
                 "ll_path.product_conformer": self._task("product_conformer", ["ll_path.product_pre_opt"])}
        stage_filter = MagicMock(type="barrier", source="egat", threshold=50.0)
        self._run(tmp_path, mocker, st, {}, tasks, stage_filters={"ll_path": stage_filter})

        assert st["reactions"]["rxn_1"]["tasks"]["ll_path.product_conformer"]["status"] == "pending"

    def test_leftover_submitted_task_does_not_block_an_identical_species(self, tmp_path, mocker):
        """
        A reaction can fail while another of its tasks is still "submitted".
        That task is never checked again, so it must not count as the active
        calculation for its species, or every other reaction needing the same
        species would wait on it forever.
        """
        mocker.patch('yarp.progress_yarp.species_registry_key', return_value=("same", "species"))
        st = self._status({
            "rxn_failed": {"failed": True, "tasks": {"ll_path.product_conformer": self._t("submitted", "9", "/s/x")}},
            "rxn_live": {"tasks": {"ll_path.product_conformer": self._t("ready")}},
        })
        reactions = {"rxn_live": MagicMock()}
        tasks = {"ll_path.product_conformer": self._task("product_conformer")}
        jm = self._run(tmp_path, mocker, st, reactions, tasks)

        assert st["reactions"]["rxn_live"]["tasks"]["ll_path.product_conformer"]["status"] == "submitted"
        assert jm.submit.called

    def test_global_task_still_fast_forwards_with_a_failed_reaction(self, tmp_path, mocker):
        """
        An ML global task is fast-forwarded when every reaction already has its
        prediction. A failed reaction has no data object in the active pickle,
        so it must not count, or the global task could never be skipped.
        """
        live = MagicMock()
        live.barrier = {"egat_rgd1": 30.0}
        st = self._status(
            {"rxn_failed": {"failed": True, "tasks": {}}, "rxn_live": {"tasks": {}}},
            global_tasks={"egat.ml_predict": self._t("ready")})
        g_task = MagicMock()
        g_task.config.model = "egat_rgd1"
        self._run(tmp_path, mocker, st, {"rxn_live": live}, {}, global_tasks={"egat.ml_predict": g_task})

        assert st["global_tasks"]["egat.ml_predict"]["status"] == "terminated_normally"


class TestRestartSupport:
    """The two places a progress pass cooperates with yarp-restart."""

    def test_pass_is_skipped_while_another_process_holds_the_run_lock(self, tmp_path, mocker, capsys):
        """
        yarp-restart holds the run lock while it edits STATUS.json. A pass that
        starts meanwhile must not load and rewrite the run; it skips and returns
        normally (a non-zero exit would end yarp-loop), to try again next tick.
        """
        from yarp.util.run_lock import run_lock
        load = mocker.patch('yarp.progress_yarp.load_state')

        with run_lock(tmp_path, "yarp-restart"):
            progress_yarp(tmp_path)

        assert not load.called
        assert "Skipping this progress pass" in capsys.readouterr().out
        assert not (tmp_path / ".yarp_run.lock").exists()

    def test_restarted_task_seed_reaches_the_calculator(self, tmp_path, mocker):
        """
        yarp-restart --new-seed stores the retry's seed on the task (43); the
        stage's shared config still says 42. The calculator built for that task
        must get 43, and the shared config must stay 42 for every other reaction.
        """
        from yarp.util.config import ConformerConfig
        from yarp.util.input import TaskDef
        shared = ConformerConfig(software="rdkit", lot="uff", charge=0, seed=42)
        task = TaskDef(task_id="ll_path.reactant_conformer", task_type="reactant_conformer",
                       parent_stage="ll_path", depends_on=[], config=shared)
        st = TestFailedReactionsKeepHistory._status({"rxn_1": {"tasks": {
            "ll_path.reactant_conformer": {"status": "ready", "job_id": None, "scratch_dir": None, "seed": 43}}}})
        mocker.patch('yarp.progress_yarp.load_state', return_value=(st, {"rxn_1": MagicMock()}))
        inp = MagicMock()
        inp.job_manager.max_active_jobs = 10
        inp.pipeline_tasks = {"ll_path.reactant_conformer": task}
        inp.global_tasks = {}
        inp.stage_filters = {}
        mocker.patch('yarp.progress_yarp.InputParser', return_value=inp)
        jm = MagicMock()
        jm.submit.return_value = "job_new"
        mocker.patch('yarp.progress_yarp.get_job_manager', return_value=jm)
        mocker.patch('yarp.progress_yarp.species_registry_key', return_value=None)
        mocker.patch('yarp.progress_yarp.pickle.dump')
        seen = []

        def spy(task_def, rxn, job_config):
            seen.append(task_def.config.seed)
            return MagicMock()

        mocker.patch('yarp.progress_yarp.get_calculator', side_effect=spy)
        progress_yarp(tmp_path)

        assert seen == [43]
        assert shared.seed == 42
