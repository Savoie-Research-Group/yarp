# test_progress_yarp.py
import pytest
import json
from unittest.mock import MagicMock, patch
from pathlib import Path
from yarp.progress_yarp import progress_yarp, save_state

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

def test_happy_path_submission_and_completion(mock_filesystem, mock_calculators, mocker):
    """
    Simulates a 'ready' task being submitted, and in the next tick, completing successfully,
    which unlocks the next 'pending' task in the DAG.
    """
    # 1. Setup Initial State (Task 1 is ready, Task 2 is pending)
    status_tracker = {
        "input_config": {"initialize": {"scheduler": "slurm"}},
        "reactions": {
            "rxn_1": {
                "tasks": {
                    "stage1.conf": {"status": "ready", "job_id": None, "scratch_dir": None},
                    "stage1.ts_guess": {"status": "pending", "job_id": None, "scratch_dir": None}
                }
            }
        }
    }
    rxns = {"rxn_1": MagicMock()} # Mock reaction object
    
    mocker.patch('yarp.progress_yarp.load_state', return_value=(status_tracker, rxns))
    
    # Mock JobManager to accept the submission
    jm_mock = MagicMock()
    jm_mock.submit.return_value = "9999"
    jm_mock.is_running.return_value = False # For the next tick
    mocker.patch('yarp.progress_yarp.get_job_manager', return_value=jm_mock)
    
    # Mock the InputParser's DAG logic
    inp_mock = MagicMock()
    inp_mock.job_manager.scheduler = "slurm"
    inp_mock.job_manager.container = "docker"
    inp_mock.job_manager.max_active_jobs = 10
    inp_mock.global_tasks = {} # Prevent iteration errors

    inp_mock.pipeline_tasks = {
        "stage1.conf": MagicMock(task_type="reactant_conformer", parent_stage="stage1", depends_on=[]),
        "stage1.ts_guess": MagicMock(task_type="gsm", parent_stage="stage1", depends_on=["stage1.conf"])
    }

    mocker.patch('yarp.progress_yarp.InputParser', return_value=inp_mock)

    # --- TICK 1: Submission ---
    progress_yarp(Path("/fake/dir"))
    
    assert status_tracker["reactions"]["rxn_1"]["tasks"]["stage1.conf"]["status"] == "submitted"
    assert status_tracker["reactions"]["rxn_1"]["tasks"]["stage1.conf"]["job_id"] == "9999"
    assert status_tracker["reactions"]["rxn_1"]["tasks"]["stage1.ts_guess"]["status"] == "pending"

    # --- TICK 2: Completion & Advancing the DAG ---
    jm_mock.is_running.return_value = False 
    mock_calculators.has_prerequisites.return_value = True # Passed the pre-flight check
    
    progress_yarp(Path("/fake/dir"))
    
    # Task 1 should be finished
    assert status_tracker["reactions"]["rxn_1"]["tasks"]["stage1.conf"]["status"] == "terminated_normally"
    # Task 2 should now see its dependency met (ready) and IMMEDIATELY get submitted!
    assert status_tracker["reactions"]["rxn_1"]["tasks"]["stage1.ts_guess"]["status"] == "submitted"
    assert status_tracker["reactions"]["rxn_1"]["tasks"]["stage1.ts_guess"]["job_id"] == "9999"


# =====================================================================
# BATCH 1: IDENTIFICATION OF FAILED REACTIONS
# =====================================================================

def test_failure_scenario_1_global_task_failed(tmp_path):
    """
    Scenario 1: The global task failed for all reactions.
    All reactions should be added to failed_rxns.pkl and removed from status_tracker.
    """
    status_tracker = {
        "status_output_file": "STATUS.json",
        "reaction_output_file": "YARP_RXNS.pkl",
        "reactions": {
            "rxn_1": {"tasks": {"stage1.ml": {"status": "finished_with_error", "error_log": "Global task failure"}}},
            "rxn_2": {"tasks": {"stage1.ml": {"status": "finished_with_error", "error_log": "Global task failure"}}}
        }
    }
    reactions = {"rxn_1": MagicMock(), "rxn_2": MagicMock()}
    failed_rxns = {"rxn_1": reactions["rxn_1"], "rxn_2": reactions["rxn_2"]}

    # PATCH pickle inside the module where save_state is defined
    with patch("yarp.progress_yarp.pickle.dump") as mock_pickle_dump, \
         patch("yarp.progress_yarp.pickle.load", return_value={}):
        
        # Run the save_state routine directly
        save_state(tmp_path, status_tracker, reactions, failed_rxns)

        # Verify that pickle was called with mocks
        assert mock_pickle_dump.call_count == 2

    # Verify that the human-readable JSON outputs were processed correctly
    fail_json_file = tmp_path / "failed_status.json"
    assert fail_json_file.exists()
    
    with open(fail_json_file, "r") as f:
        failed_status = json.load(f)
        assert "rxn_1" in failed_status
        assert failed_status["rxn_1"]["stage1.ml"]["error_log"] == "Global task failure"

    # Verify active status tracker was correctly purged of failures
    assert len(status_tracker["reactions"]) == 0


def test_failure_scenario_2_pipeline_task_failed(tmp_path, mocker):
    """
    Scenario 2: One pipeline task failed for one reaction, but not the other.
    The failed reaction is moved to failed logs, while the valid one remains active.
    """
    status_tracker = {
        "input_config": {},
        "status_output_file": "STATUS.json",
        "reaction_output_file": "YARP_RXNS.pkl",
        "reactions": {
            "rxn_failed": {"tasks": {"stage1.conf": {"status": "submitted", "job_id": "101", "scratch_dir": "/tmp/scratch"}}},
            "rxn_success": {"tasks": {"stage1.conf": {"status": "submitted", "job_id": "102", "scratch_dir": "/tmp/scratch"}}}
        }
    }
    rxn_failed = MagicMock()
    rxn_success = MagicMock()
    reactions = {"rxn_failed": rxn_failed, "rxn_success": rxn_success}
    
    mocker.patch('yarp.progress_yarp.load_state', return_value=(status_tracker, reactions))
    
    jm_mock = MagicMock()
    jm_mock.is_running.return_value = False # Jobs complete execution on queue
    mocker.patch('yarp.progress_yarp.get_job_manager', return_value=jm_mock)
    
    # Assign failure condition specifically to rxn_failed calculator evaluation
    def get_calc_side_effect(task_def, rxn_obj, job_manager):
        calc = MagicMock()
        calc.write_submission_script.return_value = "submit.sh"
        if rxn_obj == rxn_failed:
            calc.check_output.return_value = False # Triggers downstream validation error
        else:
            calc.check_output.return_value = True
            calc.scrape_data.return_value = True
        return calc
        
    mocker.patch('yarp.progress_yarp.get_calculator', side_effect=get_calc_side_effect)
    
    inp_mock = MagicMock()
    inp_mock.job_manager.scheduler = "slurm"
    inp_mock.job_manager.container = "docker"
    inp_mock.job_manager.max_active_jobs = 10
    inp_mock.global_tasks = {}
    inp_mock.pipeline_tasks = {
        "stage1.conf": MagicMock(task_type="reactant_conformer", parent_stage="stage1", depends_on=[])
    }
    mocker.patch('yarp.progress_yarp.InputParser', return_value=inp_mock)
    
    progress_yarp(tmp_path)
    
    # rxn_failed should be removed from active run-state entirely
    assert "rxn_failed" not in status_tracker["reactions"]
    assert "rxn_failed" not in reactions
    
    # rxn_success must be retained and updated normally
    assert "rxn_success" in status_tracker["reactions"]
    assert "rxn_success" in reactions
    assert status_tracker["reactions"]["rxn_success"]["tasks"]["stage1.conf"]["status"] == "terminated_normally"
    
    assert (tmp_path / "failed_rxns.pkl").exists()
    assert (tmp_path / "YARP_RXNS.pkl").exists()


def test_failure_scenario_3_irc_unintended(tmp_path, mocker):
    """
    Scenario 3: A reaction completes tasks, but IRC filters it out as "unintended".
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
    rxn_unintended.outcome_label = {"b3lyp_gaussian": "unintended"}
    rxn_intended = MagicMock()
    rxn_intended.outcome_label = {"b3lyp_gaussian": "intended"}
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
    
    task_mock = MagicMock(task_type="irc_validation", parent_stage="stage1", depends_on=[])
    task_mock.config.lot = "b3lyp"
    task_mock.config.software = "gaussian"
    inp_mock.pipeline_tasks = {"stage1.irc": task_mock}
    mocker.patch('yarp.progress_yarp.InputParser', return_value=inp_mock)
    
    progress_yarp(tmp_path)
    
    assert "rxn_unintended" not in status_tracker["reactions"]
    assert "rxn_unintended" not in reactions
    
    assert "rxn_intended" in status_tracker["reactions"]
    assert "rxn_intended" in reactions
    assert status_tracker["reactions"]["rxn_intended"]["tasks"]["stage1.irc"]["status"] == "terminated_normally"
    
    assert (tmp_path / "failed_rxns.pkl").exists()


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
        "global_tasks": {
            "ml_task": {"status": "submitted", "job_id": "777", "scratch_dir": "/tmp/g_scratch"}
        },
        "reactions": {
            "rxn_1": {
                "tasks": {
                    "conf_gen": {"status": "pending", "job_id": None, "scratch_dir": None}
                }
            }
        }
    }
    reactions = {"rxn_1": MagicMock()}
    mocker.patch('yarp.progress_yarp.load_state', return_value=(status_tracker, reactions))
    
    jm_mock = MagicMock()
    jm_mock.is_running.return_value = True # Global job is still busy on the cluster
    mocker.patch('yarp.progress_yarp.get_job_manager', return_value=jm_mock)
    
    inp_mock = MagicMock()
    inp_mock.job_manager.scheduler = "slurm"
    inp_mock.job_manager.container = "docker"
    inp_mock.job_manager.max_active_jobs = 10
    inp_mock.global_tasks = {"ml_task": MagicMock(depends_on=[])}
    inp_mock.pipeline_tasks = {
        "conf_gen": MagicMock(task_type="reactant_conformer", parent_stage="init_rxn_path", depends_on=["ml_task"])
    }
    mocker.patch('yarp.progress_yarp.InputParser', return_value=inp_mock)
    
    # --- TICK 1: Global Task is still active ---
    progress_yarp(tmp_path)
    assert status_tracker["global_tasks"]["ml_task"]["status"] == "submitted"
    assert status_tracker["reactions"]["rxn_1"]["tasks"]["conf_gen"]["status"] == "pending"
    
    # --- TICK 2: Global Task finishes ---
    jm_mock.is_running.return_value = False
    mock_calculators.check_output.return_value = True
    mock_calculators.scrape_data.return_value = True
    
    # Lock max active submissions to 0 so we catch it in the "ready" state before Pass 3 submits it
    inp_mock.job_manager.max_active_jobs = 0
    
    progress_yarp(tmp_path)
    assert status_tracker["global_tasks"]["ml_task"]["status"] == "terminated_normally"
    assert status_tracker["reactions"]["rxn_1"]["tasks"]["conf_gen"]["status"] == "ready"


def test_handoff_scenario_2_init_rxn_path_only(tmp_path, mocker):
    """
    Scenario 2: Single stage initialization & execution across multiple reactions.
    Ensures only reactions that satisfy prerequisites advance, leaving busy ones intact.
    """
    status_tracker = {
        "input_config": {},
        "status_output_file": "STATUS.json",
        "reaction_output_file": "YARP_RXNS.pkl",
        "reactions": {
            "rxn_ready": {
                "tasks": {
                    "stage1.conf": {"status": "ready", "job_id": None, "scratch_dir": None},
                    "stage1.ts_guess": {"status": "pending", "job_id": None, "scratch_dir": None}
                }
            },
            "rxn_waiting": {
                "tasks": {
                    "stage1.conf": {"status": "submitted", "job_id": "888", "scratch_dir": "/tmp/scratch"},
                    "stage1.ts_guess": {"status": "pending", "job_id": None, "scratch_dir": None}
                }
            }
        }
    }
    reactions = {"rxn_ready": MagicMock(), "rxn_waiting": MagicMock()}
    mocker.patch('yarp.progress_yarp.load_state', return_value=(status_tracker, reactions))
    
    jm_mock = MagicMock()
    jm_mock.submit.return_value = "999"
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
        "stage1.conf": MagicMock(task_type="reactant_conformer", parent_stage="stage1", depends_on=[]),
        "stage1.ts_guess": MagicMock(task_type="gsm", parent_stage="stage1", depends_on=["stage1.conf"])
    }
    mocker.patch('yarp.progress_yarp.InputParser', return_value=inp_mock)
    
    # --- TICK 1: Initial Launch ---
    progress_yarp(tmp_path)
    
    assert status_tracker["reactions"]["rxn_ready"]["tasks"]["stage1.conf"]["status"] == "submitted"
    assert status_tracker["reactions"]["rxn_ready"]["tasks"]["stage1.ts_guess"]["status"] == "pending"
    assert status_tracker["reactions"]["rxn_waiting"]["tasks"]["stage1.conf"]["status"] == "submitted"
    
    # --- TICK 2: Progress Only the Finished Reaction ---
    # Manually pass rxn_ready.conf to simulate completion on a subsequent tick loop
    status_tracker["reactions"]["rxn_ready"]["tasks"]["stage1.conf"]["status"] = "terminated_normally"
    
    progress_yarp(tmp_path)
    
    # rxn_ready should kick forward to submit ts_guess
    assert status_tracker["reactions"]["rxn_ready"]["tasks"]["stage1.ts_guess"]["status"] == "submitted"
    # rxn_waiting must still remain pending since its parent conf calculation hasn't dropped out of the scheduler
    assert status_tracker["reactions"]["rxn_waiting"]["tasks"]["stage1.ts_guess"]["status"] == "pending"


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
        "reactions": {
            "rxn_partial": {
                "tasks": {
                    "stage1.conf": {"status": "terminated_normally", "job_id": None, "scratch_dir": None},
                    "stage1.ts_guess": {"status": "pending", "job_id": None, "scratch_dir": None},
                    "stage2.rp_opt": {"status": "pending", "job_id": None, "scratch_dir": None},
                    "stage2.ts_opt": {"status": "pending", "job_id": None, "scratch_dir": None}
                }
            },
            "rxn_complete": {
                "tasks": {
                    "stage1.conf": {"status": "terminated_normally", "job_id": None, "scratch_dir": None},
                    "stage1.ts_guess": {"status": "terminated_normally", "job_id": None, "scratch_dir": None},
                    "stage2.rp_opt": {"status": "pending", "job_id": None, "scratch_dir": None},
                    "stage2.ts_opt": {"status": "pending", "job_id": None, "scratch_dir": None}
                }
            }
        }
    }
    reactions = {"rxn_partial": MagicMock(), "rxn_complete": MagicMock()}
    mocker.patch('yarp.progress_yarp.load_state', return_value=(status_tracker, reactions))
    
    jm_mock = MagicMock()
    jm_mock.submit.return_value = "555"
    jm_mock.is_running.return_value = False
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
        "stage1.conf": MagicMock(task_type="reactant_conformer", parent_stage="stage1", depends_on=[]),
        "stage1.ts_guess": MagicMock(task_type="gsm", parent_stage="stage1", depends_on=["stage1.conf"]),
        "stage2.rp_opt": MagicMock(task_type="reactant_optimization", parent_stage="stage2", depends_on=["stage1.conf"]),
        "stage2.ts_opt": MagicMock(task_type="transition_state_optimization", parent_stage="stage2", depends_on=["stage1.ts_guess"])
    }
    mocker.patch('yarp.progress_yarp.InputParser', return_value=inp_mock)
    
    progress_yarp(tmp_path)
    
    # Assertions for rxn_partial (only conf branch satisfies dependencies)
    assert status_tracker["reactions"]["rxn_partial"]["tasks"]["stage2.rp_opt"]["status"] == "submitted"
    assert status_tracker["reactions"]["rxn_partial"]["tasks"]["stage2.ts_opt"]["status"] == "pending"
    
    # Assertions for rxn_complete (both branches satisfy dependencies)
    assert status_tracker["reactions"]["rxn_complete"]["tasks"]["stage2.rp_opt"]["status"] == "submitted"
    assert status_tracker["reactions"]["rxn_complete"]["tasks"]["stage2.ts_opt"]["status"] == "submitted"