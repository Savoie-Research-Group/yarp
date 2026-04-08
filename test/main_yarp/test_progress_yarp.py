# test_progress_yarp.py
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Assuming you separate the core loop into a function like `run_progress_tick(status, rxns, work_dir, inp, job_manager)`
# Or we can just mock the load/save state and run the main function.
from yarp.progress_yarp import progress_yarp

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
                    "stage1.gsm": {"status": "pending", "job_id": None, "scratch_dir": None}
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
        "stage1.gsm": MagicMock(task_type="gsm", parent_stage="stage1", depends_on=["stage1.conf"])
    }

    mocker.patch('yarp.progress_yarp.InputParser', return_value=inp_mock)

    # --- TICK 1: Submission ---
    progress_yarp(Path("/fake/dir"))
    
    assert status_tracker["reactions"]["rxn_1"]["tasks"]["stage1.conf"]["status"] == "submitted"
    assert status_tracker["reactions"]["rxn_1"]["tasks"]["stage1.conf"]["job_id"] == "9999"
    assert status_tracker["reactions"]["rxn_1"]["tasks"]["stage1.gsm"]["status"] == "pending"

    # --- TICK 2: Completion & Advancing the DAG ---
    # We alter the state slightly to simulate that the job is no longer running on the cluster
    jm_mock.is_running.return_value = False 
    mock_calculators.has_prerequisites.return_value = True # Passed the pre-flight check
    
    progress_yarp(Path("/fake/dir"))
    
    # Task 1 should be finished
    assert status_tracker["reactions"]["rxn_1"]["tasks"]["stage1.conf"]["status"] == "terminated_normally"
    # Task 2 should now see its dependency met (ready) and IMMEDIATELY get submitted!
    assert status_tracker["reactions"]["rxn_1"]["tasks"]["stage1.gsm"]["status"] == "submitted"
    assert status_tracker["reactions"]["rxn_1"]["tasks"]["stage1.gsm"]["job_id"] == "9999"

