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
    Validates that the reaction is purged from active pools, and human-readable
    logs are cleanly extracted into failed_status.json.
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

    # Verify active state was successfully cleaned and purged of the failed reaction
    assert "rxn_1" not in status_tracker["reactions"]
    assert "rxn_1" not in reactions

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

    # Patch pickle operations to prevent MagicMock pickling errors
    mocker.patch('yarp.progress_yarp.pickle.dump')
    mocker.patch('yarp.progress_yarp.pickle.load', return_value={})

    progress_yarp(tmp_path)

    assert "rxn_unintended" not in status_tracker["reactions"]
    assert "rxn_unintended" not in reactions

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
        "ll_path.ts_guess": MagicMock(task_type="ts_guess", parent_stage="ll_path", depends_on=["ll_path.reactant_conformer", "llpath.product_conformer"])
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