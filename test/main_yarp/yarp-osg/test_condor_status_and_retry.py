import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "helper" / "yarp-osg" / "YARP-OSG-Automated"))

from yarp_osg.condor import classify_hold, parse_cluster_id, query_job
from yarp_osg.config import RetryConfig
from yarp_osg.state import OSGTask, can_retry, retry_or_quarantine


def completed(stdout):
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def test_condor_submit_cluster_id_parser():
    output = "Submitting job(s).\n1 job(s) submitted to cluster 98765.\n"
    assert parse_cluster_id(output) == "98765"


def test_hold_classification_matches_osg_failure_modes():
    assert classify_hold("Error from slot: transfer input files failed for OSDF path") == "infrastructure"
    assert classify_hold("Job has gone over memory limit") == "resource"
    assert classify_hold("Cannot execute: No such file or directory") == "configuration"


def test_query_job_classifies_held_job_from_condor_q_json():
    def runner(cmd, **kwargs):
        assert cmd[0] == "condor_q"
        return completed(
            json.dumps(
                [
                    {
                        "ClusterId": 123,
                        "ProcId": 0,
                        "JobStatus": 5,
                        "HoldReason": "OSDF container fetch failed",
                    }
                ]
            )
        )

    status = query_job("123.0", runner=runner)

    assert status.state == "held"
    assert status.error_category == "infrastructure"
    assert status.hold_reason == "OSDF container fetch failed"


def test_query_job_falls_back_to_history_for_completed_job():
    calls = []

    def runner(cmd, **kwargs):
        calls.append(cmd[0])
        if cmd[0] == "condor_q":
            return completed("")
        return completed(json.dumps([{"ExitCode": 0}]))

    status = query_job("123.0", runner=runner)

    assert calls == ["condor_q", "condor_history"]
    assert status.state == "complete"


def test_retry_and_quarantine_state_transitions():
    retries = RetryConfig(infrastructure=3, chemistry=1, quarantine_after=3)
    task = OSGTask(task_id="global.egat", yarp_task_id="egat.ml_predict", status="failed", attempt=1)
    task.error_category = "infrastructure"

    assert can_retry(task, retries)
    retried = retry_or_quarantine(task, retries)
    assert retried.status == "planned"
    assert retried.condor_id is None

    maxed = OSGTask(task_id="global.egat", yarp_task_id="egat.ml_predict", status="failed", attempt=3)
    maxed.error_category = "infrastructure"
    quarantined = retry_or_quarantine(maxed, retries)
    assert quarantined.status == "quarantined"

    config_error = OSGTask(task_id="global.egat", yarp_task_id="egat.ml_predict", status="failed", attempt=1)
    config_error.error_category = "configuration"
    assert not can_retry(config_error, retries)
