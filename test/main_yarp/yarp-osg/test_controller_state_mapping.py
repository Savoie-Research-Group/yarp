import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "helper" / "yarp-osg" / "YARP-OSG-Automated"))

from yarp_osg.cli import record_submit
from yarp_osg.state import OSGTask, load_state, save_state


def test_record_submit_maps_manifest_rows_to_cluster_proc_ids(tmp_path):
    work_dir = tmp_path
    status = {
        "status_output_file": "STATUS.json",
        "reaction_output_file": "YARP_RXNS.pkl",
        "input_config": {"initialize": {"job_manager": {"osg": {}}}},
        "global_tasks": {
            "egat.ml_predict": {"status": "ready", "job_id": None, "scratch_dir": None}
        },
        "reactions": {},
    }
    (work_dir / "STATUS.json").write_text(json.dumps(status))
    with (work_dir / "YARP_RXNS.pkl").open("wb") as handle:
        pickle.dump({}, handle)

    state_dir = work_dir / ".yarp_osg"
    state_path = state_dir / "state.json"
    task_dir = state_dir / "tasks" / "global" / "egat" / "attempt_1"
    task_dir.mkdir(parents=True)
    state = {
        "version": 1,
        "tasks": {
            "global.egat.ml_predict": OSGTask(
                task_id="global.egat.ml_predict",
                yarp_task_id="egat.ml_predict",
                status="prepared",
                attempt=1,
                task_dir=str(task_dir),
            ).to_dict()
        },
        "events": [],
    }
    save_state(state_path, state)

    batch = state_dir / "job_1.tsv"
    batch.write_text(f"task_id task_dir attempt\nglobal.egat.ml_predict {task_dir} 1\n")

    args = SimpleNamespace(work_dir=str(work_dir), cluster_id="4567", batch_file=str(batch))
    record_submit(args)

    updated_state = load_state(state_path)
    task = updated_state["tasks"]["global.egat.ml_predict"]
    assert task["status"] == "submitted"
    assert task["condor_id"] == "4567.0"
    assert task["cluster_id"] == "4567"
    assert task["proc_id"] == "0"

    updated_status = json.loads((work_dir / "STATUS.json").read_text())
    meta = updated_status["global_tasks"]["egat.ml_predict"]
    assert meta["status"] == "submitted"
    assert meta["job_id"] == "4567.0"
    assert meta["scratch_dir"] == str(task_dir)
