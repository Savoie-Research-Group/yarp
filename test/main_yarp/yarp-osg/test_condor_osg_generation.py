import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "helper" / "yarp-osg" / "YARP-OSG-Automated"))

from yarp_osg.condor import manifest_rows, split_manifest, write_submit_file
from yarp_osg.config import OSGConfig, ResourceProfile
from yarp_osg.egat import write_worker_script


def test_worker_script_uses_worker_scratch_and_no_nested_apptainer(tmp_path):
    script = write_worker_script(tmp_path / "run_egat_osg.sh")
    text = script.read_text()

    assert '${_CONDOR_SCRATCH_DIR:-$PWD}' in text
    assert "cd \"$WORKDIR\"" in text
    assert "apptainer" not in text
    assert "singularity" not in text
    assert "/home/" not in text
    assert "task_result.json" in text
    assert "touch forward.log forward.err reverse.log reverse.err forward_out.csv reverse_out.csv" in text


def test_submit_file_uses_osdf_container_and_transfers_only_task_files(tmp_path):
    config = OSGConfig(
        egat_container="osdf:///ospool/ap40/data/test/yarp-egat.sif",
        egat_resources=ResourceProfile(cpus=8, memory_mb=8000, disk_mb=2048),
    )
    submit = write_submit_file(
        tmp_path / "batch_egat.submit",
        config=config,
        worker_script_name="run_egat_osg.sh",
        log_dir=tmp_path / "logs",
        resources=config.egat_resources,
    )
    text = submit.read_text()

    assert "container_image = osdf:///ospool/ap40/data/test/yarp-egat.sif" in text
    assert "initialdir = $(task_dir)" in text
    assert "transfer_input_files = run_egat_osg.sh, forward_in.csv, reverse_in.csv, egat_command.txt" in text
    assert "transfer_output_files = forward_out.csv, reverse_out.csv, forward.log, forward.err, reverse.log, reverse.err, task_result.json" in text
    assert "queue task_id, task_dir, attempt from $(input_list)" in text
    assert "request_cpus = 8" in text
    assert "request_memory = 8000 MB" in text
    assert "request_disk = 2048 MB" in text


def test_manifest_split_preserves_header_and_rows(tmp_path):
    manifest = tmp_path / "egat_jobs.tsv"
    manifest.write_text(
        "\n".join(
            [
                "task_id task_dir attempt",
                "global.a /tmp/a 1",
                "global.b /tmp/b 1",
                "global.c /tmp/c 2",
            ]
        )
        + "\n"
    )

    batches = split_manifest(manifest, 2, tmp_path)

    assert [path.name for path in batches] == ["job_1.tsv", "job_2.tsv"]
    assert batches[0].read_text().splitlines() == [
        "task_id task_dir attempt",
        "global.a /tmp/a 1",
        "global.b /tmp/b 1",
    ]
    assert manifest_rows(batches[1]) == [
        {"task_id": "global.c", "task_dir": "/tmp/c", "attempt": "2"}
    ]
