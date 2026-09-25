# YARP OSG Design

## Scope

This design starts an OSG-specific YARP workflow without changing core YARP code.
Where the original planning prompt and the working `DFT-Energy-Automated` workflow
conflict, the DFT Energy workflow is treated as the operational source of truth:
script-first controls, manifest batches, HTCondor submit templates, OSDF container
paths, worker-local scratch, queue monitors, health monitors, and explicit result
staging.

The first implemented milestone is EGAT-only. CREST, pysisyphus/xTB, and ORCA are
left as phased follow-up work.

## Current YARP Lifecycle

`yarp-init` writes a pickle of reaction objects and a `STATUS.json` tracker. The
tracker stores the original input config, global task status, per-reaction task
status, job IDs, and scratch directories.

`yarp-progress` then performs these passes:

1. Load `STATUS.json` and the reaction pickle.
2. Reconstruct task definitions with `InputParser`.
3. Fast-forward tasks whose reaction objects already contain the requested data.
4. Check submitted global and per-reaction jobs through `job_manager.is_running`.
5. Validate finished outputs with each calculator's `check_output`.
6. Scrape results with each calculator's `scrape_data`.
7. Update pending tasks to ready when dependencies are satisfied.
8. Generate inputs, write scheduler scripts, submit jobs, and record job IDs.
9. Save the pickle and `STATUS.json`.

The reusable chemistry/state logic is in the calculator classes, especially:

- EGAT: `EgatMLPredict.generate_input`, `check_output`, and `scrape_data`
- CREST: `CrestConfCalculator.generate_input`, `check_output`, and `scrape_data`
- pysisyphus/xTB: GSM, min opt, TS opt, and IRC calculator methods
- ORCA: min opt, TS opt, and IRC calculator methods

The part that does not carry to OSG is the submission script model, because those
scripts assume submit-host paths, shared scratch, and nested container launches.

## OSG Mismatch

The Condor branch adds a useful shared-filesystem `CondorJobManager`, but OSG jobs
run in isolated worker scratch. The worker must not `cd` to `/home/...`, must not
expect submit-host paths to exist, and should not run `apptainer run` inside a job.

OSG execution should instead:

- stage small task-local inputs through HTCondor transfer
- launch the job inside an OSDF-hosted container with `container_image`
- run in `${_CONDOR_SCRATCH_DIR:-$PWD}`
- return a small, explicit output set
- write a durable `task_result.json`
- keep enough diagnostics for held/failed jobs

## DFT Energy Patterns Reused

The YARP OSG helper mirrors these proven DFT Energy workflow patterns:

- `start.sh` preflight and one-command preparation
- `monitor_submit.sh` queue-capacity throttling and batch-file submission
- `monitor_health.sh` held-job classification and resource escalation
- `status.sh` quick queue and workflow summary
- HTCondor submit template using an OSDF container image
- manifest-driven `queue ... from $(input_list)`
- worker script that runs only on worker-local files
- `logs/`, `submitted_batches/`, resource tracking, and held-job logs

The chemistry workflow is not copied. The access-point controller calls YARP's
existing EGAT input and scraping methods.

## Directory Layout

Generated workflow files live under:

```text
helper/yarp-osg/YARP-OSG-Automated/
├── WORKFLOW.md
├── start.sh
├── status.sh
├── monitor_submit.sh
├── monitor_health.sh
├── batch_egat.submit
├── run_egat_osg.sh
├── yarp-osg
└── yarp_osg/
```

Runtime state is written inside the initialized YARP working directory:

```text
workdir/
├── STATUS.json
├── YARP_RXNS.pkl
└── .yarp_osg/
    ├── state.json
    ├── batch_egat.submit
    ├── egat_jobs.tsv
    ├── job_1.tsv
    ├── submitted_batches/
    ├── logs/
    └── tasks/
        └── global/<task-id>/attempt_1/
            ├── forward_in.csv
            ├── reverse_in.csv
            ├── egat_command.txt
            ├── run_egat_osg.sh
            ├── forward_out.csv
            ├── reverse_out.csv
            ├── forward.log
            ├── reverse.log
            └── task_result.json
```

## Configuration

The helper reads OSG settings from the raw `STATUS.json` input config under:

```yaml
initialize:
  job_manager:
    osg:
      containers:
        egat: osdf:///ospool/ap40/data/$USER/yarp-containers/v1/yarp-egat.sif
      commands:
        egat: /path/or/command/inside/container
      resources:
        egat:
          cpus: 8
          memory_mb: 8000
          disk_mb: 2048
      retries:
        infrastructure: 3
        chemistry: 1
        quarantine_after: 3
```

Environment variables override missing config values:

- `YARP_OSG_EGAT_CONTAINER`
- `YARP_OSG_OSDF_NAMESPACE`
- `YARP_OSG_EGAT_COMMAND`
- `YARP_OSG_LOCAL_SIF_DIR`
- `YARP_OSG_MAX_JOBS`

YARP's existing EGAT calculator identifies its container image as
`erm42/yarp:egat`, which follows YARP's local Apptainer filename convention:
`erm42_yarp_egat.sif`. The local CRC-built SIFs were found under
`../containers/yarp_sifs_from_crc`. Static SIF metadata inspection showed the
EGAT runscript entrypoint is:

```text
/opt/micromamba/bin/micromamba run -p /opt/egat-env python /opt/egat/egat_predict_reaction_csv.py
```

That command is now the default direct in-container EGAT command. The OSG
container path defaults to the DFT-style namespace
`osdf:///ospool/ap40/data/$USER/yarp-containers/v1/erm42_yarp_egat.sif`, and remains
configurable because OSG workers need an OSDF-readable SIF, not the local checkout
copy.

## Manifest Format

Milestone 1 writes one EGAT manifest row:

```text
task_id task_dir attempt
global.egat_stage.ml_predict /abs/workdir/.yarp_osg/tasks/global/egat_stage_ml_predict/attempt_1 1
```

The submit template uses:

```condor
queue task_id, task_dir, attempt from $(input_list)
```

Future per-reaction manifests will add reaction hash, resource profile, retry
count, expected outputs, and retention policy fields.

## Worker Script Model

`run_egat_osg.sh` is transferred with each task. It:

- uses `${_CONDOR_SCRATCH_DIR:-$PWD}`
- prints host, working directory, task ID, attempt, and key environment values
- reads the direct EGAT command from `egat_command.txt`
- runs forward and reverse EGAT commands inside the HTCondor-selected container
- creates `forward_out.csv`, `reverse_out.csv`, logs, and `task_result.json`
- never references submit-host absolute paths
- does not invoke `apptainer` or `singularity`

The output CSVs are touched even on failure so HTCondor output transfer does not
hold merely because a failed task did not create an expected file. Harvest checks
`task_result.json` before calling YARP's scraper.

## Container Model

Milestone 1 uses:

```condor
container_image = osdf://...
```

The DFT Energy reference uses `+SingularityImage`; the helper keeps the directive
configurable in code, but defaults to `container_image` because that is the target
OSG model requested for YARP.

Reusable SIF files should live in immutable, versioned OSDF paths. Licensed ORCA
containers must remain private and are not part of this EGAT milestone.

## State Model

`.yarp_osg/state.json` is the durable audit record:

- logical YARP task ID
- task type
- status
- attempt count
- task directory
- manifest path
- Condor `ClusterId.ProcId`
- error category and message
- timestamps
- event log

The YARP `STATUS.json` remains the chemistry workflow source of truth. OSG state
tracks access-point orchestration details that are not represented in YARP today.

## Retry And Quarantine

Failures are categorized as:

- `infrastructure`: transfer, OSDF/Pelican, eviction, temporary site or filesystem errors
- `resource`: memory, CPU, disk holds
- `chemistry`: EGAT ran but did not produce valid outputs
- `configuration`: missing command, missing executable, malformed input
- `missing_output`: task completed without required result files

Infrastructure/resource failures are retryable by default. Chemistry failures are
retryable once by default. Configuration failures are not automatically retried.
Tasks exceeding configured limits become `quarantined` and remain inspectable.

## Output Retention

Milestone 1 always keeps:

- `forward_in.csv`
- `reverse_in.csv`
- `forward_out.csv`
- `reverse_out.csv`
- `forward.log`
- `forward.err`
- `reverse.log`
- `reverse.err`
- `task_result.json`
- Condor `.out`, `.err`, and `.log` files

No successful EGAT cleanup calls the existing `EgatMLPredict.cleanup`, because that
method would delete OSG diagnostics that are needed for operations.

## CLI And Scripts

DFT-style scripts are the primary workflow:

- `start.sh [workdir]`: prepare EGAT inputs, manifests, submit file, and launch monitors
- `monitor_submit.sh [workdir]`: throttle queue and submit manifest batches
- `monitor_health.sh [workdir]`: classify held jobs and escalate resources
- `status.sh [workdir]`: summarize YARP OSG state and Condor queue

The bundled `yarp-osg` script exposes controller commands for direct use:

- path-only default: `yarp-osg .`
- watched default: `yarp-osg . --watch`
- `plan`
- `prepare-egat`
- `submit`
- `record-submit`
- `status`
- `harvest`
- `retry`
- `advance`
- `cleanup`

The path-only default is the preferred universal command for Milestone 1. It runs
one idempotent controller cycle: harvest, retry bookkeeping, EGAT preparation,
pending batch submission if Condor is available, and status reporting. Redirecting
it with `yarp-osg . >> osg.log 2>&1` is safe. Use `--watch` for repeated polling.
If the argument is a parent directory without its own `STATUS.json`, `yarp-osg`
discovers direct child directories that contain initialized YARP state and runs
the same controller cycle for each child. State remains isolated under each
child's `.yarp_osg/` directory.

## Phased Plan

1. EGAT global task on OSG with OSDF container, direct EGAT command, task result
   file, Condor ID mapping, status detection, and YARP scraper harvest.
2. CREST per-reaction tasks with a small batched manifest and retained conformer
   outputs.
3. pysisyphus/xTB GSM, min opt, TS opt, and IRC chains with retry classification.
4. ORCA subset workflows with licensed private container paths and conservative
   output retention.
5. High-throughput queue throttling, backup/archival policy, measured resource
   tuning, and cleanup/reporting commands.

## Risks And Open Questions

- The EGAT SIF was not present locally, so the direct in-container command must be
  filled in after inspecting the runscript on a machine with the image.
- `yarp-init` does not currently accept `scheduler: osg`; the helper can read raw
  OSG config from `STATUS.json`, but a future core parser change is needed for a
  first-class schema.
- HTCondor `container_image` support should be confirmed on the target access point;
  the reference workflow's `+SingularityImage` can be used as a fallback if needed.
- Live OSG validation is still required for container fetch, transfer remaps, and
  held-job behavior.
