# YARP OSG EGAT Workflow

This is a DFT Energy-style OSG workflow for the first YARP OSG milestone:
EGAT global ML prediction only.

## Quick Start

From an initialized YARP working directory containing `STATUS.json` and the YARP
reaction pickle:

```bash
yarp-osg . >> osg.log 2>&1
```

That command runs one idempotent controller cycle:

1. harvest finished EGAT results if Condor reports completion
2. update retry/quarantine bookkeeping
3. prepare ready EGAT inputs and manifests
4. submit pending batches when `condor_submit` is available
5. print a compact status summary

For continuous polling from a tmux session:

```bash
yarp-osg . --watch >> osg.log 2>&1
```

## Batch Directory Mode

The same command can target a parent directory containing many initialized YARP
subdirectories:

```text
batch_001/
├── case_0001/
│   ├── STATUS.json
│   └── YARP_RXNS.pkl
├── case_0002/
│   ├── STATUS.json
│   └── YARP_RXNS.pkl
└── case_0050/
    ├── STATUS.json
    └── YARP_RXNS.pkl
```

Run one controller cycle over every direct child initialized with `yarp-init`:

```bash
yarp-osg ./batch_001 >> osg.log 2>&1
```

Run repeated cycles:

```bash
yarp-osg ./batch_001 --watch >> osg.log 2>&1
```

Each child directory keeps its own `.yarp_osg/` state, manifests, Condor IDs, and
logs. The parent command only fans out over the child workdirs in sorted order.
If the target path itself contains a `STATUS.json`, it is treated as a single YARP
workdir rather than a batch root.

The direct EGAT command is discovered from the local CRC-built SIF metadata and
defaults to:

```bash
/opt/micromamba/bin/micromamba run -p /opt/egat-env python /opt/egat/egat_predict_reaction_csv.py
```

The OSG container path must be OSDF-accessible. By default, the helper assumes the
DFT-style namespace:

```text
osdf:///ospool/ap40/data/$USER/yarp-containers/v1/erm42_yarp_egat.sif
```

Override it with an explicit container path:

```bash
export YARP_OSG_EGAT_CONTAINER="osdf:///ospool/ap40/data/$USER/yarp-containers/v1/erm42_yarp_egat.sif"
```

or a different OSDF namespace:

```bash
export YARP_OSG_OSDF_NAMESPACE="/ospool/ap40/data/$USER"
```

Then run:

```bash
./yarp-osg . >> osg.log 2>&1
```

Do not set the EGAT command to `apptainer run`; HTCondor starts the worker script
inside the selected container.

## Files

- `start.sh` prepares EGAT inputs and launches monitors.
- `monitor_submit.sh` submits manifest batches when queue capacity is available.
- `monitor_health.sh` releases transfer/container holds and escalates resources.
- `status.sh` prints OSG state and Condor queue summaries.
- `batch_egat.submit` is the static submit template; the controller writes a
  configured copy to `.yarp_osg/batch_egat.submit`.
- `run_egat_osg.sh` runs on the OSG worker in local scratch.
- `yarp-osg` exposes direct controller commands.

## Runtime State

Runtime files are written under the YARP workdir:

```text
.yarp_osg/
├── state.json
├── batch_egat.submit
├── egat_jobs.tsv
├── job_1.tsv
├── submitted_batches/
├── logs/
└── tasks/global/<task>/attempt_1/
```

## Commands

```bash
./yarp-osg plan /path/to/yarp/workdir
./yarp-osg /path/to/yarp/workdir
./yarp-osg /path/to/yarp/workdir --watch
./yarp-osg prepare-egat /path/to/yarp/workdir
./yarp-osg submit /path/to/yarp/workdir
./monitor_submit.sh /path/to/yarp/workdir
./status.sh /path/to/yarp/workdir
./yarp-osg harvest /path/to/yarp/workdir
./yarp-osg advance /path/to/yarp/workdir
```

## Live OSG Requirements

- `condor_submit`, `condor_q`, and `condor_history` on the access point
- an OSDF-hosted EGAT SIF readable by the submit account
- a direct EGAT command inside that SIF
- working `container_image` support on the target access point

If the target access point needs the older DFT Energy directive, set:

```yaml
initialize:
  job_manager:
    osg:
      container_directive: "+SingularityImage"
```
