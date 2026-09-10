# DFT Energy (ORCA + Multiwfn WBO) Workflow

## Quick Start (One Button)

```bash
# 1. Place all scripts and your input *.tar.gz in one folder
# 2. Run:
./start.sh

# Check status:
./status.sh
```

## Overview

This workflow runs DFT single-point energy calculations (ORCA) and Wiberg/Mayer/Fuzzy bond order analysis (Multiwfn) on a large set of molecular geometries using HTCondor on OSPool.

Each input `.zip` file encodes charge and multiplicity in the filename (e.g., `05_-1_1.zip` → charge=-1, mult=1) and contains `.xyz` geometry files. The workflow processes up to 6 molecules per zip: `finished_last`, `finished_last_opt`, `ts_final_geometry`, `finished_first`, `finished_first_opt`, `input`.

## Prerequisites

- Access to an OSPool submit node
- HTCondor client tools (`condor_submit`, `condor_q`, etc.)
- tmux (for background monitoring)
- Shared container: `orca_6_1_1-zip-multiwfn-2.sif`
  - Each user must copy it to their own OSDF directory: `/ospool/ap40/data/$USER/`
  - OSDF paths are per-user; you cannot access another user's files

## Directory Structure

```
workdir/                            # Your working directory
├── start.sh                        # One-button launcher
├── add_data.sh                     # Add more data to a running workflow
├── prepare_round2.sh               # Prepare resubmission for timed-out jobs
├── run_orca_wbo.sh                 # Runs inside container on workers
├── batch_job.submit                # HTCondor submit file
├── monitor_submit.sh               # Job submission monitor
├── monitor_health.sh               # Auto-heal monitor
├── backup_results.sh               # Results backup to /ospool storage
├── status.sh                       # Quick status check
├── my_inputs.tar.gz                # Your input archive (before start)
│
├── inputs/                         # Extracted input data (after start)
│   └── SpringerNature/
│       └── subdir/
│           ├── 05_0_1.zip
│           └── 06_-1_1.zip
│
├── file_list.txt                   # Master list of all jobs
├── job_1.txt ... job_N.txt         # Split chunks (1000 jobs each)
├── submitted_batches/              # Tracking: submitted chunks
├── logs/                           # HTCondor job logs
│
├── results/                        # Output (mirrors input structure)
│   └── SpringerNature/
│       └── subdir/
│           ├── 05_0_1_results.tar.gz
│           └── 06_-1_1_results.tar.gz
│
├── health_monitor.log              # Health monitor log
├── backup_results.log              # Backup monitor log
├── resource_tracking.dat           # Tracks resource escalation per job
├── held_jobs_maxed.log             # Jobs that hit max resources
└── held_jobs_unknown.log           # Jobs held for unknown reasons
```

## Workflow Phases

### What `start.sh` Does Automatically

1. **Preflight checks** — verifies all scripts and input tar.gz are present
2. **Extracts inputs** — `*.tar.gz → inputs/` preserving nested structure
3. **Generates file list** — maps each `.zip` to its relative output path
4. **Mirrors directories** — pre-creates matching structure under `results/`
5. **Splits into chunks** — creates `job_1.txt`, `job_2.txt`, ... (1000 jobs each)
6. **Configures submit file** — replaces placeholder tokens with actual paths
7. **Launches monitors** — three tmux sessions: submit, health, backup

### Monitor: Submission (`tmux attach -t submit`)

- Reads `job_*.txt` files in order
- Checks HTCondor queue before each submission (cap: 10,000 jobs)
- Submits one chunk, waits for capacity, submits next
- Moves submitted chunks to `submitted_batches/`
- Skips already-submitted chunks on restart

### Monitor: Health (`tmux attach -t health`)

- Checks every 5 minutes for held jobs
- **Transfer failures** (SIF/OSDF issues) → auto-release
- **Memory/CPU exceeded** → escalate to next resource level + release
- **Disk exceeded** → increase to 40GB + release
- **Unknown holds** → logged for manual review
- Jobs at max level are NOT auto-released

### Monitor: Backup (`tmux attach -t backup`)

- Checks every 4 hours for result files older than 4 hours
- Waits for ≥100 eligible files before packing
- Creates verified tar.gz at `/ospool/ap40/data/$USER/dft_energy_backups/`
- Deletes originals only after verification
- Preserves nested directory structure inside archives
- Alerts if destination > 90% full

## Resource Escalation

| Level | Memory | CPUs | Action |
|-------|--------|------|--------|
| 0 | 12 GB | 4 | Default |
| 1 | 20 GB | 6 | Auto-escalate |
| 2 | 32 GB | 8 | Auto-escalate |
| 3 | 40 GB | 10 | **MAX** — logged, NOT auto-released |

Note: ORCA's `nproc` is set to 4 and `%maxcore` to 2500 in the input file (total ~10 GB).
CPU escalation prevents "excessive CPU usage" holds but doesn't make ORCA use more cores.

## Configuration

| File | Setting | Default |
|------|---------|---------|
| `start.sh` | `CONTAINER_IMAGE` | `osdf://.../orca_6_1_1-zip-multiwfn-2.sif` |
| `start.sh` | `JOBS_PER_CHUNK` | 1000 |
| `start.sh` | `MAX_JOBS` | 10000 |
| `monitor_submit.sh` | `MAX_JOBS` | 10000 |
| `monitor_submit.sh` | `CHECK_INTERVAL` | 300 (5 min) |
| `monitor_health.sh` | `CHECK_INTERVAL` | 300 (5 min) |
| `monitor_health.sh` | `RESOURCE_LEVELS` | 12/20/32/40 GB × 4/6/8/10 CPUs |
| `backup_results.sh` | `CHECK_INTERVAL` | 14400 (4 hrs) |
| `backup_results.sh` | `MIN_FILES_TO_PACK` | 100 |
| `batch_job.submit` | `request_cpus` | 4 |
| `batch_job.submit` | `request_memory` | 12 GB |
| `batch_job.submit` | `request_disk` | 20 GB |

## Common Commands

```bash
# Status
./status.sh                        # Quick overview
condor_q $USER                     # Full queue
condor_q -hold $USER               # Held jobs only

# View monitors
tmux attach -t submit              # Ctrl+B, D to detach
tmux attach -t health
tmux attach -t backup

# Emergency stop
condor_rm $USER                    # Remove all jobs
tmux kill-session -t submit
tmux kill-session -t health
tmux kill-session -t backup

# Manual resource fix for maxed jobs
condor_qedit JOB_ID RequestMemory 51200   # 50 GB
condor_qedit JOB_ID RequestCpus 12
condor_release JOB_ID
```

## Adding More Data to a Running Workflow

You can add new input tar.gz files without stopping the health or backup monitors:

```bash
# Option 1: Place new tar.gz in the working directory, then:
./add_data.sh

# Option 2: Specify the file explicitly:
./add_data.sh /path/to/new_inputs.tar.gz
```

What `add_data.sh` does:
1. Extracts new archives into `inputs/`
2. Scans for `.zip` files NOT already tracked in `file_list.txt`
3. Appends only new entries to `file_list.txt`
4. Creates new `job_*.txt` chunks (numbering continues from existing)
5. Restarts **only** the submit monitor (health & backup stay alive)

Safe to run multiple times — it never re-adds zips already tracked.

**Tip**: After running, move processed tar.gz to a `processed/` folder:
```bash
mkdir -p processed && mv *.tar.gz processed/
```

## Wall-Time Awareness

Worker nodes have a 20-hour limit. `run_orca_wbo.sh` handles this with three layers:

1. **Pre-molecule check**: Before each molecule, checks if ≥2 hours remain. If not, skips remaining molecules and packs what's done.
2. **Background watchdog**: Fires at 19 hours (30 min before wall limit). If ORCA is mid-computation, the watchdog kills it so the script can still pack results.
3. **status.json**: Every result tarball includes a `status.json` recording per-molecule status:

```json
{
  "status": "partial",
  "molecules": {
    "finished_last": "complete",
    "finished_last_opt": "complete",
    "ts_final_geometry": "timeout_killed",
    "finished_first": "timeout_skipped",
    "finished_first_opt": "timeout_skipped",
    "input": "timeout_skipped"
  }
}
```

Possible molecule statuses: `complete`, `complete_previous_round`, `timeout_skipped`, `timeout_killed`, `orca_failed`, `orca_no_gbw`, `molden_failed`, `missing`.

The wall time defaults to 19.5 hours and can be overridden via environment variable:
```bash
WALL_SECONDS=36000 ./run_orca_wbo.sh input.zip   # 10-hour limit
```

## Round-2 Resubmission

For jobs that timed out (status: "partial"), run:

```bash
./prepare_round2.sh
```

This:
1. Scans `results/` (and backup archives) for partial results
2. For each partial job, creates a self-contained round-2 zip containing:
   - Original `.xyz` files for unfinished molecules
   - Completed `.out`, `.inp`, `*_mat.txt` files from round-1
   - `skip.txt` listing already-completed molecules
3. Generates `file_list_round2.txt`

Submit with:
```bash
# Move into inputs/ and use add_data.sh (recommended)
cp -r round2_inputs/* inputs/
./add_data.sh

# Or submit directly
condor_submit batch_job.submit input_list=file_list_round2.txt
```

Round-2 jobs read `skip.txt`, skip completed molecules, compute only the remaining ones, then pack **everything** (old + new) into `results.tar.gz`. The result is a complete superset that replaces the partial round-1 file.

## Post-Processing

Result filtering will be added later. For now, results are in `results/` (local)
and archived in `/ospool/ap40/data/$USER/dft_energy_backups/` (long-term).
