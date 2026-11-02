# Step 02: one raw b2f2 cycle per RGD1 reactant

Runs `bnfn(R, 2)` once for every reactant in step 01's ground truth and checks
which RGD1 b2f2 reactions it recovers. Needs
`../01_get_b2f2_ground_truth/results/b2f2_rows.csv`. Run everything from this
folder (`02_batch_enum`) with the yarp env's python.

| Path | What it is |
|---|---|
| `scripts/make_run.py` | Builds `RUN_DIR/rows.csv` for the `pilot` or `full` run, split into shards |
| `scripts/submit.py` | Fills `inputs/cluster_submit_template.sh` into `RUN_DIR/job.sh` (it never runs `qsub`), or runs the shards here with `--local` |
| `scripts/run_shard.py` | Parses, enumerates and scores one shard (called by `job.sh` or `--local`) |
| `scripts/merge.py` | Merges the shards and writes `outputs/<run>/` |
| `scripts/compare_runs.py` | Optional: compares two runs' outputs: recovery counts, and SMILES parsing related failures (a CSV per run) |
| `inputs/pilot_reactions.csv` | The pilot's 50 reactions: 40 random, plus 10 that hit known parse problems |
| `inputs/cluster_submit_template.sh` | SGE job-array template; edit it for another cluster |
| `runs/<run>/` | Working files: `rows.csv`, `job.sh`, `logs/`, `out/`, `merged/` (not tracked) |
| `outputs/<run>/` | Results, below (only `summary.txt` is tracked) |

## Outputs

`scripts/merge.py` puts every RGD1 row of a run into exactly one CSV in
`outputs/<run>/`, with its reactant and product SMILES, its results, and its
reactant's kept-BEM count (`n_bem_R`) and time (`t_total`):

| File | Rows |
|---|---|
| `rxn_match.csv` | RGD1 reaction recovered by reaction hash |
| `rxn_clean_fail.csv` | Not recovered; reactant and product both parsed clean. Step 03 diagnoses these |
| `rxn_parse_fail.csv` | Not recovered; an endpoint parsed with a warning, fell back to RDKit, or failed |
| `summary.txt` | Readable summary: parsing, recovery, timing, and the counts above |

## Pilot, locally (about 1 minute)

```bash
python scripts/make_run.py pilot
python scripts/submit.py runs/pilot --local
python scripts/merge.py runs/pilot
```

## Full run, on an SGE cluster

Check `inputs/cluster_submit_template.sh` first (queue, wall time,
environment). Put the run folder on fast scratch, not a home directory.

```bash
python scripts/make_run.py full --run-dir /path/to/scratch/full
python scripts/submit.py /path/to/scratch/full
qsub /path/to/scratch/full/job.sh
python scripts/merge.py /path/to/scratch/full
```

200 tasks of about 5 minutes each. Submitting `job.sh` again reruns only the
shards without a `DONE` marker. The pilot can go through the cluster the same
way. `merge.py` writes to `outputs/<run folder name>/` unless given
`--out-dir`.

## Reference results

The first two rows are from `debug/261020_rgd1_baseline` (CRC, 2026-10-21),
the last from `debug/261028_exhaust_enum` (CRC, 2026-10-28). Scored rows are
those where both endpoints parse and `reaction()` builds.

| yarp version | Pilot recovered | Full recovered | Full CPU time |
|---|---|---|---|
| master d042ee2 + #67 hashes | 88/92 | 116,310/122,870 (94.66%) | 17.4 h |
| conjugated-lewis-moves da3a1ae | 89/92 | 117,764/122,870 (95.84%) | 24.2 h |
| exhaust_enum fedf73a | 89/92 | 117,766/122,870 (95.85%) | 29.0 h |

On the full runs, the split is 116,310 / 6,013 / 8,926 (match / clean fail /
parse fail) on master d042ee2, 117,764 / 4,605 / 8,880 on
conjugated-lewis-moves, and 117,766 / 4,603 / 8,880 on exhaust_enum.

Compare two versions with
`python scripts/compare_runs.py --a02 outputs/RUN_A --b02 outputs/RUN_B`, which
writes `outputs/compare_RUN_A_vs_RUN_B/`.
