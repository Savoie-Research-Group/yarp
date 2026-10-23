# Step 03: retry step 02's clean failures with an exhaustive bnfn

Takes step 02's `outputs/<run>/rxn_clean_fail.csv`: RGD1 b2f2 reactions that
one raw `bnfn` cycle missed although both endpoints parsed clean. It tests two
hypotheses:

1. `bnfn` enumerates off only one BEM, the one with the lowest sum |formal
   charge|, so it misses chemistry that needs another BEM, such as a
   zwitterion over a neutral carbene.
2. Many atom-order failures go away if `bnfn` enumerates off every kept BEM.

`scripts/bnfn_exhaustive.py` is a copy of `yarp.reaction.enum.bnfn` with two
changes, marked `STEP 03 CHANGE`:
- radical sites come from every kept BEM (`return_radicals(y, all_bmats=True)`);
- the break and form loop runs once per kept BEM.

The yarp source is not changed. With one kept BEM, `bnfn_exhaustive` yields
exactly what `bnfn` yields.

Each reaction goes through these stages and stops at the first that recovers
its exact RGD1 reaction hash:

| Stage | Applies to | Test | Outcome |
|---|---|---|---|
| 1 | reactants with more than one kept BEM | `bnfn_exhaustive` | `exhaustive_pass` |
| 2 | everything still failing, including all single-BEM reactants | `bnfn_exhaustive` after each of k = 3 seeded atom-map relabellings | `reorder_pass` |
| 3 | the rest | | `persist_fail` |

Run everything from this folder (`03_diagnose_failures`) with the yarp env's
python. Parsing and hashing are step 02's.

| Path | What it is |
|---|---|
| `scripts/bnfn_exhaustive.py` | The patched `bnfn` copy |
| `scripts/make_run.py` | Builds `RUN_DIR/rows.csv` from a step 02 run's `rxn_clean_fail.csv`, split into shards |
| `scripts/run_shard.py` | Runs the three stages on one shard |
| `scripts/merge.py` | Merges the shards and writes `outputs/<run>/` |
| `runs/<run>/` | Working files (not tracked) |
| `outputs/<run>/` | `rxn_exhaustive_pass.csv`, `rxn_reorder_pass.csv`, `rxn_persist_fail.csv`, `summary.txt` (only `summary.txt` is tracked) |

Submission reuses step 02's `scripts/submit.py` and
`inputs/cluster_submit_template.sh`, with `--driver` pointing at this step's
`run_shard.py`.

## Pilot, locally

```bash
python scripts/make_run.py pilot
python ../02_batch_enum/scripts/submit.py runs/pilot --local --driver scripts/run_shard.py
python scripts/merge.py runs/pilot
```

## Full run, on an SGE cluster

`make_run.py full` reads `../02_batch_enum/outputs/full/rxn_clean_fail.csv`,
so run step 02's `merge.py` first.

```bash
python scripts/make_run.py full --run-dir /path/to/scratch/full_03
python ../02_batch_enum/scripts/submit.py /path/to/scratch/full_03 --driver scripts/run_shard.py --job-name rgd1_03_full
qsub /path/to/scratch/full_03/job.sh
python scripts/merge.py /path/to/scratch/full_03 --out-dir outputs/full
```

50 shards by default. `make_run.py` prints an upper-bound CPU time.
