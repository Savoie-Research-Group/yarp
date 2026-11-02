# Step 03: retry step 02's clean failures with an exhaustive bnfn

Takes step 02's `outputs/<run>/rxn_clean_fail.csv`: RGD1 b2f2 reactions that
one raw `bnfn` cycle missed although both endpoints parsed clean. It tests two
hypotheses:

1. `bnfn` enumerates off only one BEM, the one with the lowest sum |formal
   charge|, so it misses chemistry that needs another BEM, such as a
   zwitterion over a neutral carbene.
2. Many atom-order failures go away if `bnfn` enumerates off every kept BEM.

The exhaustive enumeration is yarp's own `bnfn(R, 2, exhaustive=True)`: the
break and form loop runs once per kept BEM, and every BEM uses the radical
sites of BEM 0, as plain `bnfn` does. So its pass on bnfn's BEM (`fc_ind`) is
exactly step 02's `bnfn`, and with one kept BEM the two are identical.

Before yarp had `exhaustive`, this step used a patched copy of `bnfn`
(`scripts/bnfn_exhaustive.py`, now removed) that took the radical sites of
every kept BEM (their union) instead of BEM 0's.

Each reaction goes through these stages and stops at the first that recovers
its exact RGD1 reaction hash:

| Stage | Applies to | Test | Outcome |
|---|---|---|---|
| 1 | reactants with more than one kept BEM | `bnfn(exhaustive=True)` | `exhaustive_pass` |
| 2 | everything still failing, including all single-BEM reactants | `bnfn(exhaustive=True)` after each of k = 3 seeded atom-map relabellings | `reorder_pass` |
| 3 | the rest | | `persist_fail` |

Run everything from this folder (`03_diagnose_failures`) with the yarp env's
python. Parsing and hashing are step 02's.

| Path | What it is |
|---|---|
| `scripts/make_run.py` | Builds `RUN_DIR/rows.csv` from a step 02 run's `rxn_clean_fail.csv`, split into shards |
| `scripts/run_shard.py` | Runs the three stages on one shard |
| `scripts/merge.py` | Merges the shards and writes `outputs/<run>/` |
| `scripts/compare_runs.py` | Optional: compares two runs' step 02 + 03 outputs: persistent failures, cross-run recoveries, what the trials add, and reactions YARP doesn't count as b2f2 (CSV lists for the first and last) |
| `scripts/draw_reactions.py` | Optional: draws each reaction of a CSV as `<reaction>.png`: YARP's fewest-change BEM pair, atom maps, changed bonds in red/green |
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

Compare two versions with the command below, which writes
`outputs/compare_RUN_A_vs_RUN_B/`:

```bash
python scripts/compare_runs.py --a02 ../02_batch_enum/outputs/RUN_A --a03 outputs/RUN_A \
    --b02 ../02_batch_enum/outputs/RUN_B --b03 outputs/RUN_B
```

Draw the reactions of any CSV with `reaction`, `reactant` and `product`
columns, such as a `compare_runs.py` list. The BEMs come from the checked-out
yarp version. Reactions that don't parse are listed in `skipped.csv`.

```bash
python scripts/draw_reactions.py outputs/compare_RUN_A_vs_RUN_B/q1_persist_fail_both.csv --out-dir /path/to/figures
```
