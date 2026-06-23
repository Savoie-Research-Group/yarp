# Hard-coded paths in these scripts

These scripts were lifted verbatim from a working project tree. The
hard-coded paths reference my local layout and **will need to be edited
before running on another machine**. Listed here so nothing surprises
you mid-run.

## `os_new_yarp_shard.py`  (the bench worker)

```python
NEW_YARP_PATH = os.environ.get(
    "NEW_YARP_PATH",
    "/home/li1724/061226-YARP-again/Zhao-YARP/classy-yarp")     # ← fallback
```

This one is **safe**: set `NEW_YARP_PATH` in your environment (or in
the sbatch script) before invoking, and the fallback never fires.
Default in `os_p5.sbatch` already does this.

## `os_p5.sbatch`

```bash
cd /scratch/negishi/li1724/SI-Downloads/SI_Agent/doi_tar_zsts          # edit
PY=/home/li1724/.conda/envs/2022.10-py39/copy-classy-yarp/bin/python   # edit
export NEW_YARP_PATH=/home/li1724/061226-YARP-again/Zhao-YARP/classy-yarp-final   # edit
```

Plus the `SHARD` / `OUT` paths assume `Scripts/v2/os_test_final/{shards_p5,results_p5}/`
relative to that `cd`.

## `build_compare_matrices.py`

```python
ROOT       = Path("/scratch/.../doi_tar_zsts")                     # project root
SLIM_CSV   = Path("/scratch/.../doi_zips_slim/.../transition_metal_oxidation_states.csv")
FINAL_CSV  = ROOT / "Scripts/v2/os_test_final/transition_metal_oxidation_states_FINAL.csv"
DEDUP_LIST = ROOT / "Scripts/v2/os_test_new_yarp/dedup_tm_picks.txt"
```

Five paths. Easiest fix: edit these to point at:
- the `FINAL.csv` we shipped in `../corpus_181450/`
- whatever you're using as the "reference" OS CSV
- a text file with the 181,450 zip paths to compare on
  (we can ship `dedup_tm_picks.txt` separately if you want it).

## `plot_os_compare.py`

```python
ROOT = Path("/scratch/.../doi_tar_zsts")
OLD  = ROOT / "Scripts/v2/os_test_final/tm_os_matrix_OLD.csv"
NEW  = ROOT / "Scripts/v2/os_test_final/tm_os_matrix_NEW.csv"
OUT  = ROOT / "Scripts/v2/os_test_final/tm_os_compare_OLD_vs_NEW.png"
```

Point at `tm_os_matrix_OLD.csv` / `tm_os_matrix_NEW.csv` shipped in
`../corpus_181450/` and you're good.

## `aggregate.py`

Uses **relative paths** from the CWD (`Scripts/v2/os_test_final/...`).
Run it from a project root that mirrors that layout, or edit the
constants at the top.

## `draw_tm_os_radial_dials_fullcircle.py`

**No hard-coded paths.** Takes a matrix CSV as positional arg, output
SVG path via `-o`. Drop-in usable.

## Suggested clean-up (out of scope for the PR, but if it matters)

All of the above are easy `argparse` refactors. I left them as-is
because they were build-tool scripts inside a one-off project, not
library code. If they get adopted upstream as benchmarks, the obvious
refactor is to parameterize via `--corpus-csv`, `--reference-csv`,
`--dedup-list`, etc.
