# `zhao-patches-doc/` — supplementary material for the patch bundle

This directory is shipped alongside the code patches on branch
`zhao-final-20260619`. Everything here is optional context — the
maintainers can drop it on merge if they prefer the working tree clean.

## What's in here

```
zhao-patches-doc/
├── README.md                                    ← this file
│
├── YARP-3.0-OS-divergence-investigation.md      ← full ~10-page writeup
├── YARP-3.0-OS-divergence-summary.md            ← 1-page exec summary
│
├── tm_os_matrix_OLD.csv                         ← per-metal OS bin matrix
├── tm_os_matrix_NEW.csv                         ← per-metal OS bin matrix
├── tm_os_compare_OLD_vs_NEW.png                 ← grouped bar comparison
├── tm_os_dials_OLD.png                          ← published-style dial plot
├── tm_os_dials_NEW.png                          ← dial plot from FINAL CSV
│
├── scripts/                                     ← build / bench / plot scripts
│   ├── PATHS_NOTE.md                            ← READ FIRST — hardcoded paths
│   ├── os_new_yarp_shard.py
│   ├── os_p5.sbatch
│   ├── aggregate.py
│   ├── build_compare_matrices.py
│   ├── draw_tm_os_radial_dials_fullcircle.py
│   └── plot_os_compare.py
│
└── bench_stratified_144/                        ← per-condition bisection
    ├── sanity_stratified_input.txt              ← the 144-archive sample list
    ├── master_strat.csv                         ← raw new YARP master
    ├── only-A_strat.csv                         ← +recursion limit only
    ├── only-B_strat.csv                         ← +min_opt/min_win (NOT kept)
    ├── only-C_strat.csv                         ← +outer for-loop removal
    ├── only-D_strat.csv                         ← +move 4-bis removal
    ├── ABCD_strat.csv                           ← all four (including B)
    ├── ABCDw_strat.csv                          ← above + w_rad revert
    ├── ABCDwE_strat.csv                         ← above + always-disable re-pool
    └── FINAL_strat.csv                          ← final stack (A+C+D+w_rad+F)
```

Total size: 1.7 MB.

## What's NOT in here

The full 181,450-row master OS CSV and the input zip-path list
(`transition_metal_oxidation_states_FINAL.csv` 24 MB +
`dedup_tm_picks.txt` 21 MB) are intentionally excluded to keep the
PR branch small. Both are included in the
`PR-classy-yarp-zhao-final-20260619.zip` attachment on the PR
description (3.7 MB compressed; the two CSVs share long common
prefixes and shrink dramatically). Unpack the zip to get a directory
that mirrors this `zhao-patches-doc/` layout plus the two heavy files
under `corpus_181450/`.

## Reading order

1. Skim `YARP-3.0-OS-divergence-summary.md` (1 page).
2. If you want code-review context, read sections 1–4 of the
   investigation MD.
3. To verify our chemistry claims, the `bench_stratified_144/*.csv`
   files are the per-condition outputs the investigation tables
   summarize. Each row is `zip_path, R_OS, P_OS` strings.
4. `tm_os_compare_OLD_vs_NEW.png` is the single most informative
   figure — per-metal grouped bar chart with the chemically-impossible
   OS region shaded pink.

## Reproducing

See `scripts/PATHS_NOTE.md` first — several scripts have hard-coded
paths to my local project tree that need editing before they'll run on
your machine.

Build commands and which script feeds which output are documented in
section 6 of the investigation MD ("Reproducing the bench") and in
the cross-reference table at the bottom of this README.

## Original locations of the scripts

| Script | Project path |
|---|---|
| `os_new_yarp_shard.py` | `Scripts/v2/os_test_new_yarp/os_new_yarp_shard.py` |
| `os_p5.sbatch` | `Scripts/v2/os_test_final/os_p5.sbatch` |
| `aggregate.py` | `Scripts/v2/os_test_final/aggregate.py` |
| `build_compare_matrices.py` | `Scripts/v2/build_compare_matrices.py` |
| `plot_os_compare.py` | `Scripts/v2/os_test_final/plot_os_compare.py` |
| `draw_tm_os_radial_dials_fullcircle.py` | `Scripts/draw_tm_os_radial_dials_fullcircle.py` |
