#!/usr/bin/env python3
"""plot_os_compare.py

Side-by-side per-metal OS histogram comparison: OLD slim CSV vs NEW FINAL CSV
restricted to the same 181k dedup_tm_picks set. For each metal, show grouped
bars (old vs new) per OS bin, shade the bins ABOVE the group max in light red
to highlight chemically-impossible regions, and annotate Δ = new - old.

Output: tm_os_compare_OLD_vs_NEW.png
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/scratch/negishi/li1724/SI-Downloads/SI_Agent/doi_tar_zsts")
OLD  = ROOT / "Scripts/v2/os_test_final/tm_os_matrix_OLD.csv"
NEW  = ROOT / "Scripts/v2/os_test_final/tm_os_matrix_NEW.csv"
OUT  = ROOT / "Scripts/v2/os_test_final/tm_os_compare_OLD_vs_NEW.png"

TM_ORDER = [
    'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
    'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',
    'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg',
]
GROUP_MAX = {'Sc':3,'Ti':4,'V':5,'Cr':6,'Mn':7,'Fe':6,'Co':5,'Ni':4,'Cu':3,'Zn':2,
             'Y':3,'Zr':4,'Nb':5,'Mo':6,'Tc':7,'Ru':8,'Rh':6,'Pd':4,'Ag':3,'Cd':2,
             'Hf':4,'Ta':5,'W':6,'Re':7,'Os':8,'Ir':6,'Pt':6,'Au':5,'Hg':2}
OS_COLS = list(range(-3, 7))


def load_matrix(path):
    out = {}
    with path.open() as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            m = row["Metal"]
            out[m] = {int(k): int(v) for k, v in row.items() if k not in ("Metal", "Total")}
    return out


old = load_matrix(OLD)
new = load_matrix(NEW)

# 6 rows x 5 cols subplots so 29 + 1 legend cell
nrow, ncol = 6, 5
fig, axes = plt.subplots(nrow, ncol, figsize=(20, 22), constrained_layout=True)
axes = axes.flatten()
fig.suptitle("Per-metal OS distribution — OLD published slim CSV  vs  NEW patched-YARP FINAL CSV\n"
             "(restricted to apples-to-apples deduplicated 181,450-archive set)",
             fontsize=14)

bar_w = 0.4
x_positions = np.arange(len(OS_COLS))

for ax, metal in zip(axes, TM_ORDER):
    old_vals = [old[metal].get(c, 0) for c in OS_COLS]
    new_vals = [new[metal].get(c, 0) for c in OS_COLS]
    delta = [n - o for n, o in zip(new_vals, old_vals)]
    gmax = GROUP_MAX.get(metal, 7)

    # Shade impossible-OS region (> gmax)
    for i, c in enumerate(OS_COLS):
        if c > gmax:
            ax.axvspan(i - 0.5, i + 0.5, color='#FFD6D6', alpha=0.55, zorder=0)

    ax.bar(x_positions - bar_w/2, old_vals, bar_w, label="OLD slim",
           color='#23364A', alpha=0.85, edgecolor='none')
    ax.bar(x_positions + bar_w/2, new_vals, bar_w, label="NEW FINAL",
           color='#D4AF37', alpha=0.95, edgecolor='none')

    over_old = sum(v for c, v in zip(OS_COLS, old_vals) if c > gmax)
    over_new = sum(v for c, v in zip(OS_COLS, new_vals) if c > gmax)
    total = sum(old_vals) or 1

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(c) for c in OS_COLS], fontsize=8)
    title = f"{metal}  (max OS={gmax})"
    if over_old != over_new:
        sign = "+" if over_new > over_old else ""
        title += f"  over-max: {over_old:,}→{over_new:,} ({sign}{over_new-over_old:+,d})"
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(axis='y', linewidth=0.3, alpha=0.4)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

# Hide unused subplots
for ax in axes[len(TM_ORDER):]:
    ax.axis('off')

# Use the last empty subplot for legend
leg_ax = axes[len(TM_ORDER)]
leg_ax.axis('off')
from matplotlib.patches import Patch
leg_ax.legend(handles=[
    Patch(color='#23364A', alpha=0.85, label='OLD slim CSV'),
    Patch(color='#D4AF37', alpha=0.95, label='NEW FINAL CSV'),
    Patch(facecolor='#FFD6D6', alpha=0.55, label='OS > group max\n(chemically impossible)'),
], loc='center', fontsize=11, frameon=False)

# Axis label common
fig.supxlabel("oxidation state bin", fontsize=12)
fig.supylabel("atom-level count (Σ reactant + product across 181,450 archives)", fontsize=12)

fig.savefig(OUT, dpi=120, bbox_inches='tight')
print(f"wrote: {OUT}")
