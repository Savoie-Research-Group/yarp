#!/usr/bin/env python3
"""build_compare_matrices.py

Build per-metal x OS matrices in the same format as
build_tm_os_matrix.py expects (header: Metal,-3,-2,-1,0,1,2,3,4,5,6,Total),
but for BOTH:

  1. The old slim CSV:    /scratch/.../doi_zips_slim/os_extraction/transition_metal_oxidation_states.csv
  2. The new FINAL CSV:   Scripts/v2/os_test_final/transition_metal_oxidation_states_FINAL.csv

NB: The full slim CSV (506k rows incl. dups) covers all charge/mult variants.
For an apples-to-apples comparison with the FINAL CSV (181k deduped picks),
we restrict both to the SAME archive set as dedup_tm_picks.txt, summing
reactant + product OS atom counts per archive into the per-metal bins.

OS values outside [NEG_MIN, POS_MAX] are clamped to the terminal bins.

Outputs:
  Scripts/v2/os_test_final/tm_os_matrix_OLD.csv    (slim, restricted to dedup set)
  Scripts/v2/os_test_final/tm_os_matrix_NEW.csv    (FINAL, restricted to dedup set)
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/scratch/negishi/li1724/SI-Downloads/SI_Agent/doi_tar_zsts")
SLIM_CSV  = Path("/scratch/negishi/li1724/SI-Downloads/SI_Agent/doi_zips_slim/os_extraction/transition_metal_oxidation_states.csv")
FINAL_CSV = ROOT / "Scripts/v2/os_test_final/transition_metal_oxidation_states_FINAL.csv"
DEDUP_LIST = ROOT / "Scripts/v2/os_test_new_yarp/dedup_tm_picks.txt"
OUT_DIR = ROOT / "Scripts/v2/os_test_final"
OUT_OLD = OUT_DIR / "tm_os_matrix_OLD.csv"
OUT_NEW = OUT_DIR / "tm_os_matrix_NEW.csv"

NEG_MIN, POS_MAX = -3, 6
OS_COLS = list(range(NEG_MIN, POS_MAX + 1))

TM_ORDER = [
    'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
    'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',
    'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg',
]
TM_SET = set(TM_ORDER)
ATOM_RE = re.compile(r"([A-Z][a-z]?)(\d+):(-?\d+)")


def clamp(os_val: int) -> int:
    if os_val < NEG_MIN: return NEG_MIN
    if os_val > POS_MAX: return POS_MAX
    return os_val


def build_matrix(csv_path: Path, allowed_paths: set, label: str):
    """Sum reactant + product OS atom counts per metal for the allowed archives only."""
    bins = defaultdict(lambda: defaultdict(int))  # metal -> os -> count
    n_archives = 0
    n_archive_atoms = 0
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            zp = row["zip_path"]
            if zp not in allowed_paths:
                continue
            n_archives += 1
            for side in ("reactant_metal_oxidation_states", "product_metal_oxidation_states"):
                s = row.get(side, "") or ""
                if not s or "ERR" in s or "SYSEXIT" in s:
                    continue
                for m in ATOM_RE.finditer(s):
                    metal = m.group(1)
                    if metal not in TM_SET:
                        continue
                    os_val = clamp(int(m.group(3)))
                    bins[metal][os_val] += 1
                    n_archive_atoms += 1

    print(f"[{label}]  archives matched: {n_archives:,}   per-atom OS rows: {n_archive_atoms:,}")
    return bins


def write_matrix(bins, out_path: Path):
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Metal"] + [str(c) for c in OS_COLS] + ["Total"])
        for metal in TM_ORDER:
            row = [bins[metal].get(c, 0) for c in OS_COLS]
            total = sum(row)
            w.writerow([metal] + row + [total])
    print(f"wrote: {out_path}")


def main():
    allowed = set(p.strip() for p in DEDUP_LIST.read_text().splitlines() if p.strip())
    print(f"allowed archives (dedup_tm_picks): {len(allowed):,}\n")

    old_bins = build_matrix(SLIM_CSV,  allowed, "OLD slim")
    new_bins = build_matrix(FINAL_CSV, allowed, "NEW FINAL")

    write_matrix(old_bins, OUT_OLD)
    write_matrix(new_bins, OUT_NEW)

    # Print over-group-max comparison
    GROUP_MAX = {'Sc':3,'Ti':4,'V':5,'Cr':6,'Mn':7,'Fe':6,'Co':5,'Ni':4,'Cu':3,'Zn':2,
                 'Y':3,'Zr':4,'Nb':5,'Mo':6,'Tc':7,'Ru':8,'Rh':6,'Pd':4,'Ag':3,'Cd':2,
                 'La':3,'Hf':4,'Ta':5,'W':6,'Re':7,'Os':8,'Ir':6,'Pt':6,'Au':5,'Hg':2}
    print("\n=== over-group-max atom-level counts (restricted dedup set) ===")
    print(f"  {'Metal':<6}{'OLD>max':>10}{'NEW>max':>10}{'Δ':>8}")
    for m in TM_ORDER:
        if m not in GROUP_MAX:
            continue
        old_over = sum(c for os_v, c in old_bins[m].items() if os_v > GROUP_MAX[m])
        new_over = sum(c for os_v, c in new_bins[m].items() if os_v > GROUP_MAX[m])
        if old_over or new_over:
            print(f"  {m:<6}{old_over:>10,}{new_over:>10,}{new_over-old_over:>+8d}")


if __name__ == "__main__":
    main()
