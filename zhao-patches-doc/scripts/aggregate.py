#!/usr/bin/env python3
"""Concatenate ALL completed shard CSVs into one master CSV.

Sources, in priority order (later overwrites earlier on duplicate zip_path):
  1. Scripts/v2/os_test_final/results/        (phase 2: 1815-archive shards)
  2. Scripts/v2/os_test_final/results_retry/  (phase 3: 400-archive shards that
                                               cover the phase-2 timeouts)

Reports missing shards in either tree.
"""
from pathlib import Path
import sys

PHASE2_RES   = Path("Scripts/v2/os_test_final/results")
PHASE2_SHARD = Path("Scripts/v2/os_test_final/shards_big")
PHASE3_RES   = Path("Scripts/v2/os_test_final/results_retry")
PHASE3_SHARD = Path("Scripts/v2/os_test_final/shards_retry")
PHASE4_RES   = Path("Scripts/v2/os_test_final/results_p4")
PHASE4_SHARD = Path("Scripts/v2/os_test_final/shards_p4")
PHASE5_RES   = Path("Scripts/v2/os_test_final/results_p5")
PHASE5_SHARD = Path("Scripts/v2/os_test_final/shards_p5")
OUT          = Path("Scripts/v2/os_test_final/transition_metal_oxidation_states_FINAL.csv")

def scan(shard_dir, results_dir, label):
    rows = {}  # zip_path -> data line
    missing, short = [], []
    for sf in sorted(shard_dir.glob("shard_*.txt")):
        rf = results_dir / f"{sf.stem}.csv"
        if not rf.exists() or rf.stat().st_size == 0:
            missing.append(sf.stem)
            continue
        expected = sum(1 for _ in sf.open()) + 1
        with rf.open() as fh:
            lines = fh.readlines()
        if len(lines) < expected:
            short.append((sf.stem, len(lines), expected))
            continue
        for line in lines[1:]:
            line = line.rstrip("\n")
            if not line:
                continue
            zp = line.split(",", 1)[0]
            rows[zp] = line
    print(f"[{label}]  shards considered: {len(list(shard_dir.glob('shard_*.txt'))):,}  "
          f"rows collected: {len(rows):,}  missing: {len(missing)}  short: {len(short)}")
    return rows, missing, short

p2_rows, p2_missing, p2_short = scan(PHASE2_SHARD, PHASE2_RES, "phase-2")
p3_rows, p3_missing, p3_short = scan(PHASE3_SHARD, PHASE3_RES, "phase-3")
p4_rows, p4_missing, p4_short = scan(PHASE4_SHARD, PHASE4_RES, "phase-4")
p5_rows, p5_missing, p5_short = scan(PHASE5_SHARD, PHASE5_RES, "phase-5")

# Merge: later phases win on overlap (each phase is a retry of prior failures)
merged = dict(p2_rows)
merged.update(p3_rows)
merged.update(p4_rows)
merged.update(p5_rows)

with OUT.open("w") as fh:
    fh.write("zip_path,reactant_metal_oxidation_states,product_metal_oxidation_states\n")
    for zp in sorted(merged):
        fh.write(merged[zp] + "\n")

print(f"\nwrote: {OUT}")
print(f"unique zip_path rows: {len(merged):,}")
print(f"target (dedup_tm_picks): 181,450")
print(f"coverage: {100*len(merged)/181450:.1f}%")

if p2_missing or p3_missing or p2_short or p3_short:
    print()
    if p2_missing[:5]:
        print(f"phase-2 missing examples: {p2_missing[:5]}  ({len(p2_missing)} total)")
    if p3_missing[:5]:
        print(f"phase-3 missing examples: {p3_missing[:5]}  ({len(p3_missing)} total)")
    if p2_short[:3]:
        print(f"phase-2 short examples:   {p2_short[:3]}")
    if p3_short[:3]:
        print(f"phase-3 short examples:   {p3_short[:3]}")

sys.exit(0 if len(merged) == 181450 else 1)
