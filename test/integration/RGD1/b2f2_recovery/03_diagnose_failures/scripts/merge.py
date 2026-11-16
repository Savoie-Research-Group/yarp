"""Step 03c: merge the shard outputs of one run, summarize them, and export
the three outcome tables.

Usage (from 03_diagnose_failures, in the yarp env):
    python scripts/merge.py RUN_DIR [--out-dir outputs/<RUN_DIR name>]
Reads RUN_DIR/rows.csv (the inputs) and RUN_DIR/out/shard_*/. Writes
RUN_DIR/merged/{rows,reactants}.csv and, in OUT_DIR:
    rxn_exhaustive_pass.csv  recovered by bnfn(exhaustive=True) (reactant keeps > 1 BEM)
    rxn_reorder_pass.csv     not recovered by that, but by bnfn(exhaustive=True)
                             after an atom-map relabelling (reorder_k says which)
    rxn_persist_fail.csv     neither (outcome persist_fail, or parse_fail_now /
                             reaction_fail_now if the row no longer parses)
    summary.txt              the readable summary that is also printed
Every input row is in exactly one of the three CSVs, with step 02's columns
(its category as category_02) and step 03's results.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd

STEP = Path(__file__).resolve().parents[1]  # 03_diagnose_failures
LEAD = ["reaction", "reactant", "product", "outcome", "category_02", "n_bem", "fc_ind", "exhaustive_tried",
        "reorder_k", "reorder_n_bem", "lp_insertion", "lp_atoms", "n_adj_break", "n_adj_form", "b_bemmin", "f_bemmin"]
FILES = {"rxn_exhaustive_pass": ["exhaustive_pass"], "rxn_reorder_pass": ["reorder_pass"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: 03_diagnose_failures/outputs/<run dir name>")
    a = ap.parse_args()
    out_dir = a.out_dir or STEP / "outputs" / Path(a.run).resolve().name
    inp = pd.read_csv(os.path.join(a.run, "rows.csv"), low_memory=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(inp.shard.unique())
    out = open(out_dir / "summary.txt", "w")

    def log(m=""):
        print(m)
        out.write(m + "\n")

    def frac(n, d):
        return f"{n:>6,} / {d:,} ({n / d:6.1%})" if d else f"{n:>6,} / 0"

    done = [s for s in shards if os.path.exists(os.path.join(a.run, "out", f"shard_{s:04d}", "DONE"))]
    missing = sorted(set(shards) - set(done))
    log(f"run: {os.path.abspath(a.run)}")
    log(f"shards: {len(done)}/{len(shards)} done" + (f"; missing: {missing}" if missing else ""))
    metas = [json.load(open(p)) for p in glob.glob(os.path.join(a.run, "out", "shard_*", "meta.json"))]
    for k in ("commit", "branch", "rdkit"):
        log(f"{k}: {sorted({str(m.get(k)) for m in metas})}")
    log(f"find_lewis.py sha256: {sorted({m['sha256']['yarp/yarpecule/lewis/find_lewis.py'][:12] for m in metas})}; "
        f"enum.py: {sorted({m['sha256']['yarp/reaction/enum.py'][:12] for m in metas})}; "
        f"args: {sorted({json.dumps({k: m['args'][k] for k in ('k', 'timeout')}) for m in metas})}")

    mdir = os.path.join(a.run, "merged")
    os.makedirs(mdir, exist_ok=True)
    tables = {}
    for name in ("rows", "reactants"):
        parts = [pd.read_csv(p, low_memory=False) for s in done
                 if os.path.exists(p := os.path.join(a.run, "out", f"shard_{s:04d}", f"{name}.csv"))
                 and os.path.getsize(p) > 1]
        tables[name] = pd.concat(parts, ignore_index=True)
        tables[name].to_csv(os.path.join(mdir, f"{name}.csv"), index=False)
    res, rcts = tables["rows"], tables["reactants"]

    run_inp = inp[inp.shard.isin(done)].drop(columns="shard")
    df = run_inp.merge(res, on="reaction", validate="one_to_one")
    df = df[[c for c in LEAD if c in df] + [c for c in df if c not in LEAD]]
    n = len(df)
    multi = df.n_bem.fillna(0) > 1

    log(f"\n== {run_inp.reactant.nunique():,} reactants -> {n:,} step 02 clean failures "
        f"({', '.join(f'{k} {v:,}' for k, v in df.category_02.value_counts().items())}) ==")
    log(f"  reactant keeps 1 BEM (straight to reordering): {frac(int((~multi).sum()), n)} rows")
    log(f"  reactant keeps > 1 BEM (exhaustive bnfn first): {frac(int(multi.sum()), n)} rows")

    log("\n== outcome ==")
    log(f"  {'':<16} {'all rows':>24} {'1 kept BEM':>14} {'> 1 kept BEM':>14}")
    for o in ("exhaustive_pass", "reorder_pass", "persist_fail", "parse_fail_now", "reaction_fail_now"):
        sel = df.outcome.eq(o)
        if o.endswith("_now") and not sel.any():
            continue
        log(f"  {o:<16} {frac(int(sel.sum()), n):>24} {int((sel & ~multi).sum()):>14,} {int((sel & multi).sum()):>14,}")

    ro = df[df.outcome.eq("reorder_pass")]
    if len(ro):
        log("\n== reorder passes ==")
        log("  by relabelling k: " + ", ".join(f"k={int(k)} {v:,}" for k, v in ro.reorder_k.value_counts().sort_index().items()))
        changed = ro.reorder_n_bem.ne(ro.n_bem)
        log(f"  relabelled reactant keeps a different number of BEMs: {frac(int(changed.sum()), len(ro))}")
        log(f"  base reactant keeps 1 BEM: {frac(int((ro.n_bem == 1).sum()), len(ro))}")
    pf = df[~df.outcome.isin(["exhaustive_pass", "reorder_pass"])]
    if len(pf):
        log("\n== persisting failures ==")
        lp = pf.lp_insertion.fillna(False).astype(bool)
        yb = (pf.b_bemmin == 2) & (pf.f_bemmin == 2)
        log(f"  lp_insertion (an atom forms >= 2 new bonds from a lone pair): {frac(int(lp.sum()), len(pf))}")
        log(f"  YARP b2f2 (fewest-change BEM pair), no lp_insertion: {frac(int((yb & ~lp).sum()), len(pf))}")
        log(f"  not YARP b2f2, no lp_insertion: {frac(int((~yb & ~lp).sum()), len(pf))}")
        top = pf.groupby(["n_adj_break", "n_adj_form"]).size().sort_values(ascending=False).head(6)
        log("  adjacency (breaks, forms): " + ", ".join(f"({int(b)}, {int(f)}) {v:,}" for (b, f), v in top.items()))

    log(f"\n== reactants: {len(rcts):,} ==")
    log(f"  t_total: sum {rcts.t_total.sum() / 3600:.2f} h, median {rcts.t_total.median():.1f} s, "
        f"max {rcts.t_total.max():.0f} s")
    touts = [c for c in rcts if c.startswith("timeout_")]
    log("  timeouts: " + (", ".join(f"{c[8:]} {int(rcts[c].fillna(False).astype(bool).sum())}" for c in touts) or "none"))

    log(f"\n== exported to {out_dir} ==")
    parts = {name: df[df.outcome.isin(o)] for name, o in FILES.items()}
    parts["rxn_persist_fail"] = df[~df.outcome.isin(["exhaustive_pass", "reorder_pass"])]
    for name, d in parts.items():
        d.to_csv(out_dir / f"{name}.csv", index=False)
        log(f"  {name + '.csv':<24} {frac(len(d), n)} rows")
    out.close()


if __name__ == "__main__":
    main()
