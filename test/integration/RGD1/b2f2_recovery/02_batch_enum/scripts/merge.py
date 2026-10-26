"""Step 02c: merge the shard outputs of one run, summarize them, and export
the per-reaction results that step 03 reads.

Usage (from 02_batch_enum, in the yarp env):
    python scripts/merge.py RUN_DIR [--out-dir outputs/<RUN_DIR name>]
Reads RUN_DIR/rows.csv (the inputs) and RUN_DIR/out/shard_*/. Writes
RUN_DIR/merged/{rows,reactants,smiles}.csv (the shard outputs concatenated)
and, in OUT_DIR:
    rxn_match.csv        RGD1 reaction recovered by reaction hash
    rxn_clean_fail.csv   not recovered; reactant and product both parsed clean
    rxn_parse_fail.csv   not recovered; an endpoint parsed with a warning,
                         fell back to RDKit, or failed
    summary.txt          the readable summary that is also printed
Every RGD1 row of the run is in exactly one of the three CSVs, with its
reactant and product SMILES, its results, and its reactant's kept-BEM count
(n_bem_R) and time (t_total).
"""
import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd

STEP = Path(__file__).resolve().parents[1]  # 02_batch_enum
LEAD = ["reaction", "reactant", "product", "category", "status_R", "status_P", "found_rxn",
        "found_species", "self_species", "b_bemmin", "f_bemmin", "n_bem_R", "t_total"]


def split_rows(inp, rows, rcts):
    """One table per reaction (inputs + results + reactant columns), split into the three exports."""
    df = inp.drop(columns="shard").merge(rows.drop(columns="set", errors="ignore"), on="reaction",
                                         validate="one_to_one")
    df = df.join(rcts.set_index("reactant")[["n_bem_R", "t_total"]], on="reactant")
    df = df[LEAD + [c for c in df if c not in LEAD]]
    match = df.category.eq("recovered_rxn")
    clean = df.status_R.eq("clean") & df.status_P.eq("clean")
    return {"rxn_match": df[match], "rxn_clean_fail": df[~match & clean], "rxn_parse_fail": df[~match & ~clean]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: 02_batch_enum/outputs/<run dir name>")
    a = ap.parse_args()
    out_dir = a.out_dir or STEP / "outputs" / Path(a.run).resolve().name
    out_dir.mkdir(parents=True, exist_ok=True)
    inp = pd.read_csv(os.path.join(a.run, "rows.csv"))
    shards = sorted(inp.shard.unique())
    mdir = os.path.join(a.run, "merged")
    os.makedirs(mdir, exist_ok=True)
    out = open(out_dir / "summary.txt", "w")

    def log(m=""):
        print(m)
        out.write(m + "\n")

    done = [s for s in shards if os.path.exists(os.path.join(a.run, "out", f"shard_{s:04d}", "DONE"))]
    missing = sorted(set(shards) - set(done))
    log(f"run: {os.path.abspath(a.run)}")
    log(f"shards: {len(done)}/{len(shards)} done" + (f"; missing: {missing}" if missing else ""))
    metas = [json.load(open(p)) for p in glob.glob(os.path.join(a.run, "out", "shard_*", "meta.json"))]
    for k in ("commit", "branch", "rdkit"):
        log(f"{k}: {sorted({str(m.get(k)) for m in metas})}")
    log(f"hashes.py sha256: {sorted({m['sha256']['yarp/yarpecule/hashes.py'][:12] for m in metas})}; "
        f"find_lewis.py: {sorted({m['sha256']['yarp/yarpecule/lewis/find_lewis.py'][:12] for m in metas})}")
    log(f"dirty files: {sorted({len(m['dirty']) for m in metas})}; args: "
        f"{sorted({json.dumps({k: v for k, v in m['args'].items() if k in ('rxn_check', 'timeout', 'self_filter')}) for m in metas})}")
    wall = [m.get("wall_seconds") for m in metas if m.get("wall_seconds")]
    if wall:
        log(f"shard wall time (s): min {min(wall):.0f}, median {pd.Series(wall).median():.0f}, max {max(wall):.0f}; "
            f"sum {sum(wall) / 3600:.2f} h")

    tables = {}
    for name in ("rows", "reactants", "smiles"):
        parts = [pd.read_csv(p) for s in done
                 if os.path.exists(p := os.path.join(a.run, "out", f"shard_{s:04d}", f"{name}.csv"))
                 and os.path.getsize(p) > 1]
        if parts:
            tables[name] = pd.concat(parts, ignore_index=True)
            tables[name].to_csv(os.path.join(mdir, f"{name}.csv"), index=False)
    rows, rcts, smis = tables["rows"], tables["reactants"], tables["smiles"].drop_duplicates("smiles")

    log(f"\n== parsing: {len(smis)} unique SMILES ==")
    log(smis.groupby(["role", "status"]).size().unstack(fill_value=0).to_string())
    bad = smis[~smis.strict_ok]
    if len(bad):
        log("\nstrict-parse failures by exception and innermost yarp frame:")
        log(bad.groupby(["status", "strict_exc_type", "strict_exc_yarp_frame"]).size().to_string())
    warn = smis[smis.status == "warn"]
    if len(warn):
        msgs = Counter(m[:60] for c in ("strict_stdout", "strict_stderr", "strict_warnings")
                       for s in warn[c].fillna("") for m in s.split(" | ") if m)
        log("\nwarn-status messages (first 60 chars, top 10):")
        for m, c in msgs.most_common(10):
            log(f"  {c:6d}  {m}")
    mt = smis[smis.map_text_ok.notna()]
    log(f"map/element vs SMILES text: {int(mt.map_text_ok.astype(bool).sum())}/{len(mt)} agree")

    log(f"\n== RGD1 b2f2 rows: {len(rows)} (input {len(inp[inp.shard.isin(done)])}) ==")
    log(rows.category.value_counts().to_string())
    scored = rows[rows.category.isin(["recovered_rxn", "species_only", "missed"])]
    fb = scored.status_R.eq("fallback") | scored.status_P.eq("fallback")
    log(f"\nscored rows (both endpoints parsed, reaction() built, bnfn finished): {len(scored)} "
        f"({len(scored) / len(rows):.1%} of rows)")
    for tag, d in (("all scored", scored), ("no fallback endpoint", scored[~fb])):
        log(f"  {tag}: recovered by reaction hash {int(d.found_rxn.sum())}/{len(d)} "
            f"({d.found_rxn.mean():.2%}); species found {int(d.found_species.sum())}/{len(d)} "
            f"({d.found_species.mean():.2%})")
    yb = (scored.b_bemmin == 2) & (scored.f_bemmin == 2)
    log("\nscored rows: category x YARP bemmin b2f2")
    log(pd.crosstab(scored.category, yb.rename("yarp_b2f2")).to_string())
    log("\nscored rows not recovered: YARP bemmin (b, f) counts")
    nr = scored[scored.category != "recovered_rxn"]
    log(nr.groupby(["b_bemmin", "f_bemmin"], dropna=False).size().to_string() if len(nr) else "  none")
    log(f"self_species among scored: {int(scored.self_species.astype(bool).sum())} "
        f"(recovered {int((scored.self_species.astype(bool) & scored.found_rxn.astype(bool)).sum())})")

    log(f"\n== reactants: {len(rcts)} ==")
    log(rcts.outcome.value_counts().to_string())
    ok = rcts[rcts.outcome == "ok"]
    for c in ("t_bnfn", "t_total"):
        log(f"{c}: sum {ok[c].sum() / 3600:.2f} h, median {ok[c].median():.2f} s, max {ok[c].max():.1f} s")
    log(f"raw products: sum {int(ok.n_raw.sum())}, median {ok.n_raw.median():.0f}, max {int(ok.n_raw.max())}")
    if "n_rxn_check" in ok:
        log(f"full reaction() spot check: {int(ok.rxn_check_mismatch.sum())} mismatches in {int(ok.n_rxn_check.sum())}")

    run_inp = inp[inp.shard.isin(done)]
    parts = split_rows(run_inp, rows, rcts)
    n_rxn = sum(len(d) for d in parts.values())
    log(f"\n== exported to {out_dir} ==")
    log(f"{run_inp.reactant.nunique():,} reactants -> {n_rxn:,} RGD1 b2f2 reactions; "
        f"each reaction is one row in exactly one file:")
    for name, d in parts.items():
        d.to_csv(out_dir / f"{name}.csv", index=False)
        cats = ", ".join(f"{k} {v:,}" for k, v in d.category.value_counts().items())
        log(f"  {name + '.csv':<20} {len(d):>7,} / {n_rxn:,} reactions ({len(d) / n_rxn:6.1%})   {cats}")
    out.close()


if __name__ == "__main__":
    main()
