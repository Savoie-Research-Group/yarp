"""Step 02d (optional): compare the step 02 results of two runs, A and B
(e.g. two yarp versions on the same RGD1 rows).

Each run is given by its step 02 output folder (what merge.py writes). The
summary reports:
  1. recovered vs not recovered RGD1 reactions, per run and per reaction
     (recovered in both runs, A only, B only, neither). "Recovered" means the
     exact RGD1 reaction hash was found; species hashes are not used.
  2. SMILES parsing related failures: reactions not recovered whose reactant
     or product did not parse clean (it failed, fell back to RDKit, or printed
     a warning), i.e. the rows of rxn_parse_fail.csv. Per run, and how much
     the two runs overlap.
Failures with clean parses on both sides are step 03's input; compare those
with ../03_diagnose_failures/scripts/compare_runs.py.

Usage (from 02_batch_enum, in the yarp env):
    python scripts/compare_runs.py --a02 DIR --b02 DIR [--label-a NAME] [--label-b NAME] [--out-dir DIR]
DIRs are step 02 output folders (e.g. outputs/full). Writes to OUT_DIR
(default outputs/compare_<A folder>_vs_<B folder>):
    summary.txt        the readable summary that is also printed
    parse_fail_A.csv   A's SMILES parsing related failures
    parse_fail_B.csv   B's
CSV columns are prefixed A_ / B_ whatever the labels; parse_fail_in says
whether the reaction is a parsing related failure in both runs or in one.
"""
import argparse
from pathlib import Path

import pandas as pd

STEP = Path(__file__).resolve().parents[1]  # 02_batch_enum
FILES = {"rxn_match": "recovered", "rxn_clean_fail": "clean failure", "rxn_parse_fail": "parsing failure"}
COLS = ["reaction", "reactant", "product", "category", "status_R", "status_P", "rxn_error"]
OUTCOMES = list(FILES.values())  # every table lists outcomes in this order
PARSE_CATEGORY = {  # categories of parsing related failures, in table order
    "parse_fail_R": "reactant failed to parse (parse_fail_R)",
    "parse_fail_P": "product failed to parse (parse_fail_P)",
    "reaction_fail": "both parsed, reaction() failed (reaction_fail)",
    "missed": "parsed, not clean; RGD1 product not found (missed)",
    "species_only": "parsed, not clean; product found, other mapping (species_only)",
}

lines = []


def log(m=""):
    print(m)
    lines.append(m)


def pct(n, d):
    return f"{n:,} ({n / d:.1%})" if d else f"{n:,}"


def counts(s):
    """Outcome counts of a Series of outcome labels, e.g. 'recovered 12, clean failure 3'."""
    vc = s.value_counts()
    return ", ".join(f"{k} {vc[k]:,}" for k in OUTCOMES if k in vc) or "none"


def table(df):
    return "  " + df.to_string().replace("\n", "\n  ")


def describe(d):
    """The identifying lines of an output folder's summary.txt (run, commit, hashes, dirty files)."""
    p = d / "summary.txt"
    if not p.exists():
        return ["(no summary.txt)"]
    return [l for l in p.read_text().splitlines()[:8]
            if l.startswith(("run:", "commit:", "hashes.py", "dirty files"))]


def load_run(d):
    """One row per RGD1 reaction, with its outcome: recovered / clean failure / parsing failure."""
    parts = [pd.read_csv(d / f"{f}.csv", usecols=lambda c: c in COLS, low_memory=False).assign(file=o)
             for f, o in FILES.items()]
    df = pd.concat(parts, ignore_index=True).set_index("reaction", verify_integrity=True)
    df = df.reindex(columns=COLS[1:] + ["file"])
    outcome = pd.Series("parsing failure", index=df.index)  # later assignments win
    outcome[df.status_R.eq("clean") & df.status_P.eq("clean")] = "clean failure"
    outcome[df.category.eq("recovered_rxn")] = "recovered"
    df["outcome"] = outcome
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for x in ("a", "b"):
        ap.add_argument(f"--{x}02", type=Path, required=True, help=f"run {x.upper()}'s step 02 output folder")
        ap.add_argument(f"--label-{x}", default=x.upper(), help=f"name for run {x.upper()} in the summary")
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    LA, LB = a.label_a, a.label_b
    out_dir = a.out_dir or STEP / "outputs" / f"compare_{a.a02.resolve().name}_vs_{a.b02.resolve().name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    A, B = load_run(a.a02), load_run(a.b02)
    J = pd.concat([A.add_prefix("A_"), B.add_prefix("B_")], axis=1)
    w = max(len("in neither run"), len(LA) + 8, len(LB) + 8)  # label width for the 'in ... only' lines

    log(f"Step 02 comparison: {LA} vs {LB}")
    log("=" * len(lines[-1]))
    for lab, x, d in ((LA, "A", a.a02), (LB, "B", a.b02)):
        log(f"\n{x} = {lab}")
        log(f"  step 02 outputs: {d.resolve()}")
        for l in describe(d):
            log(f"    {l}")
    log('\n"Recovered" = the exact RGD1 reaction hash is among bnfn\'s products (species hashes are not used).')

    log("\n== Checks ==")
    both = J.A_outcome.notna() & J.B_outcome.notna()
    log(f"  RGD1 reactions: {LA} {len(A):,}, {LB} {len(B):,}, in both {int(both.sum()):,}")
    for lab, df in ((LA, A), (LB, B)):
        log(f"  {lab}: rows whose output file disagrees with their category and parse status: "
            f"{int(df.file.ne(df.outcome).sum()):,}")
    diff = both & (J.A_status_R.ne(J.B_status_R) | J.A_status_P.ne(J.B_status_P))
    log(f"  reactions whose parse status (reactant or product) differs between the runs: {int(diff.sum()):,}")

    # ---------------------------------------------------------------------- 1
    log("\n== 1. Recovered vs not recovered ==")
    rows = {}
    for lab, df in ((LA, A), (LB, B)):
        n, o = len(df), df.outcome
        scored = df.category.isin(["recovered_rxn", "species_only", "missed"])
        rows[lab] = {
            "RGD1 b2f2 reactions": f"{n:,}",
            "recovered": pct(int(o.eq("recovered").sum()), n),
            "not recovered": pct(int(o.ne("recovered").sum()), n),
            "  both endpoints parse clean (step 03's input)": pct(int(o.eq("clean failure").sum()), n),
            "  SMILES parsing related (section 2)": pct(int(o.eq("parsing failure").sum()), n),
            "recovered, of scored reactions*": f"{int(o.eq('recovered').sum()):,} / {int(scored.sum()):,} "
                                               f"({o.eq('recovered').sum() / scored.sum():.2%})",
        }
    log(table(pd.DataFrame(rows)))
    log("  * scored: both endpoints parsed, reaction() built and bnfn finished (the rate quoted in step 02's summary)")
    ra, rb = J.A_outcome.eq("recovered") & both, J.B_outcome.eq("recovered") & both
    log("\n  Per reaction:")
    log(f"    {'in both runs':<{w}} {int((ra & rb).sum()):>9,}   recovered by both")
    log(f"    {'in ' + LA + ' only':<{w}} {int((ra & ~rb).sum()):>9,}   {LB}'s outcome for these: {counts(J.B_outcome[ra & ~rb])}")
    log(f"    {'in ' + LB + ' only':<{w}} {int((rb & ~ra).sum()):>9,}   {LA}'s outcome for these: {counts(J.A_outcome[rb & ~ra])}")
    log(f"    {'in neither run':<{w}} {int((both & ~ra & ~rb).sum()):>9,}")

    # ---------------------------------------------------------------------- 2
    log("\n== 2. SMILES parsing related failures ==")
    log("  Not recovered, and the reactant or product did not parse clean: it failed to parse, fell back")
    log("  to RDKit, or parsed but printed a warning. Step 03 never sees these.")
    rows = {}
    for lab, df in ((LA, A), (LB, B)):
        pf = df[df.outcome.eq("parsing failure")]
        rows[lab] = {"parsing related failures": len(pf)}
        rows[lab].update({f"  {v}": int(pf.category.eq(k).sum()) for k, v in PARSE_CATEGORY.items()})
        for k in sorted(set(pf.category) - set(PARSE_CATEGORY)):
            rows[lab][f"  {k}"] = int(pf.category.eq(k).sum())
    log(table(pd.DataFrame(rows).fillna(0).astype(int).map(lambda v: f"{v:,}")))
    pa, pb = J.A_outcome.eq("parsing failure") & both, J.B_outcome.eq("parsing failure") & both
    log("\n  Per reaction:")
    log(f"    {'in both runs':<{w}} {int((pa & pb).sum()):>9,}")
    log(f"    {'in ' + LA + ' only':<{w}} {int((pa & ~pb).sum()):>9,}   {LB}'s outcome for these: {counts(J.B_outcome[pa & ~pb])}")
    log(f"    {'in ' + LB + ' only':<{w}} {int((pb & ~pa).sum()):>9,}   {LA}'s outcome for these: {counts(J.A_outcome[pb & ~pa])}")

    sel = pa | pb
    P = J[sel].assign(reactant=J.A_reactant.combine_first(J.B_reactant), product=J.A_product.combine_first(J.B_product),
                      parse_fail_in=["both" if x and y else "A only" if x else "B only" for x, y in zip(pa[sel], pb[sel])])
    P = P[["reactant", "product", "parse_fail_in", "A_category", "B_category", "A_status_R", "A_status_P",
           "B_status_R", "B_status_P", "A_rxn_error", "B_rxn_error"]]
    for x, s in (("A", pa), ("B", pb)):
        f = out_dir / f"parse_fail_{x}.csv"
        P[s[P.index]].to_csv(f, index_label="reaction")
        log(f"  -> list: {f} ({int(s.sum()):,} rows)")

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
