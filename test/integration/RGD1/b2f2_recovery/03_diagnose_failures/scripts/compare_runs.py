"""Step 03d (optional): compare the step 03 results of two runs, A and B
(e.g. two yarp versions on the same RGD1 rows).

Each run is given by its step 02 and step 03 output folders (what the two
merge.py scripts write). Every RGD1 b2f2 reaction whose endpoints both parse
clean gets one final stage per run:
  02 raw bnfn    recovered by step 02's single raw bnfn call
  03 exhaustive  missed by step 02, recovered by step 03's exhaustive bnfn
  03 reorder     missed by both, recovered after an atom-map relabelling
  persist fail   no attempt recovered it
"Recovered" means the exact RGD1 reaction hash was found. Reactions with an
endpoint that does not parse clean never reach step 03 and are left out.

The summary answers, for A vs B:
  Q1  which reactions no attempt recovers, in either run
  Q2  which step 02 failures of one run the other run recovers, and at which stage
  Q3  what the exhaustive and reordering trials add, and whether they fix the
      same reactions in both runs
  Q4  which step 03 reactions YARP does not count as b2f2 (bemmin, from step 02)

Usage (from 03_diagnose_failures, in the yarp env):
    python scripts/compare_runs.py --a02 DIR --a03 DIR --b02 DIR --b03 DIR \\
        [--label-a NAME] [--label-b NAME] [--out-dir DIR]
--a02/--b02 are step 02 output folders (e.g. ../02_batch_enum/outputs/full),
--a03/--b03 step 03 output folders (e.g. outputs/full). Writes to OUT_DIR
(default outputs/compare_<A step 03 folder>_vs_<B step 03 folder>):
    summary.txt                 the readable summary that is also printed
    q1_persist_fail_both.csv    reactions that no attempt recovers in either run
    q1_persist_fail_A_only.csv  reactions that no attempt recovers in A, but B recovers
    q1_persist_fail_B_only.csv  reactions that no attempt recovers in B, but A recovers
    q4_not_b2f2_A.csv           step 03 reactions whose YARP (b, f) is not (2, 2) in A
    q4_not_b2f2_B.csv           the same for B (not_b2f2_in says whether the other run agrees)
CSV columns are prefixed A_ / B_ whatever the labels. *_bf is YARP's (b, f);
*_lp_insertion and *_lp_atoms are empty where the reaction was not in that
run's step 03.
"""
import argparse
from pathlib import Path

import pandas as pd

STEP = Path(__file__).resolve().parents[1]  # 03_diagnose_failures
FILES02 = ("rxn_match", "rxn_clean_fail", "rxn_parse_fail")
FILES03 = ("rxn_exhaustive_pass", "rxn_reorder_pass", "rxn_persist_fail")
COLS02 = ["reaction", "reactant", "product", "category", "status_R", "status_P", "b_bemmin", "f_bemmin", "n_bem_R"]
COLS03 = ["reaction", "outcome", "n_bem", "reorder_k", "lp_insertion", "lp_atoms", "n_adj_break", "n_adj_form"]
STEP03_STAGE = {"exhaustive_pass": "03 exhaustive", "reorder_pass": "03 reorder", "persist_fail": "persist fail"}
STAGES = {  # every table lists stages in this order
    "02 raw bnfn": "recovered by step 02's single raw bnfn call",
    "03 exhaustive": "missed by step 02; recovered by step 03's exhaustive bnfn (one bnfn per kept BEM)",
    "03 reorder": "missed by both; recovered by exhaustive bnfn after an atom-map relabelling",
    "persist fail": "no attempt recovered it",
    "03 error": "step 03 could no longer parse it or build its reaction() (should not happen)",
    "03 missing": "a clean step 02 failure absent from the step 03 output (should not happen)",
    "not clean": "an endpoint did not parse clean, so step 03 never saw it",
}
ORDER = list(STAGES)
RECOVERED = ["02 raw bnfn", "03 exhaustive", "03 reorder"]
B2F2 = "(2, 2)"

lines = []


def log(m=""):
    print(m)
    lines.append(m)


def frac(n, d):
    return f"{n:,} / {d:,} ({n / d:.1%})" if d else f"{n:,} / 0"


def counts(s):
    """Stage counts of a Series of stage labels, e.g. '02 raw bnfn 12, persist fail 3'."""
    vc = s.value_counts()
    return ", ".join(f"{k} {vc[k]:,}" for k in ORDER if k in vc) or "none"


def as_bool(s):
    return s.map(lambda v: v if isinstance(v, bool) else {"True": True, "False": False}.get(v, False)).astype(bool)


def table(df):
    """Indent a DataFrame's text form, with thousands separators on integers."""
    df = df.map(lambda v: f"{v:,}" if isinstance(v, int) else v)
    return "  " + df.to_string().replace("\n", "\n  ")


def describe(d):
    """The identifying lines of an output folder's summary.txt (run, commit, find_lewis.py)."""
    p = d / "summary.txt"
    if not p.exists():
        return ["(no summary.txt)"]
    return [l for l in p.read_text().splitlines()[:8] if l.startswith(("run:", "commit:")) or "find_lewis.py" in l]


def load_run(d02, d03):
    """One row per RGD1 reaction: step 02 results, step 03 results, and the final stage."""
    s02 = pd.concat([pd.read_csv(d02 / f"{f}.csv", usecols=lambda c: c in COLS02, low_memory=False)
                     for f in FILES02], ignore_index=True)
    s03 = pd.concat([pd.read_csv(d03 / f"{f}.csv", low_memory=False) for f in FILES03], ignore_index=True)
    s03 = s03.reindex(columns=COLS03)
    df = s02.set_index("reaction", verify_integrity=True).join(
        s03.set_index("reaction", verify_integrity=True), how="outer")
    df["clean"] = df.status_R.eq("clean") & df.status_P.eq("clean")
    df["in_03"] = df.outcome.notna()
    df["lp_insertion"] = as_bool(df.lp_insertion).where(df.in_03)  # NaN: not in this run's step 03
    for c in ("n_bem_R", "n_bem", "reorder_k", "n_adj_break", "n_adj_form"):
        df[c] = df[c].astype("Int64")
    df["bf"] = [f"({int(b)}, {int(f)})" if pd.notna(b) and pd.notna(f) else "n/a"
                for b, f in zip(df.b_bemmin, df.f_bemmin)]
    stage = pd.Series("03 missing", index=df.index)  # later assignments win
    stage[df.in_03] = "03 error"
    known = df.outcome.isin(list(STEP03_STAGE))
    stage[known] = df.outcome[known].map(STEP03_STAGE)
    stage[~df.clean] = "not clean"
    stage[df.category.eq("recovered_rxn")] = "02 raw bnfn"
    df["stage"] = stage
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for x in ("a", "b"):
        ap.add_argument(f"--{x}02", type=Path, required=True, help=f"run {x.upper()}'s step 02 output folder")
        ap.add_argument(f"--{x}03", type=Path, required=True, help=f"run {x.upper()}'s step 03 output folder")
        ap.add_argument(f"--label-{x}", default=x.upper(), help=f"name for run {x.upper()} in the summary")
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    LA, LB = a.label_a, a.label_b
    out_dir = a.out_dir or STEP / "outputs" / f"compare_{a.a03.resolve().name}_vs_{a.b03.resolve().name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    A, B = load_run(a.a02, a.a03), load_run(a.b02, a.b03)
    J = pd.concat([A.add_prefix("A_"), B.add_prefix("B_")], axis=1)

    log(f"Step 03 comparison: {LA} vs {LB}")
    log("=" * len(lines[-1]))
    for lab, x, d02, d03 in ((LA, "A", a.a02, a.a03), (LB, "B", a.b02, a.b03)):
        log(f"\n{x} = {lab}")
        for step, d in (("step 02", d02), ("step 03", d03)):
            log(f"  {step} outputs: {d.resolve()}")
            for l in describe(d):
                log(f"    {l}")
    log("\nFinal stage of a reaction in a run (the first attempt that recovers its exact RGD1 reaction hash):")
    for k, v in STAGES.items():
        log(f"  {k:<14} {v}")

    # ------------------------------------------------------------ consistency
    log("\n== Checks ==")
    both = J.A_stage.notna() & J.B_stage.notna()
    log(f"  RGD1 reactions: {LA} {len(A):,}, {LB} {len(B):,}, in both {int(both.sum()):,}")
    parse_diff = both & (J.A_status_R.ne(J.B_status_R) | J.A_status_P.ne(J.B_status_P))
    log(f"  parse status (reactant or product) differs between the runs: {int(parse_diff.sum()):,} reactions")
    for lab, x, df in ((LA, "A", A), (LB, "B", B)):
        bad = df.in_03 & ~(df.clean & df.category.ne("recovered_rxn"))
        log(f"  {lab}: step 03 rows that step 02 recovered or that did not parse clean: {int(bad.sum()):,}; "
            f"'03 error' {int(df.stage.eq('03 error').sum()):,}; '03 missing' {int(df.stage.eq('03 missing').sum()):,}")
    U = J[both & J.A_clean.eq(True) & J.B_clean.eq(True)]
    n = len(U)
    log(f"  reactions that parse clean in both runs: {n:,}. Everything below is about these.")

    log(f"\n== Overview: final stage in {LA} (rows) vs {LB} (columns) ==")
    ct = pd.crosstab(U.A_stage, U.B_stage, margins=True, margins_name="total")
    ct = ct.reindex(index=[k for k in ORDER if k in ct.index] + ["total"],
                    columns=[k for k in ORDER if k in ct.columns] + ["total"], fill_value=0)
    ct.index.name, ct.columns.name = f"{LA} \\ {LB}", None
    log(table(ct.astype(int).astype(object)))
    if ORDER[0] in ct.index and "03 reorder" in ct.columns:
        log(f"  How to read a cell: {int(ct.loc[ORDER[0], '03 reorder']):,} reactions are recovered by {LA}'s step 02 "
            f"(raw bnfn), but by {LB} only after atom reordering.")

    # --------------------------------------------------------------------- Q1
    log("\n== Q1. Persistent failures: no attempt recovers them ==")
    pa, pb = U.A_stage.eq("persist fail"), U.B_stage.eq("persist fail")
    both_pf = U[pa & pb]
    w = max(len("in either run"), len(LA) + 8, len(LB) + 8)  # label width for the 'in ... only' lines
    log(f"  {'in both runs':<{w}} {len(both_pf):>7,} of {n:,} clean reactions ({len(both_pf) / n:.1%})")
    log(f"  {'in ' + LA + ' only':<{w}} {int((pa & ~pb).sum()):>7,}   {LB}'s stage for these: {counts(U.B_stage[pa & ~pb])}")
    log(f"  {'in ' + LB + ' only':<{w}} {int((pb & ~pa).sum()):>7,}   {LA}'s stage for these: {counts(U.A_stage[pb & ~pa])}")
    if len(both_pf):
        log(f"\n  The {len(both_pf):,} persistent in both, by step 03's flags (each run's own BEMs):")
        rows = {}
        for lab, x in ((LA, "A"), (LB, "B")):
            lp, yb = both_pf[f"{x}_lp_insertion"].eq(True), both_pf[f"{x}_bf"].eq(B2F2)
            rows[lab] = {"lp_insertion (an atom forms >= 2 new bonds from a lone pair)": int(lp.sum()),
                         "no lp_insertion, YARP (b, f) = (2, 2)": int((~lp & yb).sum()),
                         "no lp_insertion, YARP (b, f) != (2, 2)": int((~lp & ~yb).sum())}
        log(table(pd.DataFrame(rows).astype(object)))
    for name, sel in (("both", pa & pb), ("A_only", pa & ~pb), ("B_only", pb & ~pa)):
        d = U[sel]
        q1 = d.assign(reactant=d.A_reactant, product=d.A_product,
                      n_adj_break=d.A_n_adj_break.combine_first(d.B_n_adj_break),  # adjacency only: same in both runs
                      n_adj_form=d.A_n_adj_form.combine_first(d.B_n_adj_form))[
            ["reactant", "product", "A_stage", "B_stage", "A_bf", "B_bf", "A_n_bem_R", "B_n_bem_R",
             "A_lp_insertion", "B_lp_insertion", "A_lp_atoms", "B_lp_atoms", "n_adj_break", "n_adj_form",
             "A_category", "B_category"]]
        q1 = q1.rename(columns={"A_category": "A_category_02", "B_category": "B_category_02"})
        f = out_dir / f"q1_persist_fail_{name}.csv"
        q1.to_csv(f, index_label="reaction")
        log(f"  -> list: {f} ({len(q1):,} rows)")

    # --------------------------------------------------------------------- Q2
    log("\n== Q2. Clean-parse step 02 failures of one run that the other run recovers ==")
    for fl, fx, ol, ox in ((LB, "B", LA, "A"), (LA, "A", LB, "B")):
        for what, sel in ((f"{fl}'s clean-parse step 02 failures (= {fl}'s step 03 input)",
                           U[f"{fx}_stage"].ne("02 raw bnfn")),
                          (f"{fl}'s persistent failures", U[f"{fx}_stage"].eq("persist fail"))):
            s = U.loc[sel, f"{ox}_stage"]
            log(f"\n  {what}: {len(s):,} reactions. Their final stage in {ol}:")
            vc = s.value_counts()
            for k in ORDER:
                if k in vc:
                    log(f"    {k:<14} {int(vc[k]):>7,}  ({vc[k] / len(s):.1%})")
            r = int(s.isin(RECOVERED).sum())
            log(f"    -> recovered by {ol} at some stage: {r:,} of {len(s):,} ({r / len(s):.1%})")

    # --------------------------------------------------------------------- Q3
    log("\n== Q3. What step 03's trials add ==")
    rows = {}
    for lab, x in ((LA, "A"), (LB, "B")):
        s = U[f"{x}_stage"]
        n03 = int(s.ne("02 raw bnfn").sum())
        ex, ro = int(s.eq("03 exhaustive").sum()), int(s.eq("03 reorder").sum())
        k = U.loc[s.eq("03 reorder"), f"{x}_reorder_k"].value_counts()
        r02 = int(s.eq("02 raw bnfn").sum())
        rows[lab] = {
            "step 03 input (clean-parse step 02 failures)": f"{n03:,}",
            "  reactant keeps > 1 BEM (exhaustive bnfn differs from bnfn)":
                frac(int((U[f"{x}_in_03"] & U[f"{x}_n_bem"].gt(1)).sum()), n03),
            "  recovered by exhaustive bnfn": frac(ex, n03),
            "  recovered after atom reordering": frac(ro, n03),
            "    at relabelling k = 1 / 2 / 3": " / ".join(f"{int(k.get(i, 0)):,}" for i in (1, 2, 3)),
            "  still failing (persist fail)": frac(int(s.eq("persist fail").sum()), n03),
            "recovered, step 02 only": frac(r02, n),
            "recovered, step 02 + exhaustive": frac(r02 + ex, n),
            "recovered, step 02 + exhaustive + reordering": frac(r02 + ex + ro, n),
        }
    log("  Denominators: clean reactions for the 'recovered' lines, the run's step 03 input for the indented lines.")
    log(table(pd.DataFrame(rows)))
    log("\n  Do the trials fix the same reactions in both runs?")
    for stage, name in (("03 exhaustive", "exhaustive bnfn"), ("03 reorder", "atom reordering")):
        ia, ib = U.A_stage.eq(stage), U.B_stage.eq(stage)
        log(f"\n  Recovered by {name}:")
        log(f"    {'in both runs':<{w}} {int((ia & ib).sum()):>7,}")
        log(f"    {'in ' + LA + ' only':<{w}} {int((ia & ~ib).sum()):>7,}   {LB}'s stage for these: {counts(U.B_stage[ia & ~ib])}")
        log(f"    {'in ' + LB + ' only':<{w}} {int((ib & ~ia).sum()):>7,}   {LA}'s stage for these: {counts(U.A_stage[ib & ~ia])}")

    # --------------------------------------------------------------------- Q4
    log("\n== Q4. Step 03 reactions whose YARP (b, f) is not RGD1's b2f2 ==")
    log("  YARP (b, f) = bond-order changes of the fewest-change kept reactant x product BEM pair (bemmin),")
    log("  from each run's step 02. It depends on the kept BEMs, so it can differ between runs.")
    V = U[U.A_stage.ne("02 raw bnfn") | U.B_stage.ne("02 raw bnfn")]
    na, nb = V.A_bf.ne(B2F2) & V.A_bf.ne("n/a"), V.B_bf.ne(B2F2) & V.B_bf.ne("n/a")
    log(f"  reactions in either run's step 03 input: {len(V):,}")
    log(f"  YARP (b, f) unknown (bemmin failed): {LA} {int(V.A_bf.eq('n/a').sum()):,}, {LB} {int(V.B_bf.eq('n/a').sum()):,}")
    log(f"  YARP (b, f) is not (2, 2):")
    log(f"    {'in both runs':<{w}} {int((na & nb).sum()):>7,}")
    log(f"    {'in ' + LA + ' only':<{w}} {int((na & ~nb).sum()):>7,}")
    log(f"    {'in ' + LB + ' only':<{w}} {int((nb & ~na).sum()):>7,}")
    log(f"    {'in either run':<{w}} {int((na | nb).sum()):>7,} of {len(V):,} ({(na | nb).sum() / len(V):.1%})")
    bfc = pd.concat({LA: V.A_bf[na].value_counts(), LB: V.B_bf[nb].value_counts()}, axis=1).fillna(0).astype(int)
    log(f"\n  YARP (b, f) of those not (2, 2), per run:")
    log(table(bfc.sort_index().astype(object)))
    st = pd.concat({LA: V.A_stage[na].value_counts(), LB: V.B_stage[nb].value_counts()},
                   axis=1).fillna(0).astype(int)
    st = st.reindex([k for k in ORDER if k in st.index])
    log("\n  Final stage, per run, of the reactions that are not (2, 2) in that run:")
    log(table(st.astype(object)))
    W = V[na | nb]
    q4 = W.assign(reactant=W.A_reactant, product=W.A_product,
                  not_b2f2_in=["both" if x and y else "A only" if x else "B only" for x, y in zip(na[na | nb], nb[na | nb])],
                  n_adj_break=W.A_n_adj_break.combine_first(W.B_n_adj_break),  # adjacency only: same in both runs
                  n_adj_form=W.A_n_adj_form.combine_first(W.B_n_adj_form))
    q4 = q4[["reactant", "product", "not_b2f2_in", "A_bf", "B_bf", "A_stage", "B_stage", "A_n_bem_R", "B_n_bem_R",
             "A_lp_insertion", "B_lp_insertion", "A_lp_atoms", "B_lp_atoms", "n_adj_break", "n_adj_form"]]
    for x, sel in (("A", na), ("B", nb)):
        f = out_dir / f"q4_not_b2f2_{x}.csv"
        q4[sel[q4.index]].to_csv(f, index_label="reaction")
        log(f"  -> list: {f} ({int(sel.sum()):,} rows)")

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
