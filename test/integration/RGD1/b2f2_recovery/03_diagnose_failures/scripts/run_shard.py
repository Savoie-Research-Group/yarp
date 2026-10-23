"""Step 03 driver: one shard of RUN_DIR/rows.csv (from make_run.py).

The rows are step 02 reactions that failed with clean parses on both sides
(rxn_clean_fail.csv). For each reactant, every row goes through these stages
and stops at the first that recovers its RGD1 reaction (exact reaction hash):

  1 exhaustive  Only if the reactant keeps more than one BEM:
                bnfn_exhaustive(R, 2) (see bnfn_exhaustive.py) -> exhaustive_pass.
                With one kept BEM, bnfn_exhaustive is exactly bnfn, which
                already failed in step 02, so these rows go straight to stage 2.
  2 reorder     K random relabellings of the atom maps (seeded per reactant),
                each parsed again and run through bnfn_exhaustive; the RGD1
                product is relabelled the same way -> reorder_pass (with k).
  3 none        -> persist_fail.

Parsing, reaction hashes and the bnfn call (hashes=set(), hash_filter=False)
are step 02's (imported from ../../02_batch_enum/scripts/run_shard.py).
Every row also gets structural flags: n_adj_break / n_adj_form, and
lp_insertion (an atom gains >= 2 bonded neighbours, loses none, and is not
a radical in any kept reactant BEM).

Usage:
  python scripts/run_shard.py --rows RUN_DIR/rows.csv --shard 3 --out RUN_DIR/out [--k 3]
Writes OUT/shard_0003/{rows,reactants}.csv and meta.json, then a DONE marker
last. A shard that already has DONE is skipped unless --rerun.
"""
import argparse
import importlib.util
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bnfn_exhaustive import bnfn_exhaustive
from yarp.reaction.reaction import reaction
from yarp.yarpecule.lewis.bem_score import return_formals

# Step 02's driver, loaded under its own name (this file is also run_shard.py)
_spec = importlib.util.spec_from_file_location(
    "step02_run_shard", Path(__file__).resolve().parents[2] / "02_batch_enum" / "scripts" / "run_shard.py")
step02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step02)
SEED, Timeout, time_limit, quiet = step02.SEED, step02.Timeout, step02.time_limit, step02.quiet
parse_one, fc_index, git_state, light_rxn_hash = step02.parse_one, step02.fc_index, step02.git_state, step02.light_rxn_hash

MAP = re.compile(r":(\d+)\]")


def relabel(smi, sigma):
    return MAP.sub(lambda m: f":{sigma[int(m.group(1))]}]", smi)


def radicals_of(b):
    return {i for i in range(len(b)) if b[i][i] % 2 == 1}


def sum_fc(y, b):
    return int(sum(abs(v) for v in return_formals(b, y.elements)))


def structural(R, P_aligned):
    d = np.asarray(P_aligned.adj_mat) - np.asarray(R.adj_mat)
    gain, loss = (d > 0).sum(axis=1), (d < 0).sum(axis=1)
    rad_any = set().union(*(radicals_of(b) for b in R.bond_mats))
    lp = [i for i in range(len(R)) if gain[i] >= 2 and loss[i] == 0 and i not in rad_any]
    return {"n_adj_break": int((d < 0).sum() // 2), "n_adj_form": int((d > 0).sum() // 2),
            "lp_insertion": bool(lp),
            "lp_atoms": " ".join(f"{R.elements[i].upper()}{R.atom_info[i]['atom_map']}" for i in lp)}


def exhaustive(R):
    """{reaction hash: sorted source BEM indices} over bnfn_exhaustive(R, 2)."""
    prods, _ = quiet(lambda: list(bnfn_exhaustive(R, 2, hashes=set(), hash_filter=False, verbose=False)))
    found = {}
    for p in prods:
        found.setdefault(light_rxn_hash(R, p), set()).add(p.source_bem)
    return {h: sorted(s) for h, s in found.items()}, len(prods)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=3, help="number of atom-map relabellings")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per bnfn_exhaustive call")
    ap.add_argument("--rerun", action="store_true", help="rerun a shard that already has DONE")
    a = ap.parse_args()

    rows = pd.read_csv(a.rows, low_memory=False)
    rows = rows[rows.shard == a.shard]
    out = os.path.join(a.out, f"shard_{a.shard:04d}")
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "DONE")):
        if not a.rerun:
            print(f"shard_{a.shard:04d} already DONE; skipping (use --rerun to redo it)", file=sys.stderr)
            return
        os.remove(os.path.join(out, "DONE"))
    t_start = time.time()
    meta = {"args": vars(a), "n_rows": len(rows), "n_reactants": int(rows.reactant.nunique()),
            "started": time.strftime("%F %T"), **git_state()}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)

    row_out, rct_out = [], []
    groups = list(rows.groupby("reactant", sort=False))
    for n, (r_smi, grp) in enumerate(groups):
        t0 = time.perf_counter()
        first = grp.reactant_first_reaction.iloc[0]
        R, _ = parse_one(r_smi)
        rr = {"reactant": r_smi, "first_reaction": first, "n_rows": len(grp)}
        todo, rec = {}, {}
        for _, row in grp.iterrows():
            o = {"reaction": row.reaction}
            P, _ = parse_one(row["product"]) if R is not None else (None, None)
            if R is None or P is None:
                o["outcome"] = "parse_fail_now"
            else:
                try:
                    rx = quiet(reaction, R, P)[0]
                    o.update(structural(R, rx.product.graph))
                    todo[row.reaction] = rx.hash
                except Exception as e:  # noqa: BLE001
                    o["outcome"], o["error"] = "reaction_fail_now", f"{type(e).__name__}: {e}"[:300]
            rec[row.reaction] = o
            row_out.append(o)

        def open_rows():
            return [rid for rid in todo if "outcome" not in rec[rid]]

        def timed(label, fn):
            t1 = time.perf_counter()
            try:
                with time_limit(a.timeout):
                    res = fn()
            except Timeout:
                rr[f"timeout_{label}"] = True
                return None
            rr[f"t_{label}"] = round(time.perf_counter() - t1, 3)
            return res

        if R is not None and todo:
            bems = R.bond_mats
            fci = fc_index(R)
            rr.update(n_bem=len(bems), fc_ind=fci)
            for rid in todo:
                rec[rid].update(n_bem=len(bems), fc_ind=fci, exhaustive_tried=len(bems) > 1)

            # Stage 1: exhaustive bnfn, only worth running with more than one kept BEM
            if len(bems) > 1:
                res = timed("exhaustive", lambda: exhaustive(R))
                if res is not None:
                    found, rr["n_raw_exhaustive"] = res
                    for rid in open_rows():
                        src = found.get(todo[rid])
                        if src is not None:
                            rec[rid].update(outcome="exhaustive_pass",
                                            exhaustive_bems=" ".join(map(str, src)),
                                            exhaustive_found_by_fc_bem=fci in src,
                                            exhaustive_bem_sum_fc=" ".join(str(sum_fc(R, bems[j])) for j in src))

            # Stage 2: atom-map relabellings, each through bnfn_exhaustive
            maps = sorted(int(m) for m in MAP.findall(r_smi))
            for k in range(1, a.k + 1):
                if not open_rows():
                    break
                shuf = maps[:]
                random.Random(f"{SEED}-{first}-{k}").shuffle(shuf)
                sig = dict(zip(maps, shuf))
                Rv, _ = parse_one(relabel(r_smi, sig))
                if Rv is None:
                    rr[f"relabel{k}"] = "parse_fail"
                    continue
                rr[f"n_bem_relabel{k}"] = len(Rv.bond_mats)
                res = timed(f"relabel{k}", lambda: exhaustive(Rv))
                if res is None:
                    continue
                found_v, _ = res
                for rid in open_rows():
                    row = grp[grp.reaction == rid].iloc[0]
                    Pv, _ = parse_one(relabel(row["product"], sig))
                    if Pv is None:
                        continue
                    try:
                        h = quiet(reaction, Rv, Pv)[0].hash
                    except Exception:  # noqa: BLE001
                        continue
                    if h in found_v:
                        rec[rid].update(outcome="reorder_pass", reorder_k=k, reorder_n_bem=len(Rv.bond_mats),
                                        reorder_bems=" ".join(map(str, found_v[h])))
            for rid in open_rows():
                rec[rid]["outcome"] = "persist_fail"

        rr["t_total"] = round(time.perf_counter() - t0, 3)
        rct_out.append(rr)
        print(f"[shard {a.shard} {n + 1}/{len(groups)}] n_bem {rr.get('n_bem', '-')} "
              f"{[rec[r].get('outcome') for r in rec]} t {rr['t_total']:.1f} s", file=sys.stderr, flush=True)

    pd.DataFrame(row_out).to_csv(os.path.join(out, "rows.csv"), index=False)
    pd.DataFrame(rct_out).to_csv(os.path.join(out, "reactants.csv"), index=False)
    meta.update(finished=time.strftime("%F %T"), wall_seconds=round(time.time() - t_start, 1))
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    open(os.path.join(out, "DONE"), "w").write(meta["finished"] + "\n")


if __name__ == "__main__":
    main()
