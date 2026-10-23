"""Step 02 driver: one shard of RUN_DIR/rows.csv (from make_run.py).

For every RGD1 b2f2 row in the shard: parse both endpoints, run one raw bnfn
b2f2 cycle per reactant, and check whether the RGD1 reaction is among the
products. Usually run by submit.py, through the cluster template or --local.

Parsing, per unique SMILES:
  yarpecule(smi, mode="yarp", canon=False, strict=True); if that fails, the
  default (strict=False) parse, recording whether the RDKit fallback fired.
  Status: clean / warn (printed something) / fallback (flagged: atom maps not
  trusted) / fail. Exceptions are unwrapped from strict mode's TypeError and
  located by their innermost yarp frame. Correctness check: parsed
  (atom_map -> element) vs the SMILES text only (no RDKit reference).

Recovery, per unique reactant SMILES:
  products = list(bnfn(R, 2, hashes=set(), hash_filter=False)): no
  self-filter, so degenerate rearrangements count, and no
  filter_enum_products. Product species hash = p.hash. Reaction hash =
  hashes.reaction_hash on a stand-in exposing .graph/.hash (bnfn products
  share R's index order); --rxn-check K random products per reactant also
  get a full reaction() object and any mismatch is counted. Each RGD1 row:
  reaction(R, P) (aligns P by atom map), then
    recovered_rxn  RGD1 reaction hash among the bnfn reaction hashes
    species_only   RGD1 product species found, only with another mapping
    missed         neither (self_species=True marks a product that is the
                   reactant's own species)
    parse_fail_R / parse_fail_P / reaction_fail / timeout / bnfn_error
  plus YARP's (b, f) for the best kept reactant x product BEM pair, in
  bond-order units (bemmin).

Each reactant's bnfn work runs under --timeout seconds.

Usage:
  python scripts/run_shard.py --rows RUN_DIR/rows.csv --shard 7 --out RUN_DIR/out
Writes OUT/shard_0007/{rows,reactants,smiles}.csv and meta.json, then a DONE
marker last. A shard that already has DONE is skipped unless --rerun.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import random
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
import warnings
from itertools import product as iproduct
from types import SimpleNamespace

import numpy as np
import pandas as pd
from rdkit import rdBase

import yarp
from yarp.yarpecule.yarpecule import yarpecule  # first: avoids a circular import
from yarp.reaction.enum import bnfn
from yarp.reaction.reaction import reaction
from yarp.yarpecule.hashes import reaction_hash
from yarp.yarpecule.lewis.bem_score import return_formals

rdBase.LogToPythonStderr()

SEED = 20261020
YARP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(yarp.__file__)))
BRACKET = re.compile(r"\[([^\]]+)\]")
TEXT_ATOM = re.compile(r"^(\d*)([A-Z][a-z]?|[a-z])")
TEXT_MAP = re.compile(r":(\d+)$")


# ---------------------------------------------------------------- utilities
class Timeout(BaseException):
    """BaseException, so the parser's and this script's `except Exception` cannot swallow it."""


@contextlib.contextmanager
def time_limit(seconds):
    def handler(signum, frame):
        raise Timeout()
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def quiet(f, *a, **k):
    """Run f silently; return (result, captured warnings)."""
    with warnings.catch_warnings(record=True) as w, \
            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("always")
        res = f(*a, **k)
    return res, [f"{x.category.__name__}: {x.message}" for x in w]


def git_state():
    run = lambda *a: subprocess.run(["git", "-C", YARP_ROOT, *a], capture_output=True,
                                    text=True).stdout.strip()
    sha = lambda p: hashlib.sha256(open(os.path.join(YARP_ROOT, p), "rb").read()).hexdigest()
    return {"yarp_root": YARP_ROOT, "commit": run("rev-parse", "--short", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": run("status", "--short", "--untracked-files=no").splitlines(),
            "sha256": {p: sha(p) for p in ("yarp/yarpecule/hashes.py", "yarp/yarpecule/yarpecule.py",
                                           "yarp/yarpecule/lewis/find_lewis.py", "yarp/reaction/enum.py",
                                           "yarp/yarpecule/graph/smiles.py")},
            "rdkit": rdBase.rdkitVersion, "python": sys.version.split()[0],
            "host": socket.gethostname()}


# ------------------------------------------------------------------ parsing
def text_map_elements(smi):
    out = {}
    for tok in BRACKET.findall(smi):
        m, a = TEXT_MAP.search(tok), TEXT_ATOM.match(tok)
        if m and a:
            out[int(m.group(1))] = a.group(2).lower()
    return out


def frames(exc):
    tb = traceback.extract_tb(exc.__traceback__)
    fmt = lambda f: (f"{os.path.relpath(f.filename, YARP_ROOT) if f.filename.startswith(YARP_ROOT) else os.path.basename(f.filename)}"
                     f":{f.lineno}:{f.name}")
    yarp_f = [f for f in tb if f.filename.startswith(os.path.join(YARP_ROOT, "yarp"))]
    return (fmt(tb[-1]) if tb else ""), (fmt(yarp_f[-1]) if yarp_f else "")


def root_exc(exc):
    if isinstance(exc, TypeError) and "expects either an xyz file" in str(exc) and exc.__context__:
        return exc.__context__
    return exc


def captured(fn):
    out, err = io.StringIO(), io.StringIO()
    res, exc = None, None
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as w, \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        warnings.simplefilter("always")
        try:
            res = fn()
        except Exception as e:  # noqa: BLE001 -- recording every failure is the point
            exc = e
    lines = lambda s: [l for l in s.getvalue().splitlines() if l.strip()]
    return res, exc, lines(out), lines(err), [f"{x.category.__name__}: {x.message}" for x in w], \
        time.perf_counter() - t0


def parse_one(smi, canon=False):
    rec = {}
    y, exc, so, se, wn, dt = captured(lambda: yarpecule(smi, mode="yarp", canon=canon, strict=True))
    rec.update(strict_ok=exc is None, strict_seconds=round(dt, 3), strict_stdout=" | ".join(so),
               strict_stderr=" | ".join(se), strict_warnings=" | ".join(wn))
    if exc is not None:
        e = root_exc(exc)
        inner, yf = frames(e)
        rec.update(strict_exc_type=type(e).__name__, strict_exc_msg=str(e)[:300],
                   strict_exc_frame=inner, strict_exc_yarp_frame=yf, strict_exc_wrapped=e is not exc)
        y, exc2, so2, se2, wn2, dt2 = captured(lambda: yarpecule(smi, mode="yarp", canon=canon))
        rec.update(default_ok=exc2 is None, default_seconds=round(dt2, 3),
                   fallback_fired=any("falling back to RDKit" in l for l in so2),
                   default_stdout=" | ".join(so2), default_stderr=" | ".join(se2),
                   default_warnings=" | ".join(wn2))
        if exc2 is not None:
            rec.update(default_exc_type=type(exc2).__name__, default_exc_msg=str(exc2)[:300],
                       default_exc_frame=frames(exc2)[0])
    if y is not None:
        maps = {y.atom_info[i]["atom_map"]: y.elements[i] for i in range(len(y))}
        rec.update(n_atoms=len(y), q=y.q, n_bem=len(y.bond_mats), map_text_ok=maps == text_map_elements(smi))
    if rec["strict_ok"]:
        st = "warn" if (rec["strict_stdout"] or rec["strict_stderr"] or rec["strict_warnings"]) else "clean"
    else:
        st = "fallback" if rec.get("fallback_fired") and rec.get("default_ok") else "fail"
    rec["status"] = st
    return y, rec


# ----------------------------------------------------------------- analysis
def changes(r, p):
    d = np.triu(np.asarray(p, dtype=int) - np.asarray(r, dtype=int), k=1)
    return int(-d[d < 0].sum()), int(d[d > 0].sum())


def bemmin(R, P):
    r_maps = [R.atom_info[i]["atom_map"] for i in range(len(R))]
    p_idx = {P.atom_info[i]["atom_map"]: i for i in range(len(P))}
    if sorted(r_maps) != sorted(p_idx):
        raise ValueError("reactant and product atom maps differ")
    perm = [p_idx[m] for m in r_maps]
    al = lambda m: np.asarray(m)[np.ix_(perm, perm)]
    return min((changes(rb, al(pb)) for rb, pb in iproduct(R.bond_mats, P.bond_mats)), key=sum)


def light_rxn_hash(R, p):
    end = lambda y: SimpleNamespace(graph=y, hash=y.hash)
    return reaction_hash(SimpleNamespace(reactant=end(R), product=end(p)))


def fc_index(y):
    fc = [sum(abs(v) for v in return_formals(b, y.elements)) for b in y.bond_mats]
    return fc.index(min(fc))


def run_bnfn(R, self_filter):
    hashes = {R.hash} if self_filter else set()
    prods, _ = quiet(lambda: list(bnfn(R, 2, hashes=hashes, hash_filter=False, verbose=False)))
    return prods


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--shard", type=int, default=None, help="1-based shard id; default: all rows")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rxn-check", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=900, help="seconds per reactant")
    ap.add_argument("--self-filter", action="store_true", help="hashes={R.hash}, as enumerate_products")
    ap.add_argument("--rerun", action="store_true", help="rerun a shard that already has DONE")
    a = ap.parse_args()

    rows = pd.read_csv(a.rows)
    if a.shard is not None:
        rows = rows[rows.shard == a.shard]
    tag = f"shard_{a.shard:04d}" if a.shard is not None else "all"
    out = os.path.join(a.out, tag)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "DONE")):
        if not a.rerun:
            print(f"{tag} already DONE; skipping (use --rerun to redo it)", file=sys.stderr)
            return
        os.remove(os.path.join(out, "DONE"))
    t_start = time.time()
    meta = {"args": vars(a), "n_rows": len(rows), "n_reactants": int(rows.reactant.nunique()),
            "started": time.strftime("%F %T"), **git_state()}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    rng = random.Random(f"{SEED}-{tag}")

    parsed = {}

    def parse(smi, role):
        if smi not in parsed:
            y, rec = parse_one(smi)
            parsed[smi] = (y, {"smiles": smi, "role": role, **rec})
        return parsed[smi]

    row_out, rct_out = [], []
    groups = list(rows.groupby("reactant", sort=False))
    for n, (r_smi, grp) in enumerate(groups):
        t0 = time.perf_counter()
        R, r_rec = parse(r_smi, "R")
        rr = {"reactant": r_smi, "first_reaction": grp.reaction.iloc[0], "n_rows": len(grp),
              "n_atoms": len(BRACKET.findall(r_smi)), "status_R": r_rec["status"],
              "t_parse_R": r_rec["strict_seconds"] + r_rec.get("default_seconds", 0)}
        species, rxns, prods, outcome = set(), set(), [], "parse_fail_R"
        if R is not None:
            rr.update(n_bem_R=len(R.bond_mats), fc_ind=fc_index(R))
            try:
                with time_limit(a.timeout):
                    t1 = time.perf_counter()
                    prods = run_bnfn(R, a.self_filter)
                    rr["t_bnfn"] = round(time.perf_counter() - t1, 3)
                    t1 = time.perf_counter()
                    light = [light_rxn_hash(R, p) for p in prods]
                    rr["t_hash"] = round(time.perf_counter() - t1, 3)
                    species, rxns = {p.hash for p in prods}, set(light)
                    if a.rxn_check > 0 and prods:
                        pick = rng.sample(range(len(prods)), min(a.rxn_check, len(prods)))
                        rr["n_rxn_check"] = len(pick)
                        rr["rxn_check_mismatch"] = sum(quiet(reaction, R, prods[k])[0].hash != light[k]
                                                       for k in pick)
                outcome = "ok"
            except Timeout:
                outcome = "timeout"
            except Exception as e:  # noqa: BLE001
                outcome, rr["bnfn_error"] = "bnfn_error", f"{type(e).__name__}: {e}"[:300]
            rr.update(n_raw=len(prods), n_species=len(species), n_rxn=len(rxns))
        rr["outcome"] = outcome

        for _, row in grp.iterrows():
            o = {"reaction": row.reaction, "reactant_first_reaction": rr["first_reaction"],
                 "set": row.get("set"), "status_R": r_rec["status"]}
            P, p_rec = parse(row["product"], "P")
            o["status_P"] = p_rec["status"]
            if R is None:
                o["category"] = "parse_fail_R"
            elif P is None:
                o["category"] = "parse_fail_P"
            elif outcome != "ok":
                o["category"] = outcome
            else:
                try:
                    o["b_bemmin"], o["f_bemmin"] = bemmin(R, P)
                except Exception as e:  # noqa: BLE001
                    o["bemmin_error"] = f"{type(e).__name__}: {e}"[:300]
                try:
                    rx, wn = quiet(reaction, R, P)
                    o["rxn_warnings"] = " | ".join(wn)
                except Exception as e:  # noqa: BLE001
                    o["category"], o["rxn_error"] = "reaction_fail", f"{type(e).__name__}: {e}"[:300]
                else:
                    o["self_species"] = P.hash == R.hash
                    o["found_species"] = P.hash in species
                    o["found_rxn"] = rx.hash in rxns
                    o["category"] = ("recovered_rxn" if o["found_rxn"] else
                                     "species_only" if o["found_species"] else "missed")
            row_out.append(o)

        rr["t_total"] = round(time.perf_counter() - t0, 3)
        rct_out.append(rr)
        print(f"[{tag} {n + 1}/{len(groups)}] {outcome} raw {rr.get('n_raw', '-')} "
              f"t {rr['t_total']:.1f} s", file=sys.stderr, flush=True)

    pd.DataFrame(row_out).to_csv(os.path.join(out, "rows.csv"), index=False)
    pd.DataFrame(rct_out).to_csv(os.path.join(out, "reactants.csv"), index=False)
    pd.DataFrame([r for _, r in parsed.values()]).to_csv(os.path.join(out, "smiles.csv"), index=False)
    meta.update(finished=time.strftime("%F %T"), wall_seconds=round(time.time() - t_start, 1))
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    open(os.path.join(out, "DONE"), "w").write(meta["finished"] + "\n")


if __name__ == "__main__":
    main()
