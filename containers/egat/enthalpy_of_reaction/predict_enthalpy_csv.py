#!/usr/bin/env python3
"""
EGAT reaction-enthalpy (DH) inference from a CSV of atom-mapped reactions.

Input CSV may provide EITHER:
  * a single reaction column "reaction_smiles" (or reactions/reaction/AAM/rxn_smiles)
    formatted as "reactant>>product", OR
  * two columns "reactant" and "product".

Output CSV columns: reaction_smiles, reaction_enthalpy  (kcal/mol).
Self-contained: only needs src/rxn_egat + the checkpoint in models/.
"""

import argparse
import os
import sys
from pathlib import Path

# Thread env must be set before numpy/torch import. ~8 is the CPU sweet spot for
# these small graphs (more threads add overhead). Override with --threads.
_DEFAULT_THREADS = min(8, os.cpu_count() or 1)
if "EGAT_THREADS" in os.environ:
    _DEFAULT_THREADS = int(os.environ["EGAT_THREADS"])
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, str(_DEFAULT_THREADS))

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rxn_egat.predict import load_model, predict_reactions

_REACTION_COLS = ("reaction_smiles", "reactions", "reaction", "AAM", "rxn_smiles")


def _extract_pairs(df, reactions_col):
    """Return (pairs, reaction_strings) from the dataframe."""
    if "reactant" in df.columns and "product" in df.columns:
        reactants = df["reactant"].astype(str).tolist()
        products = df["product"].astype(str).tolist()
        pairs = list(zip(reactants, products))
        strings = [f"{r}>>{p}" for r, p in pairs]
        return pairs, strings

    col = reactions_col
    if col not in df.columns:
        for alt in _REACTION_COLS:
            if alt in df.columns:
                col = alt
                break
        else:
            raise ValueError(
                f"Input CSV needs '{reactions_col}' (reactant>>product) or "
                f"separate 'reactant'/'product' columns. Found: {list(df.columns)}"
            )
    strings = df[col].astype(str).tolist()
    pairs = []
    for s in strings:
        if ">>" not in s:
            raise ValueError(f"reaction '{s[:40]}...' is not in reactant>>product form")
        r, p = s.split(">>", 1)
        pairs.append((r.strip(), p.strip()))
    return pairs, strings


def main():
    p = argparse.ArgumentParser(description="EGAT reaction enthalpy (DH) from CSV")
    p.add_argument("--input", help="Input CSV path")
    p.add_argument("--output", help="Output CSV path")
    p.add_argument("--reactions-col", default="reaction_smiles",
                   help="Reaction column name (default: reaction_smiles)")
    p.add_argument("--model", default=str(_ROOT / "models" / "egat_dh.pth"),
                   help="Path to checkpoint (.pth)")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4,
                   help="Parallel featurization workers (0 = sequential). Default 4.")
    p.add_argument("--threads", type=int, default=_DEFAULT_THREADS,
                   help=f"Torch intra-op threads (default {_DEFAULT_THREADS}; ~8 is fastest).")
    p.add_argument("--self-test", action="store_true", help="Run a built-in smoke test")
    args = p.parse_args()

    torch.set_num_threads(max(1, args.threads))

    if args.self_test:
        import tempfile
        sample = _ROOT / "examples" / "sample_reactions.csv"
        args.input = str(sample)
        # write somewhere writable (the image filesystem is read-only)
        args.output = str(Path(tempfile.gettempdir()) / "egat_enthalpy_selftest_out.csv")

    if not args.input or not args.output:
        p.error("--input and --output are required (unless --self-test)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, y_mean, y_std = load_model(args.model, device=device)

    df_in = pd.read_csv(args.input)
    pairs, strings = _extract_pairs(df_in, args.reactions_col)

    preds = predict_reactions(model, y_mean, y_std, pairs, device=device,
                              batch_size=args.batch_size, num_workers=args.num_workers)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"reaction_smiles": strings, "reaction_enthalpy": preds}).to_csv(
        out_path, index=False)
    print(f"Wrote {len(preds)} predictions -> {out_path}")

    if args.self_test:
        if not np.isfinite(preds).any():
            raise RuntimeError("Self-test failed: all predictions NaN")
        print("Self-test OK:", [round(x, 3) for x in preds])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
