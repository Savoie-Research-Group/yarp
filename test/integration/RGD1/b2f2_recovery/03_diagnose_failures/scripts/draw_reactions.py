"""Draw every reaction in a CSV: reactant -> product, one PNG per reaction.

Takes any CSV with the columns reaction, reactant and product (e.g. the
step 02 / step 03 exports or the compare_runs.py lists) and writes
OUT_DIR/<reaction>.png for each row. Each image shows:
  - the RGD1 reaction ID as the title;
  - reactant and product drawn from YARP's Lewis structures: the kept
    reactant x product BEM pair with the fewest bond-order changes (the same
    pair as step 02's bemmin), product atoms aligned to the reactant by atom map;
  - atom-map numbers next to every atom;
  - red bonds on the reactant: order drops (broken or reduced);
    green bonds on the product: order rises (formed or increased);
  - a key listing those bond changes and the BEMs drawn.
Parsing is step 02's (imported from ../../02_batch_enum/scripts/run_shard.py).
The BEMs come from the checked-out yarp, so they match a run made with the
same version (e.g. A's or B's (b, f) in a compare_runs.py list, not both).
A reaction whose reactant or product does not parse, or whose atom maps
differ, is skipped and listed in OUT_DIR/skipped.csv.

Usage (from 03_diagnose_failures, in the yarp env):
    python scripts/draw_reactions.py INPUT.csv --out-dir DIR
"""
import argparse
import contextlib
import importlib.util
import io
import os
from itertools import product as iproduct
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

# Step 02's driver, loaded under its own name for parse_one (it imports yarpecule first,
# which avoids a circular import)
_spec = importlib.util.spec_from_file_location(
    "step02_run_shard", Path(__file__).resolve().parents[2] / "02_batch_enum" / "scripts" / "run_shard.py")
step02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step02)

from yarp.util.rdkit import yarpecule_to_rdmol  # noqa: E402

RED, GREEN = (0.90, 0.25, 0.25), (0.20, 0.70, 0.30)
W, H = 640, 520  # one panel
FONTS = ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf")


def font(size):
    for f in FONTS:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default(size)


def best_pair(R, P):
    """Fewest-change kept BEM pair, with P's atoms permuted into R's order by atom map.

    Same pair as step 02's bemmin (first minimum of b + f over R.bond_mats x P.bond_mats).
    Returns (ri, pi, rb, pb, perm): BEM indices, the two BEMs in R's order, and the permutation.
    """
    r_maps = [R.atom_info[i]["atom_map"] for i in range(len(R))]
    p_idx = {P.atom_info[i]["atom_map"]: i for i in range(len(P))}
    if sorted(r_maps) != sorted(p_idx):
        raise ValueError("reactant and product atom maps differ")
    perm = [p_idx[m] for m in r_maps]
    best = None
    for (ri, rb), (pi, pb) in iproduct(enumerate(R.bond_mats), enumerate(P.bond_mats)):
        rb, pb = np.asarray(rb, dtype=int), np.asarray(pb, dtype=int)[np.ix_(perm, perm)]
        n_changes = int(np.abs(np.triu(pb - rb, k=1)).sum())
        if best is None or n_changes < best[0]:
            best = (n_changes, ri, pi, rb, pb)
    return (*best[1:], perm)


def panel(elements, adj, bem, atom_info, hl_bonds, color, legend):
    """PNG bytes of one molecule, with atom maps as notes and hl_bonds highlighted."""
    mol = yarpecule_to_rdmol(elements, np.asarray(adj), bem, atom_info=atom_info, sanitize=False)
    for a in mol.GetAtoms():  # show the map as a note instead of ":n" in the label
        a.SetProp("atomNote", str(a.GetAtomMapNum()))
        a.SetAtomMapNum(0)
    AllChem.Compute2DCoords(mol)
    bonds = [b.GetIdx() for i, j in hl_bonds if (b := mol.GetBondBetweenAtoms(int(i), int(j))) is not None]
    d = rdMolDraw2D.MolDraw2DCairo(W, H)
    o = d.drawOptions()
    o.legendFontSize, o.annotationFontScale, o.highlightBondWidthMultiplier = 26, 0.7, 14
    d.DrawMolecule(mol, highlightAtoms=[], highlightBonds=bonds,
                   highlightBondColors={b: color for b in bonds}, legend=legend)
    d.FinishDrawing()
    return d.GetDrawingText()


def draw(rid, R, P, path):
    ri, pi, rb, pb, perm = best_pair(R, P)
    n = len(R)
    lab = lambda a: f"{R.elements[a].capitalize()}{R.atom_info[a]['atom_map']}"
    d = np.triu(pb - rb, k=1)
    dropped = [(i, j) for i in range(n) for j in range(i + 1, n) if d[i, j] < 0]
    raised = [(i, j) for i in range(n) for j in range(i + 1, n) if d[i, j] > 0]
    p_adj = np.asarray(P.adj_mat)[np.ix_(perm, perm)]
    p_info = {i: P.atom_info[perm[i]] for i in range(n)}
    r_png = panel(R.elements, R.adj_mat, rb, R.atom_info, dropped, RED, "reactant")
    p_png = panel(R.elements, p_adj, pb, p_info, raised, GREEN, "product")

    top, bottom, gap = 70, 110, 120
    canvas = Image.new("RGB", (2 * W + gap, top + H + bottom), "white")
    canvas.paste(Image.open(io.BytesIO(r_png)), (0, top))
    canvas.paste(Image.open(io.BytesIO(p_png)), (W + gap, top))
    dr = ImageDraw.Draw(canvas)
    y0, x0, x1 = top + H // 2, W + 15, W + gap - 15
    dr.line([(x0, y0), (x1, y0)], fill="black", width=5)
    dr.polygon([(x1 + 4, y0), (x1 - 18, y0 - 12), (x1 - 18, y0 + 12)], fill="black")
    dr.text((20, 18), rid, fill="black", font=font(34))
    change = lambda i, j: f"{lab(i)}-{lab(j)} {rb[i, j]}->{pb[i, j]}"
    key = [f"red (reactant): bond order drops: {', '.join(change(i, j) for i, j in dropped) or '-'}",
           f"green (product): bond order rises: {', '.join(change(i, j) for i, j in raised) or '-'}",
           f"YARP (b, f) = ({-int(d[d < 0].sum())}, {int(d[d > 0].sum())}): reactant BEM {ri + 1} of "
           f"{len(R.bond_mats)}, product BEM {pi + 1} of {len(P.bond_mats)}.   Numbers: RGD1 atom maps."]
    for k, line in enumerate(key):
        dr.text((20, top + H + 8 + 30 * k), line, fill="black", font=font(20))
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="CSV with reaction, reactant and product columns")
    ap.add_argument("--out-dir", type=Path, required=True, help="folder for the PNGs (created if needed)")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(a.csv, usecols=["reaction", "reactant", "product"])

    parsed, skipped = {}, []

    def parse(smi):
        if smi not in parsed:
            parsed[smi] = step02.parse_one(smi)
        return parsed[smi]

    for k, r in enumerate(rows.itertuples(index=False), 1):
        R, r_rec = parse(r.reactant)
        P, p_rec = parse(r.product)
        reason = (f"reactant parse {r_rec['status']}" if R is None else
                  f"product parse {p_rec['status']}" if P is None else None)
        if reason is None:
            try:
                with contextlib.redirect_stderr(io.StringIO()):  # RDKit drawing chatter
                    draw(r.reaction, R, P, a.out_dir / f"{r.reaction}.png")
            except Exception as e:  # noqa: BLE001 -- report the row and carry on
                reason = f"{type(e).__name__}: {e}"[:200]
        if reason is not None:
            skipped.append({"reaction": r.reaction, "reason": reason, "reactant": r.reactant, "product": r.product})
        print(f"[{k}/{len(rows)}] {r.reaction}: {reason or 'drawn'}", flush=True)

    print(f"\nDrew {len(rows) - len(skipped):,} of {len(rows):,} reactions into {a.out_dir}")
    if skipped:
        pd.DataFrame(skipped).to_csv(a.out_dir / "skipped.csv", index=False)
        print(f"Skipped {len(skipped):,}; reasons in {a.out_dir / 'skipped.csv'}")


if __name__ == "__main__":
    main()
