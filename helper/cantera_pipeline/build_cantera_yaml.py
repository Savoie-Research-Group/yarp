#!/usr/bin/env python3
"""
build_cantera_yaml.py
---------------------
Generate a minimal Cantera YAML from a list of YAKS_Rxn objects (or any
YARP-like reaction objects) using only their public attributes/methods.

Usage (CRC-friendly):
python build_cantera_yaml.py --rxn-pickle reactions.pkl --out reactions.yaml

Assumptions about each reaction object (duck-typed):
- rxn.id : str (reaction ID for YAML)
- rxn.rate: Dict[str, RateModel-like] where a rate object has attributes
(A, b, Ea_kcal_per_mol). If --rate-label is omitted, the first key is used.
- Either:
    * rxn.species_smiles() -> List[str], OR
    * rxn.reactant.species and rxn.product.species iterables with each species
    having a `.canon_smi` attribute.

Notes
-----
* Species thermo is emitted as a constant-cp placeholder to keep YAML self-contained.
* Activation energy units are kcal/mol; the YAML declares the same.
* RDKit is used to derive elemental compositions from SMILES.

"""
from __future__ import annotations
import argparse, io, pickle, sys
from typing import Dict, Iterable, List, Optional, Set

# RDKit for basic composition
from rdkit import Chem
from ct_util import *

# -----------------------
# YAML writers
# -----------------------

def yaml_header():
    return (
        "units: {length: cm, time: s, quantity: mol, activation-energy: kcal/mol,\n"
        "        pressure: atm, energy: kcal}\n\n"
    )


def yaml_phase(all_species, elements):
    buf = io.StringIO()
    buf.write("phases:\n")
    buf.write("- name: gas\n")
    buf.write("  thermo: ideal-gas\n")
    buf.write("  kinetics: gas\n")
    buf.write("  elements: [" + ", ".join(elements) + "]\n")
    buf.write("  species: [" + ", ".join(quote(s) for s in all_species) + "]\n\n")
    return buf.getvalue()


def yaml_species_blocks(all_species):
    buf = io.StringIO()
    buf.write("species:\n")
    for smi in all_species:
        comp = elements_from_smiles(smi)
        comp_str = ", ".join(f"{k}: {v}" for k, v in comp.items())
        buf.write(f"- name: {quote(smi)}\n")
        buf.write(f"  composition: {{{comp_str}}}\n")
        buf.write("  thermo:\n")
        buf.write("    model: constant-cp\n")
        buf.write("    h0: 0.0 kcal/mol\n")
        buf.write("    s0: 0.0 cal/mol/K\n")
        buf.write("    cp0: 1.0 cal/mol/K\n\n")
    return buf.getvalue()


def yaml_reaction_block(rxn, rate_label):
    # Reaction equation: list reactants and products by SMILES if available
    r = extract_smiles_from_state(getattr(rxn, "reactant", None))
    p = extract_smiles_from_state(getattr(rxn, "product", None))
    r_side = " + ".join(quote(s) for s in r) if r else quote("[H]")
    p_side = " + ".join(quote(s) for s in p) if p else quote("[H]")

    rate_dict = getattr(rxn, "rate", None) or {}
    if not rate_dict:
        raise RuntimeError(f"Reaction {getattr(rxn, 'id', '<no-id>')} has no rate parameters.")
    if rate_label is None:
        rate_label = next(iter(rate_dict.keys()))
    rm = rate_dict[rate_label]

    A = getattr(rm, "A")
    b = getattr(rm, "b")
    Ea = getattr(rm, "Ea_kcal_per_mol")

    buf = io.StringIO()
    buf.write("- id: " + quote(getattr(rxn, "id", "rxn")) + "\n")
    buf.write("  equation: " + f"{r_side} => {p_side}" + "\n")
    buf.write("  rate-constant: " + f"{{A: {A:.6g}, b: {b:.6g}, Ea: {Ea:.6g} kcal/mol}}" + "\n")
    buf.write("  duplicate: false\n")
    return buf.getvalue()


def write_cantera_yaml(reactions, out_path, rate_label=None, auto_balance=True):
    # species set
    species_set: Set[str] = set()
    for rxn in reactions:
        species_set.update(species_smiles_for_rxn(rxn))
    if auto_balance:
        species_set.update({"[H]", "[H][H]"})
    all_species = sorted(species_set)

    # elements across all species
    elements = sorted({el for smi in all_species for el in elements_from_smiles(smi).keys()})

    # assemble YAML
    with open(out_path, "w") as f:
        f.write(yaml_header())
        f.write(yaml_phase(all_species, elements))
        f.write(yaml_species_blocks(all_species))
        f.write("reactions:\n")
        for rxn in reactions:
            f.write(yaml_reaction_block(rxn, rate_label))
            f.write("\n")
    return out_path


# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Build a Cantera YAML from a pickled list of YAKS_Rxn objects.")
    ap.add_argument("--rxn-pickle", required=True, help="Path to pickle containing a list of reaction objects")
    ap.add_argument("--out", default="reactions.yaml", help="Output YAML path")
    ap.add_argument("--rate-label", default=None, help="Key in rxn.rate to use (default: first)")
    ap.add_argument("--no-balance", action="store_true", help="Do not add [H] and [H][H] helper species")
    args = ap.parse_args()

    with open(args.rxn_pickle, "rb") as f:
        rxn_list = pickle.load(f)
        if not isinstance(rxn_list, (list, tuple)):
            print("ERROR: Pickle must contain a list/tuple of reactions.", file=sys.stderr)
            sys.exit(2)

    out = write_cantera_yaml(rxn_list, args.out, rate_label=args.rate_label,auto_balance=(not args.no_balance))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
