#!/usr/bin/env python3
"""
Build a minimal Cantera YAML from a list of reaction dictionaries produced by
`pull_cantera_data_from_rxn_obj`.
"""
from __future__ import annotations
import io
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from cantera_util import *

DEFAULT_A = 1e12  # s^-1, reasonable order-of-magnitude pre-exponential
DEFAULT_B = 0.0   # unitless

def initial_comp_from_lists(species_list, frac_list):
    """Build initial composition dict from parallel lists."""
    if species_list is None or frac_list is None:
        return None
    if len(species_list) != len(frac_list):
        raise ValueError("initial_species_smi and initial_species_frac must be the same length.")
    return {str(s): float(f) for s, f in zip(species_list, frac_list)}

def write_species_blocks(buf, species):
    """Write species blocks to the YAML buffer."""
    buf.write("species:\n")
    for s in species:
        comp = elements_from_smiles(s)
        comp_str = ", ".join(f"{k}: {v}" for k, v in comp.items())
        buf.write(f"- name: {quote(s)}\n")
        buf.write(f"  composition: {{{comp_str}}}\n")
        buf.write("  thermo:\n")
        buf.write("    model: constant-cp\n")
        buf.write("    h0: 0.0 kcal/mol\n")
        buf.write("    s0: 0.0 cal/mol/K\n")
        buf.write("    cp0: 1.0 cal/mol/K\n\n")

def write_reaction_blocks(buf, cantera_data_list, dg_units):
    """Write reaction blocks to the YAML buffer.
    Seperate forward and reverse reactions if reverse barrier is provided.
    Returns a list of skipped reactions with reasons.
    """
    skipped = []
    buf.write("reactions:\n")
    for i, rxn in enumerate(cantera_data_list):
        rid = rxn.get("id") or f"rxn_{i}"
        r_list = split_species(rxn.get("reactant_smi"))
        p_list = split_species(rxn.get("product_smi"))

        try:
            Ea = convert_energy_to_kcal(rxn.get("barrier"), dg_units)
        except Exception as exc:  # capture missing/invalid barrier
            skipped.append({"index": i, "id": rid, "why": str(exc)})
            continue

        r_side = " + ".join(r_list) if r_list else "[H]"
        p_side = " + ".join(p_list) if p_list else "[H]"

        # forward reaction
        equation = eq_quote(f"{r_side} => {p_side}")
        buf.write(f"- id: {quote(str(rid))}\n")
        buf.write(f"  equation: {equation}\n")
        buf.write(f"  rate-constant: {{A: {DEFAULT_A:.6g}, b: {DEFAULT_B:.6g}, Ea: {Ea:.6g} kcal/mol}}\n")
        buf.write("  duplicate: false\n\n")

        # reverse reaction if a reverse barrier is provided
        rev_barrier = rxn.get("reverse_barrier")
        if rev_barrier is not None:
            try:
                Ea_rev = convert_energy_to_kcal(rev_barrier, dg_units)
                rev_equation = eq_quote(f"{p_side} => {r_side}")
                buf.write(f"- id: {quote(str(rid))}_reverse\n")
                buf.write(f"  equation: {rev_equation}\n")
                buf.write(f"  rate-constant: {{A: {DEFAULT_A:.6g}, b: {DEFAULT_B:.6g}, Ea: {Ea_rev:.6g} kcal/mol}}\n")
                buf.write("  duplicate: false\n\n")
            except Exception as exc:
                skipped.append({"index": i, "id": f"{rid}_reverse", "why": str(exc)})
    return skipped


def write_yaml(
    cantera_data_list,
    out_yaml,
    temp,
    pressure,
    initial_species_smi,
    initial_species_frac,
    auto_balance_h,
    write_skip_csv,
    dg_units = "kcal/mol",
    hygrogen_species = ("[H]", "[H][H]")):
    """
    Serialize a Cantera mechanism YAML from reaction dicts.
    """
    out_yaml = Path(out_yaml)

    # Build species set from reactants/products
    species = sorted({
        smi for rxn in cantera_data_list
        for smi in split_species(rxn.get("reactant_smi")) + split_species(rxn.get("product_smi"))
    })
    if auto_balance_h:
        species = sorted(set(species) | set(hygrogen_species))

    elements = sorted({el for smi in species for el in elements_from_smiles(smi).keys()})
    list_comp = initial_comp_from_lists(initial_species_smi, initial_species_frac)
    comp = normalize_composition(species, list_comp)
    
    buf = io.StringIO()
    buf.write("units: {length: cm, time: s, quantity: mol, activation-energy: kcal/mol, pressure: atm, energy: kcal}\n\n")

    buf.write("phases:\n")
    buf.write("- name: gas\n  thermo: ideal-gas\n")
    buf.write(f"  elements: [{', '.join(elements)}]\n")
    buf.write("  kinetics: gas\n  reactions: all\n")
    species_block = ", ".join(quote(s) for s in species)
    comp_block = "{ " + ", ".join([f"'{s}': {comp[s]}" for s in species]) + " }\n"
    buf.write(f"  species: [{species_block}]\n")
    buf.write(f"  state: {{T: {temp}, P: {pressure} atm, X: {comp_block.strip()} }}\n\n")

    write_species_blocks(buf, species)
    skipped = write_reaction_blocks(buf, cantera_data_list, dg_units)

    out_yaml.write_text(buf.getvalue())

    if skipped and write_skip_csv:
        import csv

        skip_path = out_yaml.with_suffix(".skipped.csv")
        with skip_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["index", "id", "why"])
            writer.writeheader()
            writer.writerows(skipped)

    return out_yaml


__all__ = ["write_yaml"]
