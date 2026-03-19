"""
Development-only draft helpers for EGAT-boundary atom-map normalization.

Why this exists
---------------
YARP now preserves atom maps exactly, including fragment-local subsets carried
through `separate()` and enum. EGAT does not reliably tolerate those preserved
labels because some of its feature builders assume dense positional map labels.

The intended production use is:
1. Keep YARP reaction objects untouched.
2. Build a temporary normalized mapped-SMILES copy just before EGAT.
3. Feed the temporary copy to EGAT.
4. Discard the temporary copy after prediction.

This file is for review and experimentation only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from rdkit import Chem


@dataclass(frozen=True)
class NormalizedMappedReaction:
    reactant_smiles: str
    product_smiles: str
    reaction_smiles: str
    old_to_new_map: Dict[int, int]


def _mapped_mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"Could not parse mapped SMILES: {smiles}")
    return mol


def _atom_maps(mol: Chem.Mol) -> list[int]:
    maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    if any(m is None for m in maps):
        raise ValueError("Encountered missing atom-map value.")
    if any(m < 0 for m in maps):
        raise ValueError(f"Negative atom-map values are not allowed: {maps}")
    if len(set(maps)) != len(maps):
        dupes = sorted({m for m in maps if maps.count(m) > 1})
        raise ValueError(f"Duplicate atom-map labels are not allowed: {dupes}")
    return maps


def _require_identical_map_sets(r_maps: Iterable[int], p_maps: Iterable[int]) -> list[int]:
    r_set = set(r_maps)
    p_set = set(p_maps)
    if r_set != p_set:
        raise ValueError(
            "Reactant/product mapped atom sets differ and cannot be normalized "
            f"as one reaction.\nReactant-only: {sorted(r_set - p_set)}\n"
            f"Product-only: {sorted(p_set - r_set)}"
        )
    return sorted(r_set)


def _apply_map_renumbering(mol: Chem.Mol, old_to_new_map: Dict[int, int]) -> Chem.Mol:
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(old_to_new_map[atom.GetAtomMapNum()])
    return mol


def normalize_reaction_maps_for_egat(
    reactant_smiles: str,
    product_smiles: str,
    *,
    start_at: int = 1,
) -> NormalizedMappedReaction:
    """
    Return a temporary EGAT-only mapped-SMILES copy with dense atom maps.

    Parameters
    ----------
    reactant_smiles, product_smiles
        Atom-mapped SMILES strings already known to correspond to the same
        reaction.
    start_at
        Dense atom-map start. Use `1` for EGAT compatibility.

    Notes
    -----
    `start_at=1` is the preferred choice for EGAT because:
    - `GenerateBondFeature()` hard-codes `edge_index + 1` when it looks up
      bond-map keyed metadata.
    - `GenerateAtomFeature()` already has a nonzero-start branch (`ind + 1`).
    """

    if start_at not in (0, 1):
        raise ValueError(f"Only dense starts of 0 or 1 are supported, got {start_at}.")

    r_mol = _mapped_mol_from_smiles(reactant_smiles)
    p_mol = _mapped_mol_from_smiles(product_smiles)

    r_maps = _atom_maps(r_mol)
    p_maps = _atom_maps(p_mol)
    ordered_old_maps = _require_identical_map_sets(r_maps, p_maps)

    old_to_new_map = {
        old_map: new_map
        for new_map, old_map in enumerate(ordered_old_maps, start=start_at)
    }

    nr_mol = _apply_map_renumbering(r_mol, old_to_new_map)
    np_mol = _apply_map_renumbering(p_mol, old_to_new_map)

    normalized_reactant = Chem.MolToSmiles(nr_mol)
    normalized_product = Chem.MolToSmiles(np_mol)
    return NormalizedMappedReaction(
        reactant_smiles=normalized_reactant,
        product_smiles=normalized_product,
        reaction_smiles=f"{normalized_reactant}>>{normalized_product}",
        old_to_new_map=old_to_new_map,
    )


if __name__ == "__main__":
    # Representative depth-2 failure that should normalize cleanly.
    example = normalize_reaction_maps_for_egat(
        "[O:0]=[C:1]([O:4][O:5][H:11])[H:6]",
        "[H:6][H:11].[O:0]=[C:1]1[O:4][O:5]1",
        start_at=1,
    )
    print(example)
