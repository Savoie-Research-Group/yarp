"""Support functions for reaction enumeration."""

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SharedAtomB2F2Change:
    """One legacy shared-atom B2F2 bond/electron rearrangement."""

    bonds_to_form: tuple
    shared_atom: int
    electron_donor: int
    criterion: str


def legacy_shared_atom_b2f2_changes(n, formset, radicals, bond_mat, elements):
    """Return strict and loose legacy changes for a shared-atom B2F2 step.

    ``None`` means the two broken bonds are not a shared-atom special case. An
    empty tuple means that they are a shared-atom case, but no untouched
    neighbour satisfies either applicable legacy lone-pair criterion.

    ``elements`` is required because the loose criterion must distinguish
    hydrogens from heavy atoms and compare the donor and shared-atom elements.
    """
    if n != 2 or radicals:
        return None

    shared_atoms = [
        atom for atom, count in Counter(formset).items() if count > 1
    ]
    if len(shared_atoms) != 1:
        return None

    shared_atom = shared_atoms[0]
    non_shared_atoms = [atom for atom in formset if atom != shared_atom]
    if len(non_shared_atoms) != 2 or len(set(non_shared_atoms)) != 2:
        return ()

    changes = []
    for neighbor, bond_order in enumerate(bond_mat[shared_atom]):
        if neighbor == shared_atom or bond_order <= 0 or neighbor in formset:
            continue

        other_connections = [
            atom
            for atom, order in enumerate(bond_mat[neighbor])
            if atom not in (neighbor, shared_atom) and order > 0
        ]
        has_lone_electrons = bond_mat[neighbor, neighbor] > 0
        strict_match = not other_connections and has_lone_electrons

        heavy_connections = [
            atom
            for atom in other_connections
            if elements[atom].lower() != "h"
        ]
        loose_match = (
            len(other_connections) < 3
            and not heavy_connections
            and elements[neighbor].lower() != elements[shared_atom].lower()
            and has_lone_electrons
        )

        if strict_match or loose_match:
            if strict_match and loose_match:
                criterion = "strict+loose"
            elif strict_match:
                criterion = "strict"
            else:
                criterion = "loose"
            changes.append(
                SharedAtomB2F2Change(
                    bonds_to_form=(
                        frozenset(non_shared_atoms),
                        frozenset((shared_atom, neighbor)),
                    ),
                    shared_atom=shared_atom,
                    electron_donor=neighbor,
                    criterion=criterion,
                )
            )

    return tuple(changes)


def apply_legacy_shared_atom_b2f2(bond_mat, bonds_to_break, change):
    """Apply the complete legacy shared-atom B2F2 BEM transformation.

    The legacy implementation first returns one electron to each endpoint of
    every broken bond, consumes one electron from each endpoint of every formed
    bond, and finally transfers an electron pair from the untouched terminal
    neighbour to the shared atom.
    """
    product_bmat = np.array(bond_mat, copy=True)

    for bond in bonds_to_break:
        atom_i, atom_j = bond[:2]
        product_bmat[atom_i, atom_j] -= 1
        product_bmat[atom_j, atom_i] -= 1
        product_bmat[atom_i, atom_i] += 1
        product_bmat[atom_j, atom_j] += 1

    for bond in change.bonds_to_form:
        atom_i, atom_j = tuple(bond)
        product_bmat[atom_i, atom_j] += 1
        product_bmat[atom_j, atom_i] += 1
        product_bmat[atom_i, atom_i] -= 1
        product_bmat[atom_j, atom_j] -= 1

    product_bmat[change.electron_donor, change.electron_donor] -= 2
    product_bmat[change.shared_atom, change.shared_atom] += 2

    return product_bmat
