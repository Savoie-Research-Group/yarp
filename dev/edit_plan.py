"""
Development-only edit plan.

Each block below uses real Python definitions instead of quoted snippets:
- `*_old` is copied from the current codebase or reduced only by renaming.
- `*_new` is the proposed replacement body.

These are not meant to be imported by production code.
They are here so we can review exact edits side by side in code form.
"""

# Status note:
# - The core parser/yarpecule/enum production pass has been implemented.
# - This file remains useful as an archival side-by-side reference, but it is no
#   longer the source of truth for next work.
# - Latest cluster state:
#   - base depth-1, larger, aromatic, and isotope runs pass through EGAT
#   - sparse-map and depth-2 runs now log exact failing AAM strings and still
#     write pickles
#   - duplicate-map input now fails hard at yarpecule construction
# - Remaining active work is EGAT-side debugging / compatibility:
#   sparse-map handling, depth-2 failures, and possible EGAT-boundary map
#   normalization.

import os
import re
import sys
import types
import numpy as np

try:
    import openbabel  # type: ignore
except ModuleNotFoundError:
    class _DummyErrorLog:
        def SetOutputLevel(self, *_args, **_kwargs):
            return None

    openbabel_stub = types.ModuleType('openbabel')
    openbabel_stub.pybel = types.SimpleNamespace()
    openbabel_stub.openbabel = types.SimpleNamespace(obErrorLog=_DummyErrorLog(), obError=0)
    sys.modules['openbabel'] = openbabel_stub

from rdkit import Chem
from rdkit.Chem import AddHs, AllChem, Atom, BondType, MolFromSmiles, rdchem, rdmolfiles

from yarp.util.properties import el_expand_octet, el_mass, el_metals, el_n_expand_octet, el_to_an, el_valence
from yarp.util.write_files import mol_write_yp, xyz_write
from yarp.yarpecule.atom_mapping import canon_order
from yarp.yarpecule.graph.adjacency import adjmat_to_adjlist, table_generator
from yarp.yarpecule.graph.fragment import return_rings
from yarp.yarpecule.graph.smiles import OctetError, smiles2adjmat
from yarp.yarpecule.hashes import atom_hash
from yarp.yarpecule.input_parsers import mol_parse, xyz_from_smiles, xyz_parse, xyz_q_parse
from yarp.yarpecule.lewis.be_mat import return_formals


def smiles2adjmat_old(smiles, verbose=False):
    """
    Old top-level parser shape from yarp/yarpecule/graph/smiles.py.

    Difference:
    - returns list-based atom_info.
    - aromatic handling is the current alternating-bond path.
    - no stereo metadata persistence.
    - no dict-based atom_index / atom_map split.

    This block is intentionally a direct structural mirror of the current code:
    tokenization -> atom_info build -> graph build -> aromatic handling ->
    add_hydrogens() -> reorder_by_mappings() -> bond-electron-matrix build.
    """

    if not hasattr(smiles2adjmat_old, 'aromatics'):
        smiles2adjmat_old.aromatics = {"b", "c", "n", "o", "p", "s"}
        smiles2adjmat_old.token_pattern = r'(\[[^\]]*\]|[A-Z](?:[a-z]+)?|[a-z]|\d{1}|[=#+\-\\\/.()])'
        smiles2adjmat_old.atom_pattern = r'([A-Z](?:[a-z]+)?|[a-z])'
        smiles2adjmat_old.isotope_pattern = re.compile(r'^\[(\d+)')
        smiles2adjmat_old.charge_pattern = re.compile(r'(\+\d+|-\d+|\++(?!\d)|-+(?!\d))')
        smiles2adjmat_old.hydrogen_pattern = re.compile(r'(H\d+|H+)')
        smiles2adjmat_old.element_label_pattern = re.compile(r'([A-Z](?:[a-z]+)?|[a-z])')
        smiles2adjmat_old.mapping_pattern = re.compile(r':(\d+)')
        smiles2adjmat_old.valid_elements = {
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Hf', 'Ta', 'W',
            'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
            'At', 'Rn', 'b', 'c', 'n', 'o', 'p', 's'
        }

    preliminary_tokens = re.findall(smiles2adjmat_old.token_pattern, smiles)

    final_tokens = []
    for token in preliminary_tokens:
        if len(token) == 2 and token not in smiles2adjmat_old.valid_elements:
            final_tokens.extend(list(token))
        else:
            final_tokens.append(token)

    preliminary_tokens = final_tokens

    atom_info = []
    tokens = []

    for token in preliminary_tokens:
        if token.startswith('['):
            formal_charge = 0
            explicit_hydrogens = 0
            isotope = None
            atom_mapping = None
            should_infer_hydrogens = False

            isotope_match = smiles2adjmat_old.isotope_pattern.search(token)
            if isotope_match:
                isotope = int(isotope_match.group(1))

            charge_match = smiles2adjmat_old.charge_pattern.search(token)
            if charge_match:
                charge_str = charge_match.group(1)
                if charge_str[-1].isdigit():
                    formal_charge = int(charge_str)
                else:
                    formal_charge = charge_str.count('+') - charge_str.count('-')

            hydrogen_match = smiles2adjmat_old.hydrogen_pattern.search(token)
            if verbose:
                print(f"{token=} {hydrogen_match=}")
            if hydrogen_match:
                h_str = hydrogen_match.group(1)
                element_label = smiles2adjmat_old.element_label_pattern.search(token).group(1)
                if element_label.lower() != 'h':
                    if h_str[-1].isdigit():
                        explicit_hydrogens = int(h_str[1:])
                    else:
                        explicit_hydrogens = len(h_str)

            mapping_match = smiles2adjmat_old.mapping_pattern.search(token)
            if mapping_match:
                atom_mapping = int(mapping_match.group(1))

            element_label_match = smiles2adjmat_old.element_label_pattern.search(token)
            if element_label_match:
                atom_info.append([element_label_match.group(1), formal_charge,
                                  explicit_hydrogens, isotope, atom_mapping,
                                  should_infer_hydrogens])
                tokens.append(element_label_match.group(1))
            else:
                print(f"Error: Could not detect element label at token {token}.")
                return None, None

        elif re.match(smiles2adjmat_old.atom_pattern, token):
            formal_charge = 0
            explicit_hydrogens = None
            isotope = None
            atom_mapping = None
            should_infer_hydrogens = True
            atom_info.append([token, formal_charge, explicit_hydrogens,
                              isotope, atom_mapping, should_infer_hydrogens])
            tokens.append(token)
        else:
            tokens.append(token)

    adjmat = np.zeros([len(atom_info), len(atom_info)])

    atom_indices = []
    branch_levels = []
    current_level = 0
    branch_open_atom_indices = []
    branch_close_atom_indices = []
    open_flag = False
    close_flag = False
    ring_numbers = {}

    atom_counter = 0
    for i, token in enumerate(tokens):
        if token == '(':
            current_level += 1
            open_flag = True
        elif token == ')':
            current_level -= 1
            close_flag = True
        elif re.match(smiles2adjmat_old.atom_pattern, token):
            atom_indices.append(i)
            branch_levels.append(current_level)
            if close_flag and not open_flag:
                close_flag = False
                branch_close_atom_indices.append(atom_counter)
            if open_flag:
                open_flag = False
                branch_open_atom_indices.append(atom_counter)
            atom_counter += 1
        elif token.isdigit():
            atom_index = atom_counter - 1

            if token in ring_numbers:
                if tokens[i-1] == '-':
                    last_bond_type = 1
                elif tokens[i-1] == '=':
                    last_bond_type = 2
                elif tokens[i-1] == '#':
                    last_bond_type = 3
                else:
                    last_bond_type = None

                prev_atom_index, prev_bond_type = ring_numbers[token]
                bond_type = last_bond_type if last_bond_type else prev_bond_type
                if prev_bond_type and last_bond_type and prev_bond_type != last_bond_type:
                    print(f"Error: Inconsistent bond order specified for ring closure at token {token}.")
                    return None, None

                if bond_type is None:
                    bond_type = 1

                if atom_index >= len(atom_info) or prev_atom_index >= len(atom_info):
                    print(f"Error: Atom index out of bounds. atom_index={atom_index}, prev_atom_index={prev_atom_index}, num_atoms={len(atom_info)}")
                    return None, None

                adjmat[atom_index, prev_atom_index] = adjmat[prev_atom_index, atom_index] = bond_type
                del ring_numbers[token]
            else:
                if tokens[i-1] == '-':
                    last_bond_type = 1
                elif tokens[i-1] == '=':
                    last_bond_type = 2
                elif tokens[i-1] == '#':
                    last_bond_type = 3
                else:
                    last_bond_type = None
                ring_numbers[token] = (atom_index, last_bond_type)

    for i in range(len(atom_indices)-1):
        current_atom_index = atom_indices[i]
        next_atom_index = atom_indices[i + 1]
        intervening_tokens = tokens[current_atom_index + 1: next_atom_index]

        if not any(re.match(r'[\[\].()]', token) for token in intervening_tokens):
            after_digits = []
            for j in intervening_tokens[::-1]:
                if j.isdigit():
                    break
                else:
                    after_digits.append(j)

            if '=' in after_digits:
                adjmat[i, i + 1] = adjmat[i + 1, i] = 2
            elif '#' in after_digits:
                adjmat[i, i + 1] = adjmat[i + 1, i] = 3
            else:
                adjmat[i, i + 1] = adjmat[i + 1, i] = 1

    for i in branch_open_atom_indices:
        current_atom_index = atom_indices[i]
        current_branch_level = branch_levels[i]

        if current_atom_index == 0:
            continue
        else:
            for j in range(i-1, -1, -1):
                if branch_levels[j] == current_branch_level - 1:
                    intervening_tokens = []
                    for k in range(current_atom_index, 0, -1):
                        if tokens[k] == "(":
                            break
                        intervening_tokens.append(tokens[k])

                    if '=' in intervening_tokens:
                        adjmat[i, j] = adjmat[j, i] = 2
                    elif '#' in intervening_tokens:
                        adjmat[i, j] = adjmat[j, i] = 3
                    else:
                        adjmat[i, j] = adjmat[j, i] = 1
                    break

    for i in branch_close_atom_indices:
        current_atom_index = atom_indices[i]
        current_branch_level = branch_levels[i]

        for j in range(i-1, -1, -1):
            if branch_levels[j] == current_branch_level:
                intervening_tokens = []
                for k in range(current_atom_index, 0, -1):
                    if tokens[k] == ")":
                        break
                    intervening_tokens.append(tokens[k])

                if '=' in intervening_tokens:
                    adjmat[i, j] = adjmat[j, i] = 2
                elif '#' in intervening_tokens:
                    adjmat[i, j] = adjmat[j, i] = 3
                else:
                    adjmat[i, j] = adjmat[j, i] = 1
                break

    if any([_ in smiles2adjmat_old.aromatics for _ in tokens]):
        for r in return_rings(adjmat_to_adjlist(adjmat), max_size=10, remove_fused=True):
            if all([atom_info[_][0] in smiles2adjmat_old.aromatics for _ in r]):
                for ind in range(1, len(r)):
                    if (ind-1) % 2 == 0:
                        adjmat[r[ind], r[ind-1]] = 2
                        adjmat[r[ind-1], r[ind]] = 2

    adjmat, atom_info = add_hydrogens_old(
        adjmat,
        atom_info,
        smiles2adjmat_old,
        el_valence,
        el_metals,
        el_expand_octet,
        OctetError,
    )

    adjmat, atom_info = reorder_by_mappings_old(adjmat, atom_info, np, canon_order)

    for info in atom_info:
        info[0] = info[0].capitalize()

    bond_electron_mat = adjmat.copy()
    for i, info in enumerate(atom_info):
        bond_electron_mat[i, i] = el_valence[info[0].lower()] - info[1] - sum(adjmat[i])

    return np.where(adjmat > 0, 1, 0), bond_electron_mat, atom_info


def smiles2adjmat_new(smiles, verbose=False):
    """
    New top-level parser shape proposed for yarp/yarpecule/graph/smiles.py.

    Difference:
    - keeps the same function name and same `(adjmat, bond_electron_mat, atom_info)` return shape.
    - atom_info becomes dict-based.
    - stores `atom_index` separately from `atom_map`.
    - does kekulize-first aromatic handling, then warning-backed fallback.
    - keeps RDKit fallback out of this function.
    - is intended to directly improve:
      - lower-case aromatic-input failures
      - parser breakage on some steric markings
      - map metadata retention for downstream output generation

    This function body stays minimal by reusing the existing parser flow and
    delegating the concrete structural changes to the edited helper blocks
    below: `add_hydrogens_new()` and `reorder_by_mappings_new()`.
    """

    if not hasattr(smiles2adjmat_new, 'aromatics'):
        smiles2adjmat_new.aromatics = {"b", "c", "n", "o", "p", "s"}
        smiles2adjmat_new.token_pattern = r'(\[[^\]]*\]|%\d{2}|[A-Z](?:[a-z]+)?|[a-z]|\d{1}|[=#+\-\\\/.@:()])'
        smiles2adjmat_new.atom_pattern = r'([A-Z](?:[a-z]+)?|[a-z])'
        smiles2adjmat_new.ring_pattern = re.compile(r'^(%\d{2}|\d)$')
        smiles2adjmat_new.isotope_pattern = re.compile(r'^\[(\d+)')
        smiles2adjmat_new.charge_pattern = re.compile(r'(\+\d+|-\d+|\++(?!\d)|-+(?!\d))')
        smiles2adjmat_new.hydrogen_pattern = re.compile(r'(H\d+|H+)')
        smiles2adjmat_new.element_label_pattern = re.compile(r'([A-Z](?:[a-z]+)?|[a-z])')
        smiles2adjmat_new.mapping_pattern = re.compile(r':(\d+)')
        smiles2adjmat_new.stereo_atom_pattern = re.compile(r'(@@?|\\\/)')
        smiles2adjmat_new.valid_elements = {
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Hf', 'Ta', 'W',
            'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
            'At', 'Rn', 'b', 'c', 'n', 'o', 'p', 's'
        }

    preliminary_tokens = re.findall(smiles2adjmat_new.token_pattern, smiles)

    final_tokens = []
    for token in preliminary_tokens:
        if len(token) == 2 and token not in smiles2adjmat_new.valid_elements and not token.startswith("%"):
            final_tokens.extend(list(token))
        else:
            final_tokens.append(token)
    preliminary_tokens = final_tokens

    atom_info = {}
    atom_parse_meta = {}
    tokens = []
    atom_counter = 0

    for token in preliminary_tokens:
        if token.startswith('['):
            formal_charge = 0
            explicit_hydrogens = 0
            isotope = None
            atom_mapping = None
            should_infer_hydrogens = False

            isotope_match = smiles2adjmat_new.isotope_pattern.search(token)
            if isotope_match:
                isotope = int(isotope_match.group(1))

            charge_match = smiles2adjmat_new.charge_pattern.search(token)
            if charge_match:
                charge_str = charge_match.group(1)
                if charge_str[-1].isdigit():
                    formal_charge = int(charge_str)
                else:
                    formal_charge = charge_str.count('+') - charge_str.count('-')

            hydrogen_match = smiles2adjmat_new.hydrogen_pattern.search(token)
            if verbose:
                print(f"{token=} {hydrogen_match=}")
            if hydrogen_match:
                h_str = hydrogen_match.group(1)
                element_label = smiles2adjmat_new.element_label_pattern.search(token).group(1)
                if element_label.lower() != 'h':
                    if h_str[-1].isdigit():
                        explicit_hydrogens = int(h_str[1:])
                    else:
                        explicit_hydrogens = len(h_str)

            mapping_match = smiles2adjmat_new.mapping_pattern.search(token)
            if mapping_match:
                atom_mapping = int(mapping_match.group(1))

            element_label_match = smiles2adjmat_new.element_label_pattern.search(token)
            if element_label_match:
                element_label = element_label_match.group(1)
                stereo_atom = '@@' if '@@' in token else '@' if '@' in token else None
                atom_info[atom_counter] = {
                    "atom_index": atom_counter,
                    "atom_map": atom_mapping,
                    "element": element_label.lower(),
                    "formal_charge": int(formal_charge),
                    "mass": float(isotope) if isotope is not None else None,
                    "stereo": {"atom": stereo_atom, "bonds": {}},
                    "aromatic_input": element_label.islower(),
                }
                atom_parse_meta[atom_counter] = {
                    "explicit_hydrogens": explicit_hydrogens,
                    "should_infer_hydrogens": should_infer_hydrogens,
                    "isotope": isotope,
                }
                tokens.append(element_label)
                atom_counter += 1
            else:
                print(f"Error: Could not detect element label at token {token}.")
                return None, None

        elif re.match(smiles2adjmat_new.atom_pattern, token):
            atom_info[atom_counter] = {
                "atom_index": atom_counter,
                "atom_map": None,
                "element": token.lower(),
                "formal_charge": 0,
                "mass": None,
                "stereo": {"atom": None, "bonds": {}},
                "aromatic_input": token.islower(),
            }
            atom_parse_meta[atom_counter] = {
                "explicit_hydrogens": None,
                "should_infer_hydrogens": True,
                "isotope": None,
            }
            tokens.append(token)
            atom_counter += 1
        else:
            tokens.append(token)

    adjmat = np.zeros([len(atom_info), len(atom_info)])

    atom_indices = []
    branch_levels = []
    current_level = 0
    branch_open_atom_indices = []
    branch_close_atom_indices = []
    open_flag = False
    close_flag = False
    ring_numbers = {}
    pending_bond_stereo = None
    pending_atom_stereo_owner = None
    sequential_atom_counter = 0

    for i, token in enumerate(tokens):
        if token == '(':
            current_level += 1
            open_flag = True
        elif token == ')':
            current_level -= 1
            close_flag = True
        elif token in ['/', '\\']:
            pending_bond_stereo = token
        elif token in ['@', '@@']:
            pending_atom_stereo_owner = sequential_atom_counter - 1
        elif re.match(smiles2adjmat_new.atom_pattern, token):
            atom_indices.append(i)
            branch_levels.append(current_level)
            if close_flag and not open_flag:
                close_flag = False
                branch_close_atom_indices.append(sequential_atom_counter)
            if open_flag:
                open_flag = False
                branch_open_atom_indices.append(sequential_atom_counter)

            if pending_atom_stereo_owner is not None and pending_atom_stereo_owner in atom_info:
                atom_info[pending_atom_stereo_owner]["stereo"]["atom"] = '@'
                pending_atom_stereo_owner = None

            if pending_bond_stereo is not None and sequential_atom_counter > 0:
                atom_info[sequential_atom_counter]["stereo"]["bonds"][sequential_atom_counter - 1] = pending_bond_stereo
                pending_bond_stereo = None

            sequential_atom_counter += 1
        elif smiles2adjmat_new.ring_pattern.match(token):
            atom_index = sequential_atom_counter - 1
            if token in ring_numbers:
                if tokens[i-1] == '-':
                    last_bond_type = 1
                elif tokens[i-1] == '=':
                    last_bond_type = 2
                elif tokens[i-1] == '#':
                    last_bond_type = 3
                else:
                    last_bond_type = None

                prev_atom_index, prev_bond_type, prev_bond_stereo = ring_numbers[token]
                bond_type = last_bond_type if last_bond_type else prev_bond_type
                if prev_bond_type and last_bond_type and prev_bond_type != last_bond_type:
                    print(f"Error: Inconsistent bond order specified for ring closure at token {token}.")
                    return None, None
                if bond_type is None:
                    bond_type = 1
                adjmat[atom_index, prev_atom_index] = adjmat[prev_atom_index, atom_index] = bond_type

                if pending_bond_stereo is not None:
                    atom_info[atom_index]["stereo"]["bonds"][prev_atom_index] = pending_bond_stereo
                elif prev_bond_stereo is not None:
                    atom_info[atom_index]["stereo"]["bonds"][prev_atom_index] = prev_bond_stereo
                pending_bond_stereo = None
                del ring_numbers[token]
            else:
                if tokens[i-1] == '-':
                    last_bond_type = 1
                elif tokens[i-1] == '=':
                    last_bond_type = 2
                elif tokens[i-1] == '#':
                    last_bond_type = 3
                else:
                    last_bond_type = None
                ring_numbers[token] = (atom_index, last_bond_type, pending_bond_stereo)
                pending_bond_stereo = None

    for i in range(len(atom_indices)-1):
        current_atom_index = atom_indices[i]
        next_atom_index = atom_indices[i + 1]
        intervening_tokens = tokens[current_atom_index + 1: next_atom_index]

        if not any(re.match(r'[\[\].()]', token) for token in intervening_tokens):
            after_digits = []
            for j in intervening_tokens[::-1]:
                if smiles2adjmat_new.ring_pattern.match(j):
                    break
                else:
                    after_digits.append(j)

            if '=' in after_digits:
                adjmat[i, i + 1] = adjmat[i + 1, i] = 2
            elif '#' in after_digits:
                adjmat[i, i + 1] = adjmat[i + 1, i] = 3
            else:
                adjmat[i, i + 1] = adjmat[i + 1, i] = 1

    for i in branch_open_atom_indices:
        current_atom_index = atom_indices[i]
        current_branch_level = branch_levels[i]

        if current_atom_index == 0:
            continue
        for j in range(i-1, -1, -1):
            if branch_levels[j] == current_branch_level - 1:
                intervening_tokens = []
                for k in range(current_atom_index, 0, -1):
                    if tokens[k] == "(":
                        break
                    intervening_tokens.append(tokens[k])

                if '=' in intervening_tokens:
                    adjmat[i, j] = adjmat[j, i] = 2
                elif '#' in intervening_tokens:
                    adjmat[i, j] = adjmat[j, i] = 3
                else:
                    adjmat[i, j] = adjmat[j, i] = 1
                break

    for i in branch_close_atom_indices:
        current_atom_index = atom_indices[i]
        current_branch_level = branch_levels[i]
        for j in range(i-1, -1, -1):
            if branch_levels[j] == current_branch_level:
                intervening_tokens = []
                for k in range(current_atom_index, 0, -1):
                    if tokens[k] == ")":
                        break
                    intervening_tokens.append(tokens[k])

                if '=' in intervening_tokens:
                    adjmat[i, j] = adjmat[j, i] = 2
                elif '#' in intervening_tokens:
                    adjmat[i, j] = adjmat[j, i] = 3
                else:
                    adjmat[i, j] = adjmat[j, i] = 1
                break

    if any(atom_info[i]["aromatic_input"] for i in atom_info):
        try:
            for r in return_rings(adjmat_to_adjlist(adjmat), max_size=10, remove_fused=True):
                if all(atom_info[_]["element"] in smiles2adjmat_new.aromatics for _ in r):
                    for ind in range(1, len(r)):
                        if (ind - 1) % 2 == 0:
                            adjmat[r[ind], r[ind-1]] = 2
                            adjmat[r[ind-1], r[ind]] = 2
        except Exception as e:
            print(f"WARNING: kekulization/aromatic assignment fallback used: {e}")
            for r in return_rings(adjmat_to_adjlist(adjmat), max_size=10, remove_fused=True):
                if all(atom_info[_]["element"] in smiles2adjmat_new.aromatics for _ in r):
                    for ind in range(1, len(r)):
                        if (ind - 1) % 2 == 0:
                            adjmat[r[ind], r[ind-1]] = 2
                            adjmat[r[ind-1], r[ind]] = 2

    adjmat, atom_info = add_hydrogens_new(
        adjmat,
        atom_info,
        atom_parse_meta,
        smiles2adjmat_new,
        el_valence,
        el_metals,
        el_expand_octet,
        OctetError,
        el_mass,
    )

    for idx in atom_parse_meta:
        if idx in atom_info:
            isotope = atom_parse_meta[idx]["isotope"]
            if atom_info[idx]["mass"] is None:
                atom_info[idx]["mass"] = float(isotope) if isotope is not None else el_mass[atom_info[idx]["element"]]

    provided = [atom_info[i]["atom_map"] for i in atom_info if atom_info[i]["atom_map"] is not None]
    if len(provided) != len(set(provided)):
        dupes = sorted({m for m in provided if provided.count(m) > 1})
        raise ValueError(f"Duplicate atom-map indices in input SMILES: {dupes}")

    adjmat, atom_info = reorder_by_mappings_new(adjmat, atom_info, np, canon_order)

    bond_electron_mat = adjmat.copy()
    for i in atom_info:
        bond_electron_mat[i, i] = el_valence[atom_info[i]["element"]] - atom_info[i]["formal_charge"] - sum(adjmat[i])

    return np.where(adjmat > 0, 1, 0), bond_electron_mat, atom_info


def properties_block_old():
    """
    Old code block from yarp/util/properties.py.

    Difference:
    - the repo already has `el_to_an` and `el_mass`.
    - no new parser-facing lookup table is required if we reuse them.
    """

    el_to_an = {"h": 1, "he": 2, "li": 3, "be": 4, "b": 5, "c": 6, "n": 7, "o": 8}
    el_mass = {'H': 1.00794, 'He': 4.002602, 'Li': 6.941, 'Be': 9.012182, 'B': 10.811, 'C': 12.011, 'N': 14.00674, 'O': 15.9994}
    return el_to_an, el_mass


def properties_block_new():
    """
    New code block for yarp/util/properties.py usage.

    Difference:
    - do not add a duplicate `ATOM_MASS` dictionary elsewhere.
    - import and use existing `el_to_an` and `el_mass`.
    """

    from yarp.util.properties import el_to_an, el_mass
    return el_to_an, el_mass


def add_hydrogens_old(adjmat, atom_info, smiles2adjmat, el_valence, el_metals, el_expand_octet, OctetError):
    """
    Old code from yarp/yarpecule/graph/smiles.py.

    Difference:
    - atom_info is a list.
    - H records are appended as list rows.
    """

    hydrogens_to_add = []
    bonded_hydrogens = []

    for atom in range(len(atom_info)):
        h_count = sum(1 for i in range(len(atom_info))
                      if atom_info[i][0].lower() == 'h' and adjmat[atom, i] != 0)
        bonded_hydrogens.append(h_count)

    for atom, info in enumerate(atom_info):
        element = info[0].lower()
        formal_charge = info[1]
        explicit_hydrogens = info[2]
        should_infer_hydrogens = info[5]
        valence_electrons = el_valence.get(element, None)

        if valence_electrons is None:
            print(f"Warning: Element '{element}' is not recognized or has an undefined valence.")
            hydrogens_to_add.append(0)
            continue
        elif element in el_metals:
            hydrogens_to_add.append(0)
            continue

        bonds = sum(adjmat[atom])

        if explicit_hydrogens is not None:
            if bonded_hydrogens[atom] > 0:
                needed_hydrogens = 0
            else:
                needed_hydrogens = explicit_hydrogens
        elif should_infer_hydrogens:
            if info[0] in smiles2adjmat.aromatics:
                bonded_neighbors = sum(1 for i in range(len(atom_info)) if adjmat[atom, i] > 0)
                if element in ['c', 'b', 'p']:
                    target_neighbors = 3
                elif element in ['n', 'o', 's']:
                    target_neighbors = 2
                else:
                    target_neighbors = 3
                needed_hydrogens = max(0, target_neighbors - bonded_neighbors)
            else:
                desired_electrons = 8 if element not in ['h', 'he'] else 2
                current_electrons = int(valence_electrons + bonds)
                needed_hydrogens = max(0, desired_electrons - current_electrons)

                if formal_charge > 0:
                    e = desired_electrons - 2*needed_hydrogens - 2*bonds
                    if (formal_charge - int(e/2)) <= 0:
                        needed_hydrogens += formal_charge
                    else:
                        needed_hydrogens -= formal_charge
                elif formal_charge < 0:
                    e = desired_electrons - 2*needed_hydrogens - 2*bonds
                    if (needed_hydrogens + formal_charge) >= 0:
                        needed_hydrogens += formal_charge
                    else:
                        needed_hydrogens -= formal_charge
        else:
            needed_hydrogens = 0

        if needed_hydrogens < 0:
            print("Warning: add_hydrogens() was unable to satisfy formal charge specification with hydrogens.")
        if (2*bonds + 2*needed_hydrogens) > 8 and not el_expand_octet[element]:
            raise OctetError(atom)

        hydrogens_to_add.append(needed_hydrogens)

    total_atoms = len(hydrogens_to_add) + int(sum(hydrogens_to_add))
    new_adjmat = np.zeros((total_atoms, total_atoms))
    new_adjmat[:len(adjmat), :len(adjmat)] = adjmat

    current_index = len(adjmat)
    for i, num_hydrogens in enumerate(hydrogens_to_add):
        for _ in range(num_hydrogens):
            new_adjmat[i, current_index] = new_adjmat[current_index, i] = 1
            current_index += 1

    return new_adjmat, atom_info + [['H', 0, None, None, None, True] for _ in range(int(sum(hydrogens_to_add)))]


def add_hydrogens_new(adjmat, atom_info, atom_parse_meta, smiles2adjmat, el_valence, el_metals, el_expand_octet, OctetError, el_mass):
    """
    New proposed code for yarp/yarpecule/graph/smiles.py.

    Difference:
    - atom_info is now a dict keyed by atom index.
    - new H records are dicts.
    - return shape is unchanged.
    - explicit hydrogen behavior is kept as close as possible to the old logic.
    """

    hydrogens_to_add = []
    bonded_hydrogens = []

    for atom in range(len(atom_info)):
        h_count = sum(1 for i in range(len(atom_info))
                      if atom_info[i]["element"] == 'h' and adjmat[atom, i] != 0)
        bonded_hydrogens.append(h_count)

    for atom in range(len(atom_info)):
        info = atom_info[atom]
        element = info["element"]
        formal_charge = info["formal_charge"]
        explicit_hydrogens = atom_parse_meta[atom]["explicit_hydrogens"]
        should_infer_hydrogens = atom_parse_meta[atom]["should_infer_hydrogens"]
        valence_electrons = el_valence.get(element, None)

        if valence_electrons is None:
            print(f"Warning: Element '{element}' is not recognized or has an undefined valence.")
            hydrogens_to_add.append(0)
            continue
        elif element in el_metals:
            hydrogens_to_add.append(0)
            continue

        bonds = sum(adjmat[atom])

        if explicit_hydrogens is not None:
            if bonded_hydrogens[atom] > 0:
                needed_hydrogens = 0
            else:
                needed_hydrogens = explicit_hydrogens
        elif should_infer_hydrogens:
            if info["aromatic_input"]:
                bonded_neighbors = sum(1 for i in range(len(atom_info)) if adjmat[atom, i] > 0)
                if element in ['c', 'b', 'p']:
                    target_neighbors = 3
                elif element in ['n', 'o', 's']:
                    target_neighbors = 2
                else:
                    target_neighbors = 3
                needed_hydrogens = max(0, target_neighbors - bonded_neighbors)
            else:
                desired_electrons = 8 if element not in ['h', 'he'] else 2
                current_electrons = int(valence_electrons + bonds)
                needed_hydrogens = max(0, desired_electrons - current_electrons)

                if formal_charge > 0:
                    e = desired_electrons - 2*needed_hydrogens - 2*bonds
                    if (formal_charge - int(e/2)) <= 0:
                        needed_hydrogens += formal_charge
                    else:
                        needed_hydrogens -= formal_charge
                elif formal_charge < 0:
                    e = desired_electrons - 2*needed_hydrogens - 2*bonds
                    if (needed_hydrogens + formal_charge) >= 0:
                        needed_hydrogens += formal_charge
                    else:
                        needed_hydrogens -= formal_charge
        else:
            needed_hydrogens = 0

        if needed_hydrogens < 0:
            print("Warning: add_hydrogens() was unable to satisfy formal charge specification with hydrogens.")
        if (2*bonds + 2*needed_hydrogens) > 8 and not el_expand_octet[element]:
            raise OctetError(atom)

        hydrogens_to_add.append(needed_hydrogens)

    total_atoms = len(hydrogens_to_add) + int(sum(hydrogens_to_add))
    new_adjmat = np.zeros((total_atoms, total_atoms))
    new_adjmat[:len(adjmat), :len(adjmat)] = adjmat

    current_index = len(adjmat)
    for i, num_hydrogens in enumerate(hydrogens_to_add):
        for _ in range(num_hydrogens):
            new_adjmat[i, current_index] = new_adjmat[current_index, i] = 1
            atom_info[current_index] = {
                "atom_index": current_index,
                "atom_map": None,
                "element": "h",
                "formal_charge": 0,
                "mass": el_mass["h"],
                "stereo": {"atom": None, "bonds": {}},
                "aromatic_input": False,
            }
            current_index += 1

    return new_adjmat, atom_info


def reorder_by_mappings_old(adjmat, atom_info, np, canon_order):
    """
    Old code from yarp/yarpecule/graph/smiles.py.

    Difference:
    - atom_info is still list-based.
    - atom_index is not tracked separately.
    """

    elements = [info[0].lower() for info in atom_info]
    mappings = [info[4] for info in atom_info]

    if all(m is None for m in mappings):
        return adjmat, atom_info

    hash_list = [m if m is not None else float('inf') for m in mappings]
    hash_list = -np.array(hash_list)

    ordered_elements, ordered_adjmat, ordered_hash, idx = canon_order(
        elements, adjmat, hash_list=hash_list, return_index=True
    )

    ordered_atom_info = [atom_info[i] for i in idx]

    return ordered_adjmat, ordered_atom_info


def reorder_by_mappings_new(adjmat, atom_info, np, canon_order):
    """
    New proposed code for yarp/yarpecule/graph/smiles.py.

    Difference:
    - atom_info is dict-based.
    - atom_index is refreshed after reorder.
    - user-provided atom_map values are preserved exactly.
    """

    elements = [atom_info[i]["element"] for i in atom_info]
    mappings = [atom_info[i]["atom_map"] for i in atom_info]

    if all(m is None for m in mappings):
        return adjmat, atom_info

    hash_list = [m if m is not None else float('inf') for m in mappings]
    hash_list = -np.array(hash_list)

    ordered_elements, ordered_adjmat, ordered_hash, idx = canon_order(
        elements, adjmat, hash_list=hash_list, return_index=True
    )

    ordered_atom_info = {}
    for new_idx, old_idx in enumerate(idx):
        record = dict(atom_info[old_idx])
        record["atom_index"] = new_idx
        ordered_atom_info[new_idx] = record

    return ordered_adjmat, ordered_atom_info


def mol_parse_old(mol, rdmolfiles, np):
    """
    Old code from yarp/yarpecule/input_parsers.py.

    Difference:
    - returns only elements, geometry, adjacency, and charge.
    """

    m = rdmolfiles.MolFromMolFile(mol)
    N_atoms = len(m.GetAtoms())
    elements = []
    geo = np.zeros((N_atoms, 3))
    q = 0

    for i in range(N_atoms):
        atom = m.GetAtomWithIdx(i)
        elements += [atom.GetSymbol()]
        coord = m.GetConformer().GetAtomPosition(i)
        geo[i] = np.array([coord.x, coord.y, coord.z])
        q += atom.GetFormalCharge()

    adj_mat = np.zeros((N_atoms, N_atoms))
    for i in [(_.GetBeginAtomIdx(), _.GetEndAtomIdx()) for _ in m.GetBonds()]:
        adj_mat[i[0], i[1]] = 1
        adj_mat[i[1], i[0]] = 1

    return elements, geo, adj_mat, q


def mol_parse_new(mol, rdmolfiles, np, el_mass):
    """
    New proposed code for yarp/yarpecule/input_parsers.py.

    Difference:
    - returns atom_info as a fifth value.
    - atom_info is created even if no user mapping is present.
    """

    m = rdmolfiles.MolFromMolFile(mol)
    N_atoms = len(m.GetAtoms())
    elements = []
    geo = np.zeros((N_atoms, 3))
    q = 0
    atom_info = {}

    for i in range(N_atoms):
        atom = m.GetAtomWithIdx(i)
        elements += [atom.GetSymbol()]
        coord = m.GetConformer().GetAtomPosition(i)
        geo[i] = np.array([coord.x, coord.y, coord.z])
        q += atom.GetFormalCharge()

        atom_map = None
        if atom.HasProp("molAtomMapNumber"):
            atom_map = int(atom.GetProp("molAtomMapNumber"))

        isotope = atom.GetIsotope()
        mass = float(isotope) if isotope else el_mass[atom.GetSymbol().lower()]

        atom_info[i] = {
            "atom_index": i,
            "atom_map": atom_map,
            "element": atom.GetSymbol().lower(),
            "formal_charge": atom.GetFormalCharge(),
            "mass": mass,
            "stereo": {"atom": None, "bonds": {}},
            "aromatic_input": atom.GetIsAromatic(),
        }

    adj_mat = np.zeros((N_atoms, N_atoms))
    for i in [(_.GetBeginAtomIdx(), _.GetEndAtomIdx()) for _ in m.GetBonds()]:
        adj_mat[i[0], i[1]] = 1
        adj_mat[i[1], i[0]] = 1

    return elements, geo, adj_mat, q, atom_info


def xyz_from_smiles_old(smiles, mode, smiles2adjmat, BondType, MolFromSmiles, rdchem, Atom, el_to_an, el_n_expand_octet, el_expand_octet, OctetError, AddHs, AllChem, np):
    """
    Old code from yarp/yarpecule/input_parsers.py.

    Difference:
    - returns a 4-tuple.
    - RDKit path does not build atom_info.
    """

    if not hasattr(xyz_from_smiles_old, "bond_to_type"):
        xyz_from_smiles_old.bond_to_type = {0: BondType.DATIVE, 1: BondType.SINGLE, 2: BondType.DOUBLE,
                                            3: BondType.TRIPLE, 4: BondType.QUADRUPLE, 5: BondType.QUINTUPLE,
                                            6: BondType.HEXTUPLE}

    if mode == "yarp":
        adj_mat, bemat, atom_info = smiles2adjmat(smiles)
        elements = [_[0].lower() for _ in atom_info]
        fc = [0 if _[1] is None else int(_[1]) for _ in atom_info]
        q = int(sum(fc))
        e_exp = np.array([el_n_expand_octet[_] for _ in elements])
        e = np.sum(2*bemat, axis=1)-np.diag(bemat)
        violations = [count for count, _ in enumerate(e) if not el_expand_octet[elements[count]] and _-e_exp[count] > 0]
        if violations:
            raise OctetError(violations)

        mol = MolFromSmiles("C")
        mol = rdchem.RWMol(mol)
        mol.RemoveAtom(0)
        [mol.AddAtom(Atom(el_to_an[_.lower()])) for _ in elements]

        for count_j, j in enumerate(adj_mat):
            for count_k, k in enumerate(j):
                if count_k < count_j:
                    if k != 0:
                        mol.AddBond(count_j, count_k, xyz_from_smiles_old.bond_to_type[bemat[count_j, count_k]])
                else:
                    break

        for count_j, j in enumerate(bemat):
            mol.GetAtomWithIdx(count_j).SetFormalCharge(int(fc[count_j]))
            mol.GetAtomWithIdx(count_j).SetNumRadicalElectrons(int(j[count_j] % 2))

        mol.UpdatePropertyCache()
        AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
        N_atoms = len(mol.GetAtoms())
        geo = np.zeros((N_atoms, 3))
        for i in range(N_atoms):
            coord = mol.GetConformer().GetAtomPosition(i)
            geo[i] = np.array([coord.x, coord.y, coord.z])

        return elements, geo, adj_mat, q

    m = MolFromSmiles(smiles)
    m = AddHs(m)
    AllChem.EmbedMolecule(m, randomSeed=0xf00d)
    N_atoms = len(m.GetAtoms())
    elements = []
    geo = np.zeros((N_atoms, 3))
    q = 0

    for i in range(N_atoms):
        atom = m.GetAtomWithIdx(i)
        elements += [atom.GetSymbol()]
        coord = m.GetConformer().GetAtomPosition(i)
        geo[i] = np.array([coord.x, coord.y, coord.z])
        q += atom.GetFormalCharge()

    adj_mat = np.zeros((N_atoms, N_atoms))
    for i in [(_.GetBeginAtomIdx(), _.GetEndAtomIdx()) for _ in m.GetBonds()]:
        adj_mat[i[0], i[1]] = 1
        adj_mat[i[1], i[0]] = 1

    return elements, geo, adj_mat, q


def xyz_from_smiles_new(smiles, mode, smiles2adjmat, BondType, MolFromSmiles, rdchem, Atom, el_to_an, el_n_expand_octet, el_expand_octet, OctetError, AddHs, AllChem, np, el_mass):
    """
    New proposed code for yarp/yarpecule/input_parsers.py.

    Difference:
    - still keeps fallback logic out of this function.
    - returns atom_info as a fifth value in both modes.
    """

    if not hasattr(xyz_from_smiles_new, "bond_to_type"):
        xyz_from_smiles_new.bond_to_type = {0: BondType.DATIVE, 1: BondType.SINGLE, 2: BondType.DOUBLE,
                                            3: BondType.TRIPLE, 4: BondType.QUADRUPLE, 5: BondType.QUINTUPLE,
                                            6: BondType.HEXTUPLE}

    if mode == "yarp":
        adj_mat, bemat, atom_info = smiles2adjmat(smiles)
        elements = [atom_info[i]["element"] for i in atom_info]
        fc = [int(atom_info[i]["formal_charge"]) for i in atom_info]
        q = int(sum(fc))
        e_exp = np.array([el_n_expand_octet[_] for _ in elements])
        e = np.sum(2*bemat, axis=1)-np.diag(bemat)
        violations = [count for count, _ in enumerate(e) if not el_expand_octet[elements[count]] and _-e_exp[count] > 0]
        if violations:
            raise OctetError(violations)

        mol = MolFromSmiles("C")
        mol = rdchem.RWMol(mol)
        mol.RemoveAtom(0)
        [mol.AddAtom(Atom(el_to_an[_.lower()])) for _ in elements]

        for count_j, j in enumerate(adj_mat):
            for count_k, k in enumerate(j):
                if count_k < count_j:
                    if k != 0:
                        mol.AddBond(count_j, count_k, xyz_from_smiles_new.bond_to_type[bemat[count_j, count_k]])
                else:
                    break

        for count_j, j in enumerate(bemat):
            mol.GetAtomWithIdx(count_j).SetFormalCharge(int(fc[count_j]))
            mol.GetAtomWithIdx(count_j).SetNumRadicalElectrons(int(j[count_j] % 2))

        mol.UpdatePropertyCache()
        AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
        N_atoms = len(mol.GetAtoms())
        geo = np.zeros((N_atoms, 3))
        for i in range(N_atoms):
            coord = mol.GetConformer().GetAtomPosition(i)
            geo[i] = np.array([coord.x, coord.y, coord.z])

        return elements, geo, adj_mat, q, atom_info

    m = MolFromSmiles(smiles)
    m = AddHs(m)
    AllChem.EmbedMolecule(m, randomSeed=0xf00d)
    N_atoms = len(m.GetAtoms())
    elements = []
    geo = np.zeros((N_atoms, 3))
    q = 0
    atom_info = {}

    for i in range(N_atoms):
        atom = m.GetAtomWithIdx(i)
        elements += [atom.GetSymbol()]
        coord = m.GetConformer().GetAtomPosition(i)
        geo[i] = np.array([coord.x, coord.y, coord.z])
        q += atom.GetFormalCharge()
        isotope = atom.GetIsotope()
        mass = float(isotope) if isotope else el_mass[atom.GetSymbol().lower()]
        atom_map = None
        if atom.HasProp("molAtomMapNumber"):
            atom_map = int(atom.GetProp("molAtomMapNumber"))
        atom_info[i] = {
            "atom_index": i,
            "atom_map": atom_map,
            "element": atom.GetSymbol().lower(),
            "formal_charge": atom.GetFormalCharge(),
            "mass": mass,
            "stereo": {"atom": None, "bonds": {}},
            "aromatic_input": atom.GetIsAromatic(),
        }

    adj_mat = np.zeros((N_atoms, N_atoms))
    for i in [(_.GetBeginAtomIdx(), _.GetEndAtomIdx()) for _ in m.GetBonds()]:
        adj_mat[i[0], i[1]] = 1
        adj_mat[i[1], i[0]] = 1

    return elements, geo, adj_mat, q, atom_info


def yarpecule_init_old(self, mol, mode='yarp', canon=True):
    """
    Old code from yarp/yarpecule/yarpecule.py.

    Difference:
    - __init__ does not accept strict or atom_info.
    - _atom_info is declared on the object, but not guaranteed to be populated by _read_structure().
    """

    self._geo = None
    self._elements = None
    self._q = 0
    self._masses = None
    self._adj_mat = None
    self._atom_info = {}

    self._read_structure(mol, mode)

    self._atom_hashes = None
    self._mapping = None

    self._order_atoms(canon=canon)

    self._lewis_struct = None
    self._bond_order_dict = None
    self._yarpecule_hash = None

    self._gen_lewis_struct()

    self._canon_smi = None
    self._map_smi = None
    self._inchi = None


def yarpecule_init_new(self, mol, mode='yarp', canon=True, strict=False, atom_info=None):
    """
    New proposed code for yarp/yarpecule/yarpecule.py.

    Difference:
    - adds strict=False for caller-side smiles fallback behavior.
    - adds atom_info=None so enum/join/separate can carry parent metadata forward.
    - __init__ still declares _atom_info, but population remains inside _read_structure().
    """

    self._geo = None
    self._elements = None
    self._q = 0
    self._masses = None
    self._adj_mat = None
    self._atom_info = None

    self._read_structure(mol, mode, strict=strict, atom_info=atom_info)

    self._atom_hashes = None
    self._mapping = None

    self._order_atoms(canon=canon)

    self._lewis_struct = None
    self._bond_order_dict = None
    self._yarpecule_hash = None

    self._gen_lewis_struct()

    self._canon_smi = None
    self._map_smi = None
    self._inchi = None


def clone_atom_info_new(atom_info, elements, el_mass):
    """
    Draft helper for yarpecule-side metadata normalization.

    Difference:
    - centralizes the "copy carried records or generate defaults" logic so
      tuple, xyz, join, separate, and enum all use the same shape.
    - supports either a dict keyed by local atom index or a list/tuple already
      ordered to local atom index.
    """

    normalized = {}
    for i in range(len(elements)):
        if atom_info is None:
            record = {}
        elif isinstance(atom_info, dict):
            record = dict(atom_info[i])
        else:
            record = dict(atom_info[i])

        normalized[i] = {
            "atom_index": i,
            "atom_map": record.get("atom_map", None),
            "element": elements[i].lower(),
            "formal_charge": record.get("formal_charge", None),
            "mass": record.get("mass", el_mass[elements[i].lower()]),
            "stereo": {
                "atom": record.get("stereo", {}).get("atom", None),
                "bonds": dict(record.get("stereo", {}).get("bonds", {})),
            },
            "aromatic_input": record.get("aromatic_input", False),
        }

    return normalized


def yarpecule_read_structure_old(self, mol, mode, np, xyz_parse, table_generator, xyz_q_parse, mol_parse, xyz_from_smiles, el_mass):
    """
    Old code from yarp/yarpecule/yarpecule.py.

    Difference:
    - no strict flag.
    - xyz/mol paths do not create _atom_info.
    - smiles fallback handling is broad and unstructured.
    """

    if isinstance(mol, (tuple, list)) and len(mol) == 4:
        if (isinstance(mol[0], np.ndarray) is False or
            isinstance(mol[1], np.ndarray) is False or
            isinstance(mol[2], list) is False or
                isinstance(mol[3], int) is False):
            raise TypeError("The yarpecule constructor expects a string or a tuple containing (adj_mat,geo,elements,q).")
        elif (len(mol[0]) != len(mol[1]) or len(mol[0]) != len(mol[2])):
            raise TypeError("The size of the adjacency array, geometry array, and elements list do not match.")
        self._adj_mat = mol[0]
        self._geo = mol[1]
        self._elements = mol[2]
        self._q = mol[3]
    elif len(mol) > 4 and mol[-4:] == ".xyz":
        self._elements, self._geo = xyz_parse(mol)
        self._adj_mat = table_generator(self._elements, self._geo)
        self._q = xyz_q_parse(mol)
    elif len(mol) > 4 and mol[-4:] == ".mol":
        self._elements, self._geo, self._adj_mat, self._q = mol_parse(mol)
    else:
        try:
            self._elements, self._geo, self._adj_mat, self._q = xyz_from_smiles(mol, mode=mode)
        except Exception:
            raise TypeError("The yarpecule constructor expects either an xyz file, mol file, or a smiles string.")

    self._elements = [_.lower() for _ in self._elements]
    self._masses = np.array([el_mass[_] for _ in self._elements])


def yarpecule_read_structure_new(self, mol, mode, strict, atom_info, np, xyz_parse, table_generator, xyz_q_parse, mol_parse, xyz_from_smiles, el_mass):
    """
    New proposed code for yarp/yarpecule/yarpecule.py.

    Difference:
    - adds call-site `strict=False` fallback behavior.
    - accepts atom_info=None and also supports tuple-carried atom_info so
      enum/join/separate can pass metadata through without a side channel.
    - every yarpecule gets full _atom_info.
    - xyz path creates atom_info after adjacency generation.
    """

    carried_atom_info = atom_info

    if isinstance(mol, (tuple, list)) and len(mol) in [4, 5]:
        if (isinstance(mol[0], np.ndarray) is False or
            isinstance(mol[1], np.ndarray) is False or
            isinstance(mol[2], list) is False or
                isinstance(mol[3], int) is False):
            raise TypeError("The yarpecule constructor expects a string or a tuple containing (adj_mat,geo,elements,q).")
        elif (len(mol[0]) != len(mol[1]) or len(mol[0]) != len(mol[2])):
            raise TypeError("The size of the adjacency array, geometry array, and elements list do not match.")
        elif len(mol) == 5 and carried_atom_info is not None:
            raise TypeError("Pass carried atom_info either as the fifth tuple item or as atom_info=..., not both.")

        self._adj_mat = mol[0]
        self._geo = mol[1]
        self._elements = mol[2]
        self._q = mol[3]
        if len(mol) == 5:
            carried_atom_info = mol[4]

        self._atom_info = clone_atom_info_new(carried_atom_info, self._elements, el_mass)
    elif len(mol) > 4 and mol[-4:] == ".xyz":
        self._elements, self._geo = xyz_parse(mol)
        self._adj_mat = table_generator(self._elements, self._geo)
        self._q = xyz_q_parse(mol)
        self._atom_info = clone_atom_info_new(carried_atom_info, self._elements, el_mass)
    elif len(mol) > 4 and mol[-4:] == ".mol":
        self._elements, self._geo, self._adj_mat, self._q, self._atom_info = mol_parse(mol)
    else:
        try:
            self._elements, self._geo, self._adj_mat, self._q, self._atom_info = xyz_from_smiles(mol, mode=mode)
        except Exception as e:
            if mode == "yarp" and strict is False:
                print(f"WARNING: yarp SMILES parsing failed, falling back to RDKit: {e}")
                self._elements, self._geo, self._adj_mat, self._q, self._atom_info = xyz_from_smiles(mol, mode="rdkit")
            else:
                raise TypeError("The yarpecule constructor expects either an xyz file, mol file, or a smiles string.")

    self._elements = [_.lower() for _ in self._elements]
    self._masses = np.array([el_mass[_] for _ in self._elements])
    self._atom_info = clone_atom_info_new(self._atom_info, self._elements, el_mass)


def yarpecule_order_atoms_old(self, canon, canon_order, atom_hash, np):
    """
    Old code from yarp/yarpecule/yarpecule.py.

    Difference:
    - _atom_info is not reordered.
    """

    if canon:
        self._elements, self._adj_mat, self._atom_hashes, self._mapping, self._geo, self._masses = canon_order(
            self._elements, self._adj_mat, masses=self._masses, things_to_order=[self._geo, self._masses])
    else:
        self._atom_hashes = np.array([atom_hash(_, self._adj_mat, self._masses) for _ in range(len(self._elements))])


def yarpecule_order_atoms_new(self, canon, canon_order, atom_hash, np):
    """
    New proposed code for yarp/yarpecule/yarpecule.py.

    Difference:
    - _atom_info is reordered with the graph.
    - atom_index is refreshed.
    - existing user maps are preserved.
    - missing maps are generated at yarpecule creation regardless of canon.
    - reorder carries those maps forward; it is not the place that first creates them.
    """

    if self._atom_info is not None:
        used = {self._atom_info[i]["atom_map"] for i in self._atom_info if self._atom_info[i]["atom_map"] is not None}
        next_map = 0
        for i in self._atom_info:
            if self._atom_info[i]["atom_map"] is None:
                while next_map in used:
                    next_map += 1
                self._atom_info[i]["atom_map"] = next_map
                used.add(next_map)
                next_map += 1

    if canon:
        self._elements, self._adj_mat, self._atom_hashes, self._mapping, self._geo, self._masses = canon_order(
            self._elements, self._adj_mat, masses=self._masses, things_to_order=[self._geo, self._masses])

        if self._atom_info is not None:
            reordered_atom_info = {}
            for new_idx, old_idx in enumerate(self._mapping):
                record = dict(self._atom_info[old_idx])
                record["atom_index"] = new_idx
                reordered_atom_info[new_idx] = record
            self._atom_info = reordered_atom_info
    else:
        self._atom_hashes = np.array([atom_hash(_, self._adj_mat, self._masses) for _ in range(len(self._elements))])


def carry_atom_info_by_index_new(parent_atom_info):
    """
    Draft helper for enum-created products that keep atom ordering unchanged.

    Difference:
    - makes it explicit that enumeration should preserve parent metadata by
      copying records indexed by the current local atom order.
    """

    return {
        i: {
            **dict(parent_atom_info[i]),
            "atom_index": i,
            "stereo": {
                "atom": parent_atom_info[i].get("stereo", {}).get("atom", None),
                "bonds": dict(parent_atom_info[i].get("stereo", {}).get("bonds", {})),
            },
        }
        for i in parent_atom_info
    }


def join_atom_info_new(all_y):
    """
    Draft helper for yarpecule.join().

    Difference:
    - merges `_atom_info` blocks from all parents into a single local index
      space so join does not silently drop user maps or isotope-adjusted masses.
    """

    merged_atom_info = {}
    offset = 0
    for y in all_y:
        for i in range(len(y.elements)):
            merged_atom_info[offset + i] = {
                **dict(y._atom_info[i]),
                "atom_index": offset + i,
                "stereo": {
                    "atom": y._atom_info[i].get("stereo", {}).get("atom", None),
                    "bonds": {
                        offset + nbr: flag
                        for nbr, flag in y._atom_info[i].get("stereo", {}).get("bonds", {}).items()
                    },
                },
            }
        offset += len(y.elements)

    return merged_atom_info


def separate_atom_info_new(parent_atom_info, group):
    """
    Draft helper for yarpecule.separate().

    Difference:
    - slices `_atom_info` onto fragment-local indices so separate() products
      retain atom maps and isotope-adjusted masses.
    """

    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(group)}
    fragment_atom_info = {}
    for old_idx in group:
        new_idx = old_to_new[old_idx]
        fragment_atom_info[new_idx] = {
            **dict(parent_atom_info[old_idx]),
            "atom_index": new_idx,
            "stereo": {
                "atom": parent_atom_info[old_idx].get("stereo", {}).get("atom", None),
                "bonds": {
                    old_to_new[nbr]: flag
                    for nbr, flag in parent_atom_info[old_idx].get("stereo", {}).get("bonds", {}).items()
                    if nbr in old_to_new
                },
            },
        }

    return fragment_atom_info


def mol_write_yp_old(file, elements, geo, bond_mat, adj_mat, return_formals, np):
    """
    Old code from yarp/util/write_files.py.

    Difference:
    - no atom_info argument.
    - mol atom-map field is never written.
    """

    if len(elements) >= 1000:
        print("ERROR in mol_write: the V2000 format can only accomodate up to 1000 atoms per molecule.")
        return
    mol_dict = {3: 1, 2: 2, 1: 3, -1: 5, -2: 6, -3: 7, 0: 0}
    open_cond = 'w'

    base_name = file.split(".")
    if len(base_name) > 1:
        base_name = ".".join(base_name[:-1])
    else:
        base_name = base_name[0]

    keep_lone = [count_i for count_i, i in enumerate(bond_mat) if i[count_i] % 2 == 1]
    fc = list(return_formals(bond_mat, elements))
    chrg = len([i for i in fc if i != 0])
    valence = []
    for count_i, i in enumerate(bond_mat):
        bond = 0
        for count_j, j in enumerate(i):
            if count_i != count_j:
                bond = bond + int(j)
        valence.append(bond)

    with open(file, open_cond) as f:
        f.write(f"{base_name}\n")
        f.write("  yarp{}*3D*\n".format(__import__('datetime').datetime.now().strftime("%m%d%H%M%S")))
        f.write("\n")
        f.write("{:>3d}{:>3d}  0  0  0  0  0  0  0  0  1 V2000\n".format(len(elements), int(np.sum(adj_mat/2.0))))

        for count_i, i in enumerate(elements):
            f.write(" {:> 9.4f} {:> 9.4f} {:> 9.4f} {:<3s} 0 {:>2d}  0  0  0  {:>2d}  0  0  0  0  0  0\n".format(
                geo[count_i][0], geo[count_i][1], geo[count_i][2], i.capitalize(), mol_dict[fc[count_i]], valence[count_i]))

        bonds = [(count_i, count_j) for count_i, i in enumerate(adj_mat)
                 for count_j, j in enumerate(i) if j == 1 and count_j > count_i]
        for i in bonds:
            bond_order = int(bond_mat[i[0], i[1]])
            if bond_order == 0:
                bond_order = 1
            f.write("{:>3d}{:>3d}{:>3d}  0  0  0  0\n".format(i[0]+1, i[1]+1, bond_order))

        if len(keep_lone) > 0:
            if len(keep_lone) == 1:
                f.write("M  RAD{:>3d}{:>4d}{:>4d}\n".format(1, keep_lone[0]+1, 2))
            elif len(keep_lone) == 2:
                f.write("M  RAD{:>3d}{:>4d}{:>4d}{:>4d}{:>4d}\n".format(2, keep_lone[0]+1, 2, keep_lone[1]+1, 2))

        if chrg > 0:
            if chrg == 1:
                charge = [i for i in fc if i != 0][0]
                f.write("M  CHG{:>3d}{:>4d}{:>4d}\n".format(1, fc.index(charge)+1, int(charge)))
            else:
                info = ""
                fc_counter = 0
                for count_c, charge in enumerate(fc):
                    if charge != 0:
                        if (fc_counter % 8 == 0):
                            info += "M  CHG{:>3d}".format(chrg - fc_counter if chrg - fc_counter <= 8 else 8)
                        info += '{:>4d}{:>4d}'.format(count_c+1, int(charge))
                        fc_counter += 1
                info += '\n'
                f.write(info)

        f.write("M  END\n$$$$\n")


def mol_write_yp_new(file, elements, geo, bond_mat, adj_mat, atom_info, return_formals, np):
    """
    New proposed code for yarp/util/write_files.py.

    Difference:
    - adds atom_info argument.
    - writes map field when atom_info is available.
    - planned implementation should validate map-field placement by round-tripping through RDKit.
    - if validation fails, warn and fall back to the current map-free MOL layout.
    """

    if len(elements) >= 1000:
        print("ERROR in mol_write: the V2000 format can only accomodate up to 1000 atoms per molecule.")
        return
    mol_dict = {3: 1, 2: 2, 1: 3, -1: 5, -2: 6, -3: 7, 0: 0}
    open_cond = 'w'

    base_name = file.split(".")
    if len(base_name) > 1:
        base_name = ".".join(base_name[:-1])
    else:
        base_name = base_name[0]

    keep_lone = [count_i for count_i, i in enumerate(bond_mat) if i[count_i] % 2 == 1]
    fc = list(return_formals(bond_mat, elements))
    chrg = len([i for i in fc if i != 0])
    valence = []
    for count_i, i in enumerate(bond_mat):
        bond = 0
        for count_j, j in enumerate(i):
            if count_i != count_j:
                bond = bond + int(j)
        valence.append(bond)

    with open(file, open_cond) as f:
        f.write(f"{base_name}\n")
        f.write("  yarp{}*3D*\n".format(__import__('datetime').datetime.now().strftime("%m%d%H%M%S")))
        f.write("\n")
        f.write("{:>3d}{:>3d}  0  0  0  0  0  0  0  0  1 V2000\n".format(len(elements), int(np.sum(adj_mat/2.0))))

        for count_i, i in enumerate(elements):
            map_field = 0
            if atom_info is not None and atom_info[count_i]["atom_map"] is not None:
                map_field = int(atom_info[count_i]["atom_map"])
            f.write(" {:> 9.4f} {:> 9.4f} {:> 9.4f} {:<3s} 0 {:>2d}  0  0  0  {:>2d}  0  0  0  0 {:>3d}  0\n".format(
                geo[count_i][0], geo[count_i][1], geo[count_i][2], i.capitalize(), mol_dict[fc[count_i]], valence[count_i], map_field))

        bonds = [(count_i, count_j) for count_i, i in enumerate(adj_mat)
                 for count_j, j in enumerate(i) if j == 1 and count_j > count_i]
        for i in bonds:
            bond_order = int(bond_mat[i[0], i[1]])
            if bond_order == 0:
                bond_order = 1
            f.write("{:>3d}{:>3d}{:>3d}  0  0  0  0\n".format(i[0]+1, i[1]+1, bond_order))

        if len(keep_lone) > 0:
            if len(keep_lone) == 1:
                f.write("M  RAD{:>3d}{:>4d}{:>4d}\n".format(1, keep_lone[0]+1, 2))
            elif len(keep_lone) == 2:
                f.write("M  RAD{:>3d}{:>4d}{:>4d}{:>4d}{:>4d}\n".format(2, keep_lone[0]+1, 2, keep_lone[1]+1, 2))

        if chrg > 0:
            if chrg == 1:
                charge = [i for i in fc if i != 0][0]
                f.write("M  CHG{:>3d}{:>4d}{:>4d}\n".format(1, fc.index(charge)+1, int(charge)))
            else:
                info = ""
                fc_counter = 0
                for count_c, charge in enumerate(fc):
                    if charge != 0:
                        if (fc_counter % 8 == 0):
                            info += "M  CHG{:>3d}".format(chrg - fc_counter if chrg - fc_counter <= 8 else 8)
                        info += '{:>4d}{:>4d}'.format(count_c+1, int(charge))
                        fc_counter += 1
                info += '\n'
                f.write(info)

        f.write("M  END\n$$$$\n")

    # planned validation step after first implementation:
    # 1. read file back with RDKit
    # 2. inspect molAtomMapNumber on atoms
    # 3. if round-trip fails, warn and rewrite using the old map-free writer layout


def xyz_write_old(name, elements, geo):
    """
    Old code from yarp/util/write_files.py.

    Difference:
    - second line is always blank.
    """

    out = open(name, 'w+')
    elements = [el.upper() for el in elements]
    out.write('{}\n\n'.format(len(elements)))
    for count_i, i in enumerate(elements):
        out.write('{} {} {} {}\n'.format(i, geo[count_i][0], geo[count_i][1], geo[count_i][2]))
    out.close()


def xyz_write_new(name, elements, geo, comment=None, append_opt=False):
    """
    New proposed code for yarp/util/write_files.py.

    Difference:
    - adds optional comment line content.
    - comment should hold the RDKit canonical smiles.
    - preserves append_opt so existing multi-structure XYZ workflows keep working.
    """

    if append_opt is False:
        out = open(name, 'w+')
    else:
        out = open(name, 'a+')
    elements = [el.upper() for el in elements]
    if comment is None:
        comment = ""
    out.write('{}\n{}\n'.format(len(elements), comment))
    for count_i, i in enumerate(elements):
        out.write('{} {} {} {}\n'.format(i, geo[count_i][0], geo[count_i][1], geo[count_i][2]))
    out.close()


def yarpecule_get_smiles_old(self, os, Chem, mol_write_yp):
    """
    Old code from yarp/yarpecule/yarpecule.py.

    Difference:
    - no verbose flag.
    - mapped smiles uses RDKit atom index, not _atom_info atom_map.
    """

    tmp_file = ".tmp.mol"
    mol_write_yp(tmp_file, self.elements, self.geo, self.bond_mats[0], self.adj_mat)

    mol1 = Chem.rdmolfiles.MolFromMolFile(tmp_file, removeHs=True)
    atoms = mol1.GetNumAtoms()
    for idx in range(atoms):
        mol1.GetAtomWithIdx(idx).ClearProp("molAtomMapNumber")
    self._canon_smi = Chem.MolToSmiles(mol1, canonical=True)

    mol2 = Chem.rdmolfiles.MolFromMolFile(tmp_file, removeHs=False)
    atoms = mol2.GetNumAtoms()
    for idx in range(atoms):
        mol2.GetAtomWithIdx(idx).SetProp("molAtomMapNumber", str(mol2.GetAtomWithIdx(idx).GetIdx()))
    self._map_smi = Chem.MolToSmiles(mol2)

    os.remove(tmp_file)


def yarpecule_get_smiles_new(self, os, Chem, mol_write_yp, verbose=False):
    """
    New proposed code for yarp/yarpecule/yarpecule.py.

    Difference:
    - adds verbose=False.
    - canonical and mapped smiles still come from RDKit.
    - this is not a new YARP SMILES writer path.
    - mapped smiles uses _atom_info atom_map values.
    - verbose prints full RDKit mol dumps and atom_index -> atom_map table.
    - this is the planned fix for user-provided atom maps not appearing in final mapped outputs.
    """

    tmp_file = ".tmp.mol"
    mol_write_yp(tmp_file, self.elements, self.geo, self.bond_mats[0], self.adj_mat, atom_info=self._atom_info)

    mol1 = Chem.rdmolfiles.MolFromMolFile(tmp_file, removeHs=True)
    if verbose:
        print("RDKit mol dump before mapping:")
        for line in Chem.MolToMolBlock(mol1).splitlines():
            print(line)

    atoms = mol1.GetNumAtoms()
    for idx in range(atoms):
        mol1.GetAtomWithIdx(idx).ClearProp("molAtomMapNumber")
    self._canon_smi = Chem.MolToSmiles(mol1, canonical=True)

    mol2 = Chem.rdmolfiles.MolFromMolFile(tmp_file, removeHs=False)
    atoms = mol2.GetNumAtoms()
    for idx in range(atoms):
        mol2.GetAtomWithIdx(idx).SetProp("molAtomMapNumber", str(self._atom_info[idx]["atom_map"]))

    if verbose:
        print("RDKit mol dump after mapping:")
        for line in Chem.MolToMolBlock(mol2).splitlines():
            print(line)
        print("atom_index -> atom_map")
        for idx in range(atoms):
            print(idx, "->", self._atom_info[idx]["atom_map"])

    self._map_smi = Chem.MolToSmiles(mol2, canonical=True)

    if verbose:
        print("written mol file contents:")
        with open(tmp_file, "r") as f:
            for line in f:
                print(line.rstrip("\n"))

    os.remove(tmp_file)


def yarpecule_export_geometry_old(self, filename, format, xyz_write, mol_write_yp):
    """
    Old code from yarp/yarpecule/yarpecule.py.

    Difference:
    - xyz writer gets no comment.
    - mol writer gets no atom_info.
    """

    if format == 'xyz':
        xyz_write(filename, self.elements, self.geo)
    elif format == 'mol':
        mol_write_yp(filename, self.elements, self.geo, self.bond_mats[0], self.adj_mat)
    else:
        raise RuntimeError("Valid export formats: xyz or mol")


def yarpecule_export_geometry_new(self, filename, format, xyz_write, mol_write_yp):
    """
    New proposed code for yarp/yarpecule/yarpecule.py.

    Difference:
    - xyz comment gets RDKit canonical smiles.
    - mol writer gets atom_info.
    """

    if format == 'xyz':
        if self._canon_smi is None:
            self.get_smiles()
        xyz_write(filename, self.elements, self.geo, comment=self._canon_smi)
    elif format == 'mol':
        mol_write_yp(filename, self.elements, self.geo, self.bond_mats[0], self.adj_mat, atom_info=self._atom_info)
    else:
        raise RuntimeError("Valid export formats: xyz or mol")


def yarpecule_join_old(self, yarpecules, canon, prepare_list, np, merge_arrays, yarpecule):
    """
    Old code from yarp/yarpecule/yarpecule.py.

    Difference:
    - join rebuilds from a 4-tuple and drops parent `_atom_info`.
    """

    yarpecules = prepare_list(yarpecules)
    all_y = [self] + yarpecules

    adj_mat = merge_arrays([y.adj_mat for y in all_y])
    geo = np.vstack([y.geo for y in all_y])
    elements = [e for y in all_y for e in y.elements]
    q = int(sum([y.q for y in all_y]))

    return yarpecule((adj_mat, geo, elements, q), canon=canon)


def yarpecule_join_new(self, yarpecules, canon, prepare_list, np, merge_arrays, yarpecule):
    """
    New proposed code for yarp/yarpecule/yarpecule.py.

    Difference:
    - join carries merged parent `_atom_info` into the tuple-created yarpecule.
    """

    yarpecules = prepare_list(yarpecules)
    all_y = [self] + yarpecules

    adj_mat = merge_arrays([y.adj_mat for y in all_y])
    geo = np.vstack([y.geo for y in all_y])
    elements = [e for y in all_y for e in y.elements]
    q = int(sum([y.q for y in all_y]))
    atom_info = join_atom_info_new(all_y)

    return yarpecule((adj_mat, geo, elements, q, atom_info), canon=canon)


def yarpecule_separate_old(self, canon, graph_seps, np, return_formals, yarpecule):
    """
    Old code from yarp/yarpecule/yarpecule.py.

    Difference:
    - separate rebuilds from 4-tuples and drops `_atom_info` on all outputs.
    """

    gs = graph_seps(self.adj_mat)
    groups = []
    loop_ind = []
    for i in range(len(gs)):
        if i not in loop_ind:
            new_group = [count_j for count_j, j in enumerate(gs[i, :]) if j >= 0]
            loop_ind += new_group
            groups += [new_group]

    if len(groups) == 1:
        return [yarpecule((self.adj_mat, self.geo, self.elements, self.q), canon=canon)]

    mols = []
    for g in groups:
        frag_adj = self.adj_mat[g][:, g]
        frag_e = [self.elements[ind] for ind in g]
        N_atom = len(g)
        frag_geo = np.zeros([N_atom, 3])
        for count_i, i in enumerate(g):
            frag_geo[count_i, :] = self.geo[i, :]
        frag_bem = [self.bond_mats[0][i] for i in g]
        frag_formals = return_formals(frag_bem, frag_e)
        frag_q = int(sum(frag_formals))
        mols.append(yarpecule((frag_adj, frag_geo, frag_e, frag_q), canon=canon))

    return mols


def yarpecule_separate_new(self, canon, graph_seps, np, return_formals, yarpecule):
    """
    New proposed code for yarp/yarpecule/yarpecule.py.

    Difference:
    - separate carries fragment-local `_atom_info` so downstream distance/state
      workflows keep metadata after splitting disconnected species.
    """

    gs = graph_seps(self.adj_mat)
    groups = []
    loop_ind = []
    for i in range(len(gs)):
        if i not in loop_ind:
            new_group = [count_j for count_j, j in enumerate(gs[i, :]) if j >= 0]
            loop_ind += new_group
            groups += [new_group]

    if len(groups) == 1:
        atom_info = carry_atom_info_by_index_new(self._atom_info)
        return [yarpecule((self.adj_mat, self.geo, self.elements, self.q, atom_info), canon=canon)]

    mols = []
    for g in groups:
        frag_adj = self.adj_mat[g][:, g]
        frag_e = [self.elements[ind] for ind in g]
        N_atom = len(g)
        frag_geo = np.zeros([N_atom, 3])
        for count_i, i in enumerate(g):
            frag_geo[count_i, :] = self.geo[i, :]
        frag_bem = [self.bond_mats[0][i] for i in g]
        frag_formals = return_formals(frag_bem, frag_e)
        frag_q = int(sum(frag_formals))
        frag_atom_info = separate_atom_info_new(self._atom_info, g)
        mols.append(yarpecule((frag_adj, frag_geo, frag_e, frag_q, frag_atom_info), canon=canon))

    return mols


def enum_tuple_call_sites_old(yarpecule, adj_mat, y, c, np, vstack):
    """
    Old representative tuple constructions from yarp/reaction/enum.py.

    Difference:
    - enum recreates yarpecules from 4-tuples and drops parent metadata.
    """

    product_a = yarpecule((adj_mat, y.geo, y.elements, y.q), canon=False)
    product_b = yarpecule((adj_mat, vstack([c[0][1].geo, c[1][1].geo]), c[0][1].elements + c[1][1].elements, c[0][1].q + c[1][1].q), canon=False)
    product_c = yarpecule((np.where(adj_mat > 0, 1, 0).astype(int), y.geo, y.elements, y.q), canon=False)
    return product_a, product_b, product_c


def enum_tuple_call_sites_new(yarpecule, adj_mat, y, c, np, vstack):
    """
    New proposed tuple constructions for yarp/reaction/enum.py.

    Difference:
    - enum passes copied `_atom_info` into every tuple-created product.
    - intramolecular tuple calls preserve ordering by direct copy.
    - intermolecular tuple calls merge both parents into one local index space.
    """

    product_a = yarpecule((adj_mat, y.geo, y.elements, y.q, carry_atom_info_by_index_new(y._atom_info)), canon=False)
    product_b = yarpecule(
        (
            adj_mat,
            vstack([c[0][1].geo, c[1][1].geo]),
            c[0][1].elements + c[1][1].elements,
            c[0][1].q + c[1][1].q,
            join_atom_info_new([c[0][1], c[1][1]]),
        ),
        canon=False,
    )
    product_c = yarpecule((np.where(adj_mat > 0, 1, 0).astype(int), y.geo, y.elements, y.q, carry_atom_info_by_index_new(y._atom_info)), canon=False)
    return product_a, product_b, product_c


def test_input_parser_call_sites_old(mol_parse, ethanol_mol, acetate_mol, betaine_mol):
    """
    Old production-adjacent call sites from test/yarpecule/test_input_parser.py.

    Difference:
    - tests currently unpack four returns from mol_parse().
    """

    elements, geo, adj_mat, q = mol_parse(ethanol_mol)
    elements, geo, adj_mat, q = mol_parse(acetate_mol)
    elements, geo, adj_mat, q = mol_parse(betaine_mol)
    return elements, geo, adj_mat, q


def test_input_parser_call_sites_new(mol_parse, ethanol_mol, acetate_mol, betaine_mol):
    """
    New proposed call-site edits for test/yarpecule/test_input_parser.py.

    Difference:
    - tests now unpack the fifth atom_info return explicitly.
    - this is required if mol_parse() returns atom_info upward.
    """

    elements, geo, adj_mat, q, atom_info = mol_parse(ethanol_mol)
    elements, geo, adj_mat, q, atom_info = mol_parse(acetate_mol)
    elements, geo, adj_mat, q, atom_info = mol_parse(betaine_mol)
    return elements, geo, adj_mat, q, atom_info


def test_smiles_call_sites_old(smiles2adjmat, cases):
    """
    Old representative test-site assumptions from test/yarpecule/graph/test_smiles.py.

    Difference:
    - tests currently assume list-based atom_info rows like `label[0]`.
    """

    outputs = []
    for case in cases:
        adjmat, bemat, atom_info = smiles2adjmat(case)
        elements = [label[0] for label in atom_info]
        outputs.append((adjmat, bemat, atom_info, elements))
    return outputs


def test_smiles_call_sites_new(smiles2adjmat, cases):
    """
    New proposed test-site assumptions for test/yarpecule/graph/test_smiles.py.

    Difference:
    - tests now read dict-based atom_info with explicit field access.
    - this is the required test migration for the parser metadata shape change.
    """

    outputs = []
    for case in cases:
        adjmat, bemat, atom_info = smiles2adjmat(case)
        elements = [atom_info[i]["element"].capitalize() for i in atom_info]
        atom_maps = [atom_info[i]["atom_map"] for i in atom_info]
        outputs.append((adjmat, bemat, atom_info, elements, atom_maps))
    return outputs


def planned_test_inventory_new():
    """
    Draft test inventory for the production migration.

    Difference:
    - distinguishes tests that need assertion rewrites from tests that should be
      rerun as regression coverage because enum/get_smiles/separate behavior changes.
    - the implementation pass should point the user to these test sites rather
      than having the assistant rewrite the tests.
    """

    tests_to_update = [
        "test/yarpecule/graph/test_smiles.py",
        "test/yarpecule/test_input_parser.py",
        "test/yarpecule/test_yarpecule.py",
        "test/reaction/test_enum.py",
    ]

    tests_to_run = tests_to_update + [
        "test/yarpecule/test_hashes.py",
        "test/reaction/test_filters.py",
    ]

    return tests_to_update, tests_to_run


def parser_dev_notebook_call_sites_old(xyz_from_smiles, mol_parse, smiles_case, mol_file):
    """
    Old dev-notebook call sites.

    Difference:
    - notebook currently unpacks four returns from xyz_from_smiles() and mol_parse().
    """

    elements, geo, adj_mat, q = xyz_from_smiles(smiles_case, mode="yarp")
    elements_mol, geo_mol, adj_mol, q_mol = mol_parse(mol_file)
    return elements, geo, adj_mat, q, elements_mol, geo_mol, adj_mol, q_mol


def parser_dev_notebook_call_sites_new(xyz_from_smiles, mol_parse, smiles_case, mol_file):
    """
    New proposed dev-notebook call sites.

    Difference:
    - notebook now unpacks atom_info from both parser helpers.
    """

    elements, geo, adj_mat, q, atom_info = xyz_from_smiles(smiles_case, mode="yarp")
    elements_mol, geo_mol, adj_mol, q_mol, atom_info_mol = mol_parse(mol_file)
    return elements, geo, adj_mat, q, atom_info, elements_mol, geo_mol, adj_mol, q_mol, atom_info_mol

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


def yarp_package_warning_policy_new(RDLogger, ob):
    """
    Draft for the package-level warning policy now living in `yarp/__init__.py`.

    Why this logic change was made
    ------------------------------
    During the branch we accumulated warning silencing in several production
    modules:
    - `input_parsers.py` disabled RDKit warnings/info
    - `yarpecule.py` suppressed Open Babel warnings

    That scattered global side effects across import sites and made it harder to
    reason about why a given run or test emitted warnings. The branch now moves
    package-wide warning policy to one import point:
    - silence RDKit warning/info globally for YARP runtime
    - silence Open Babel warnings globally for YARP runtime

    This centralization does not replace function-local logger handling that is
    intentionally scoped to a single helper call.
    """

    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.info")
    ob.obErrorLog.SetOutputLevel(ob.obError)


def get_egat_barriers_new(yp_rxns, model, args, pd, os, predict_activation_energy, FastDataset):
    """
    Draft for the later `yarp/reaction/ml_barrier.py` branch updates.

    Why this logic change was made
    ------------------------------
    Two distinct fixes landed after the core parser migration:

    1. EGAT-boundary atom-map normalization
       YARP now preserves atom maps exactly, including fragment-local subsets
       produced by `separate()`. EGAT assumed dense positional map labels and
       broke on depth-2 / sparse-map reactions. The production fix normalizes
       mapped smiles only for EGAT, using one shared renumbering for reactant
       and product, then discards that temporary mapping after prediction.

    2. `tmp/` lifecycle cleanup
       EGAT diagnostics originally deleted `tmp/` too early, which hid the
       failing AAM strings. The branch changed this in two phases:
       - keep `tmp/` alive until dataset iteration finishes so `fail.txt` can be written
       - after debugging was complete, remove `tmp/` again on clean runs while
         preserving it when `fail.txt` or `exclude.txt` exists

    Net effect:
    - depth-2 and sparse-map EGAT runs now succeed
    - failed EGAT datapoints still leave debuggable traces when they exist
    - successful runs stop littering the working directory with `tmp/`
    """

    rxn_list = list(yp_rxns.values())
    dataframe = []
    for rxn in rxn_list:
        rsmiles, psmiles = normalize_reaction_maps_for_egat(
            rxn.reactant.map_smi,
            rxn.product.map_smi,
            start_at=1,
        ).reactant_smiles, normalize_reaction_maps_for_egat(
            rxn.reactant.map_smi,
            rxn.product.map_smi,
            start_at=1,
        ).product_smiles
        dataframe.append(f"{rsmiles}>>{psmiles}")

    dataframe = pd.DataFrame(dataframe, columns=["AAM"])
    os.makedirs("tmp", exist_ok=True)
    csv_path = os.path.join("tmp", "egat_barriers.csv")
    dataframe.to_csv(csv_path, index=False)
    test_dataset = FastDataset(args, dataset=csv_path)

    try:
        for data_idx in range(len(test_dataset)):
            datapoint = test_dataset[data_idx]
            if datapoint is None:
                dataframe.loc[data_idx, "egat_barrier"] = None
                continue
            dp_idx, rgraph, pgraph, strings = datapoint
            try:
                dataframe.loc[dp_idx, "egat_barrier"] = predict_activation_energy(model, rgraph, pgraph)
            except Exception:
                dataframe.loc[dp_idx, "egat_barrier"] = None
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)
        fail_path = os.path.join("tmp", "fail.txt")
        exclude_path = os.path.join("tmp", "exclude.txt")
        if os.path.isdir("tmp") and not os.path.exists(fail_path) and not os.path.exists(exclude_path):
            try:
                os.rmdir("tmp")
            except OSError:
                pass

    for rxn_idx, rxn in enumerate(rxn_list):
        rxn.barrier["egat"] = dataframe.loc[rxn_idx, "egat_barrier"]
    return yp_rxns


def update_yarp_pickles_branch_notes_new():
    """
    Draft notes for `helper/update_yarp_pickles.py` branch changes.

    Why this logic change was made
    ------------------------------
    Late in the branch it became clear that some old test fixtures should not be
    regenerated by rerunning chemistry at all. They already encoded the intended
    historical reaction sets and only needed a schema upgrade so that modern
    YARP objects would have `_atom_info`.

    The existing pickle-upgrade helper turned out to be the correct mechanism.
    The branch only needed to extend it so it could operate on the new schema:
    - add `_atom_info` to the yarpecule guardrail
    - keep rebuilding yarpecules through the live constructor so `_atom_info`
      is populated by current production code
    - preserve higher-level reaction metadata such as stored EGAT barriers
    - add the same lightweight Open Babel stub used elsewhere so helper runs are
      possible even when local environments lack Open Babel

    This is an important branch-level decision:
    - when the goal is “same old chemistry, new object schema”, prefer pickle
      migration over fresh enumeration
    """

    return {
        "yarpecule_guardrail_addition": "_atom_info",
        "rebuild_strategy": "constructor_rebuild_with_metadata_backfill",
        "local_openbabel_stub": True,
    }


def regenerate_test_pickles_branch_notes_new():
    """
    Draft notes for `dev/regenerate_test_pickles.py` and related test-fixture work.

    Why this logic change was made
    ------------------------------
    Some tests truly do depend on fresh chemistry outputs rather than migrated
    old pickles. During the branch the helper evolved to make those regenerations
    explicit and reproducible.

    Two key lessons were learned and are worth preserving in the plan:
    1. `haa_heavy_b2f2_d1.pkl` must be regenerated as the five-product
       unrestricted depth-1 fixture with EGAT barriers, not as the restricted
       two-product heavy-atom-only case. Otherwise the downstream depth-2 tests
       collapse to the same count and produce misleading “behavior drift”.
    2. `3hp_heavy_b2f2_d1.pkl` also needs EGAT enabled when regenerated so test
       fixtures remain internally consistent.
    """

    return {
        "haa_fixture_shape": "unrestricted_b2f2_depth1_with_egat",
        "3hp_fixture_shape": "depth1_with_egat",
        "depth2_proxy_helper": "dev/check_depth2_counts.py",
    }


if __name__ == "__main__":
    # Representative depth-2 failure that should normalize cleanly.
    example = normalize_reaction_maps_for_egat(
        "[O:0]=[C:1]([O:4][O:5][H:11])[H:6]",
        "[H:6][H:11].[O:0]=[C:1]1[O:4][O:5]1",
        start_at=1,
    )
    print(example)
