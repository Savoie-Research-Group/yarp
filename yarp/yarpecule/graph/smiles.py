"""
Helper functions for converting SMILES strings into molecular graphs
"""
import re
import numpy as np
from yarp.yarpecule.graph.adjacency import adjmat_to_adjlist
from yarp.yarpecule.graph.fragment import return_rings
from yarp.util.properties import el_valence, el_metals, el_expand_octet, el_mass, el_to_an
from yarp.yarpecule.atom_mapping import canon_order


def smiles2adjmat(smiles, verbose=False, reorder_mapped=True):
    """
    In-house Savoie group SMILES parser. Written in python and transparent to debug. The main motivation
    was to consistently handle protonation of radicals and atoms with formal charges. The usual SMILES
    syntax rules apply, except that square brace annotations are handled specially. Square braces are
    reserved to annotate the isotope, formal charge, or number of hydrogens that should be added to an
    atom. The isotope number must preceed the element label. The charge and number of hydrogens must be
    after the element label. The formal charge can be specified as +d, ++++, -d, ---, where d is an integer.
    The number of hydrogens to be added can be specified as Hd or HHH, where d is an integer.

    Parameters
    ----------
    smiles: str
            The smiles string that the user wants to parse.

    Returns
    -------
    adjmat: array
            This is numpy array holding the graph defined by the smiles string.

    atom_info: dict
            This dict is indexed to the adjacency matrix and contains metadata for each atom.
    """

    if not hasattr(smiles2adjmat, 'aromatics'):
        smiles2adjmat.aromatics = {"b", "c", "n", "o", "p", "s"}
        smiles2adjmat.token_pattern = r'(\[[^\]]*\]|%\d{2}|[A-Z](?:[a-z]+)?|[a-z]|\d{1}|[=#+\-\\\/.@:()])'
        smiles2adjmat.atom_pattern = r'([A-Z](?:[a-z]+)?|[a-z])'
        smiles2adjmat.ring_pattern = re.compile(r'^(%\d{2}|\d)$')
        smiles2adjmat.isotope_pattern = re.compile(r'^\[(\d+)')
        smiles2adjmat.charge_pattern = re.compile(r'(\+\d+|-\d+|\++(?!\d)|-+(?!\d))')
        smiles2adjmat.hydrogen_pattern = re.compile(r'(H\d+|H+)')
        smiles2adjmat.element_label_pattern = re.compile(r'([A-Z](?:[a-z]+)?|[a-z])')
        smiles2adjmat.mapping_pattern = re.compile(r':(\d+)')
        smiles2adjmat.valid_elements = {
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Hf', 'Ta', 'W',
            'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
            'At', 'Rn', 'b', 'c', 'n', 'o', 'p', 's'
        }

    preliminary_tokens = re.findall(smiles2adjmat.token_pattern, smiles)
    final_tokens = []
    for token in preliminary_tokens:
        if len(token) == 2 and token not in smiles2adjmat.valid_elements and not token.startswith("%"):
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

            isotope_match = smiles2adjmat.isotope_pattern.search(token)
            if isotope_match:
                isotope = int(isotope_match.group(1))

            charge_match = smiles2adjmat.charge_pattern.search(token)
            if charge_match:
                charge_str = charge_match.group(1)
                if charge_str[-1].isdigit():
                    formal_charge = int(charge_str)
                else:
                    formal_charge = charge_str.count('+') - charge_str.count('-')

            hydrogen_match = smiles2adjmat.hydrogen_pattern.search(token)
            if verbose:
                print(f"{token=} {hydrogen_match=}")
            if hydrogen_match:
                h_str = hydrogen_match.group(1)
                element_label = smiles2adjmat.element_label_pattern.search(token).group(1)
                if element_label.lower() != 'h':
                    if h_str[-1].isdigit():
                        explicit_hydrogens = int(h_str[1:])
                    else:
                        explicit_hydrogens = len(h_str)

            mapping_match = smiles2adjmat.mapping_pattern.search(token)
            if mapping_match:
                atom_mapping = int(mapping_match.group(1))

            element_label_match = smiles2adjmat.element_label_pattern.search(token)
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

        elif re.match(smiles2adjmat.atom_pattern, token):
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
        elif re.match(smiles2adjmat.atom_pattern, token):
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
        elif smiles2adjmat.ring_pattern.match(token):
            atom_index = sequential_atom_counter - 1
            if token in ring_numbers:
                if tokens[i - 1] == '-':
                    last_bond_type = 1
                elif tokens[i - 1] == '=':
                    last_bond_type = 2
                elif tokens[i - 1] == '#':
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
                if tokens[i - 1] == '-':
                    last_bond_type = 1
                elif tokens[i - 1] == '=':
                    last_bond_type = 2
                elif tokens[i - 1] == '#':
                    last_bond_type = 3
                else:
                    last_bond_type = None
                ring_numbers[token] = (atom_index, last_bond_type, pending_bond_stereo)
                pending_bond_stereo = None

    for i in range(len(atom_indices) - 1):
        current_atom_index = atom_indices[i]
        next_atom_index = atom_indices[i + 1]
        intervening_tokens = tokens[current_atom_index + 1: next_atom_index]

        if not any(re.match(r'[\[\].()]', token) for token in intervening_tokens):
            after_digits = []
            for token in intervening_tokens[::-1]:
                if smiles2adjmat.ring_pattern.match(token):
                    break
                after_digits.append(token)

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
        for j in range(i - 1, -1, -1):
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
        for j in range(i - 1, -1, -1):
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

    ambiguous_aromatic_assignment = False
    if any(atom_info[i]["aromatic_input"] for i in atom_info):
        aromatic_rings = []
        try:
            aromatic_atoms = []
            for idx, info in atom_info.items():
                if info["aromatic_input"] and info["element"] in smiles2adjmat.aromatics:
                    aromatic_atoms.append(idx)
            aromatic_atoms = set(aromatic_atoms)

            for ring in return_rings(adjmat_to_adjlist(adjmat), max_size=10, remove_fused=False):
                if all(idx in aromatic_atoms for idx in ring):
                    aromatic_rings.append(ring)

            fused_components = []
            for ring in aromatic_rings:
                ring = set(ring)
                merged = True
                while merged:
                    merged = False
                    next_components = []
                    for comp in fused_components:
                        if comp & ring:
                            ring |= comp
                            merged = True
                        else:
                            next_components.append(comp)
                    fused_components = next_components
                fused_components.append(ring)

            for component in fused_components:
                component = sorted(component)
                component_set = set(component)
                component_edges = []
                for count_i, i in enumerate(component):
                    for j in component[count_i + 1:]:
                        if adjmat[i, j] == 1 and can_promote_aromatic_edge(i, j, component_set, adjmat):
                            component_edges.append((i, j))

                target_size = len(component) // 2
                promoted_edges, best_partial = choose_aromatic_matching(component_edges, target_size)
                if promoted_edges is None:
                    ambiguous_aromatic_assignment = True
                    promoted_edges = best_partial

                for i, j in promoted_edges:
                    adjmat[i, j] = 2
                    adjmat[j, i] = 2

        except Exception as err:
            print(f"WARNING: kekulization/aromatic assignment fallback used: {err}")
            for ring in return_rings(adjmat_to_adjlist(adjmat), max_size=10, remove_fused=True):
                if not all(atom_info[idx]["element"] in smiles2adjmat.aromatics for idx in ring):
                    continue
                if any(sum(idx in other for other in aromatic_rings) > 1 for idx in ring):
                    continue
                for ind in range(1, len(ring)):
                    if (ind - 1) % 2 == 0:
                        adjmat[ring[ind], ring[ind - 1]] = 2
                        adjmat[ring[ind - 1], ring[ind]] = 2

    adjmat, atom_info = add_hydrogens(adjmat, atom_info, atom_parse_meta)

    for idx in atom_parse_meta:
        if idx in atom_info:
            isotope = atom_parse_meta[idx]["isotope"]
            if atom_info[idx]["mass"] is None:
                atom_info[idx]["mass"] = float(isotope) if isotope is not None else el_mass[atom_info[idx]["element"]]

    provided_maps = [atom_info[i]["atom_map"] for i in atom_info if atom_info[i]["atom_map"] is not None]
    if len(provided_maps) != len(set(provided_maps)):
        dupes = sorted({m for m in provided_maps if provided_maps.count(m) > 1})
        raise ValueError(f"Duplicate atom-map indices in input SMILES: {dupes}")

    if reorder_mapped:
        adjmat, atom_info = reorder_by_mappings(adjmat, atom_info)

    bond_electron_mat = adjmat.copy()
    for i in atom_info:
        bond_electron_mat[i, i] = el_valence[atom_info[i]["element"]] - atom_info[i]["formal_charge"] - sum(adjmat[i])

    heavy_radicals = [
        i for i in atom_info
        if atom_info[i]["element"] != "h" and int(bond_electron_mat[i, i]) % 2 == 1
    ]
    if ambiguous_aromatic_assignment or (any(atom_info[i]["aromatic_input"] for i in atom_info) and heavy_radicals):
        interpreted_smiles = bemat_to_smiles(bond_electron_mat, atom_info)
        if heavy_radicals:
            print(f"WARNING: ambiguous SMILES provided; interpreted as radical structure: {interpreted_smiles}")
        else:
            print(f"WARNING: ambiguous SMILES provided; interpreted as: {interpreted_smiles}")

    return np.where(adjmat > 0, 1, 0), bond_electron_mat, atom_info


def choose_aromatic_matching(component_edges, target_size):
    """
    Return the first valid deterministic matching of `target_size` edges and
    also track the best partial matching if a full one is unavailable.
    """
    best_partial = []
    stack = [(0, set(), [])]

    while stack:
        edge_index, used_atoms, chosen_edges = stack.pop()

        if len(chosen_edges) > len(best_partial):
            best_partial = chosen_edges.copy()
        if len(chosen_edges) == target_size:
            return chosen_edges.copy(), best_partial
        if edge_index >= len(component_edges):
            continue

        i, j = component_edges[edge_index]
        stack.append((edge_index + 1, used_atoms.copy(), chosen_edges.copy()))
        if i not in used_atoms and j not in used_atoms:
            next_used = used_atoms.copy()
            next_used.update((i, j))
            next_edges = chosen_edges.copy()
            next_edges.append((i, j))
            stack.append((edge_index + 1, next_used, next_edges))

    return None, best_partial


def bemat_to_smiles(bond_electron_mat, atom_info):
    """
    Render the current interpreted graph as a kekulized SMILES string for
    diagnostics.
    """
    from rdkit import Chem

    bond_to_type = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.QUADRUPLE,
    }

    mol = Chem.RWMol()
    for idx in range(len(atom_info)):
        info = atom_info[idx]
        atom = Chem.Atom(el_to_an[info["element"]])
        atom.SetFormalCharge(int(info["formal_charge"]))
        if info.get("mass", None) not in (None, el_mass[info["element"]]):
            atom.SetIsotope(int(round(info["mass"])))
        atom.SetNumRadicalElectrons(int(bond_electron_mat[idx, idx] % 2))
        mol.AddAtom(atom)

    for i in range(len(atom_info)):
        for j in range(i + 1, len(atom_info)):
            bond_order = int(bond_electron_mat[i, j])
            if bond_order > 0:
                mol.AddBond(i, j, bond_to_type.get(bond_order, Chem.BondType.SINGLE))

    mol = mol.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(
        mol,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        catchErrors=True,
    )
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, kekuleSmiles=True)
    except Exception:
        return Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=True)


def can_promote_aromatic_edge(i, j, component_atoms, adjmat):
    """
    Guard aromatic promotion against ring atoms that already carry an exocyclic
    multiple bond outside the aromatic component.

    This prevents cases like `Oc1ncc(c(=O)[nH]1)C` from promoting an additional
    double bond onto the carbonyl-bearing ring carbon.
    """
    for atom in (i, j):
        for neighbor, bond_order in enumerate(adjmat[atom]):
            if neighbor not in component_atoms and bond_order > 1:
                return False
    return True


def add_hydrogens(adjmat, atom_info, atom_parse_meta):
    """
    Add hydrogens to atoms based on either the explicit number of hydrogens designation or
    an inference algorithm based on formal charge and number of bonds.
    """
    hydrogens_to_add = []
    bonded_hydrogens = []

    for atom in range(len(atom_info)):
        h_count = sum(1 for i in range(len(atom_info)) if atom_info[i]["element"] == 'h' and adjmat[atom, i] != 0)
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
        if element in el_metals:
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
                    e = desired_electrons - 2 * needed_hydrogens - 2 * bonds
                    if (formal_charge - int(e / 2)) <= 0:
                        needed_hydrogens += formal_charge
                    else:
                        needed_hydrogens -= formal_charge
                elif formal_charge < 0:
                    e = desired_electrons - 2 * needed_hydrogens - 2 * bonds
                    if (needed_hydrogens + formal_charge) >= 0:
                        needed_hydrogens += formal_charge
                    else:
                        needed_hydrogens -= formal_charge
        else:
            needed_hydrogens = 0

        if needed_hydrogens < 0:
            print("Warning: add_hydrogens() was unable to satisfy formal charge specification with hydrogens.")
        if (2 * bonds + 2 * needed_hydrogens) > 8 and not el_expand_octet[element]:
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


class OctetError(ValueError):
    """Exception raised when the number of electrons exceeds the allowed limit."""

    def __init__(self, atom_indices, message="Atom indices {} has an octet violation."):
        self.message = message.format(atom_indices)
        super().__init__(self.message)


def reorder_by_mappings(adjmat, atom_info):
    """
    Reorder atoms in the graph based on their atom mappings if present.
    If no mappings are present, returns the original ordering.
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
