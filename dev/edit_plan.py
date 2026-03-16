"""
Development-only edit plan.

Each block below uses real Python definitions instead of quoted snippets:
- `*_old` is copied from the current codebase or reduced only by renaming.
- `*_new` is the proposed replacement body.

These are not meant to be imported by production code.
They are here so we can review exact edits side by side in code form.
"""

import os
import re
import numpy as np


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

    raise NotImplementedError("Use the current production body from smiles.py. This block marks the current top-level parser slot.")


def smiles2adjmat_new(smiles, verbose=False):
    """
    New top-level parser shape proposed for yarp/yarpecule/graph/smiles.py.

    Difference:
    - keeps the same function name and same `(adjmat, bond_electron_mat, atom_info)` return shape.
    - atom_info becomes dict-based.
    - stores `atom_index` separately from `atom_map`.
    - does kekulize-first aromatic handling, then warning-backed fallback.
    - keeps RDKit fallback out of this function.

    This function body stays minimal by reusing the existing parser flow and
    delegating the concrete structural changes to the edited helper blocks
    below: `add_hydrogens_new()` and `reorder_by_mappings_new()`.
    """

    if not hasattr(smiles2adjmat_new, 'aromatics'):
        smiles2adjmat_new.aromatics = {"b", "c", "n", "o", "p", "s"}
        smiles2adjmat_new.token_pattern = r'(\[[^\]]*\]|[A-Z](?:[a-z]+)?|[a-z]|\d{1}|[=#+\-\\\/.()])'
        smiles2adjmat_new.atom_pattern = r'([A-Z](?:[a-z]+)?|[a-z])'
        smiles2adjmat_new.isotope_pattern = re.compile(r'^\[(\d+)')
        smiles2adjmat_new.charge_pattern = re.compile(r'(\+\d+|-\d+|\++(?!\d)|-+(?!\d))')
        smiles2adjmat_new.hydrogen_pattern = re.compile(r'(H\d+|H+)')
        smiles2adjmat_new.element_label_pattern = re.compile(r'([A-Z](?:[a-z]+)?|[a-z])')
        smiles2adjmat_new.mapping_pattern = re.compile(r':(\d+)')
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

    raise NotImplementedError("The concrete edits are the helper-body changes below plus dict-based atom_info construction in the existing parser loop.")


def constants_block_old():
    """
    Old code block from yarp/util/constants.py.

    Difference:
    - there is no parser-facing atom-mass dictionary yet.
    """

    class Constants:
        n_a = 6.022140857E23
        ha_to_kcalmol = ha2kcalmol = 627.509
        ha_to_kJmol = ha2kJmol = 2625.50
        ha_to_J = ha_to_kJmol * 1000 / n_a
        J_to_ha = 1.0 / ha_to_J
        eV_to_ha = eV2ha = 0.0367493
        ha_to_eV = ha2eV = 1.0 / eV_to_ha
        kcal_to_kJ = kcal2kJ = 4.184
        rad_to_deg = 57.29577951308232087679815
        a0_to_ang = a02ang = 0.529177
        ang_to_a0 = ang2a0 = 1.0 / a0_to_ang
        ang_to_nm = 0.1
        ang_to_pm = 100
        ang_to_m = 1E-10
        a0_to_m = a0_to_ang * ang_to_m
        per_cm_to_hz = c_in_cm = 299792458 * 100
        amu_to_kg = 1.66053906660E-27
        amu_to_me = 1822.888486209
        atm_to_pa = 101325
        dm_to_m = 0.1

    return Constants


def constants_block_new():
    """
    New code block for yarp/util/constants.py.

    Difference:
    - adds a single module-level ATOM_MASS dictionary.
    - no existing constant names are changed.
    """

    class Constants:
        n_a = 6.022140857E23
        ha_to_kcalmol = ha2kcalmol = 627.509
        ha_to_kJmol = ha2kJmol = 2625.50
        ha_to_J = ha_to_kJmol * 1000 / n_a
        J_to_ha = 1.0 / ha_to_J
        eV_to_ha = eV2ha = 0.0367493
        ha_to_eV = ha2eV = 1.0 / eV_to_ha
        kcal_to_kJ = kcal2kJ = 4.184
        rad_to_deg = 57.29577951308232087679815
        a0_to_ang = a02ang = 0.529177
        ang_to_a0 = ang2a0 = 1.0 / a0_to_ang
        ang_to_nm = 0.1
        ang_to_pm = 100
        ang_to_m = 1E-10
        a0_to_m = a0_to_ang * ang_to_m
        per_cm_to_hz = c_in_cm = 299792458 * 100
        amu_to_kg = 1.66053906660E-27
        amu_to_me = 1822.888486209
        atm_to_pa = 101325
        dm_to_m = 0.1

    ATOM_MASS = {
        "h": 1.008,
        "he": 4.002602,
        "li": 6.94,
        "be": 9.0121831,
        "b": 10.81,
        "c": 12.011,
        "n": 14.007,
        "o": 15.999,
        "f": 18.998403163,
        "ne": 20.1797,
        "na": 22.98976928,
        "mg": 24.305,
        "al": 26.9815385,
        "si": 28.085,
        "p": 30.973761998,
        "s": 32.06,
        "cl": 35.45,
        "ar": 39.948,
        "k": 39.0983,
        "ca": 40.078,
        "br": 79.904,
        "i": 126.90447,
    }

    return Constants, ATOM_MASS


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


def add_hydrogens_new(adjmat, atom_info, smiles2adjmat, el_valence, el_metals, el_expand_octet, OctetError, ATOM_MASS):
    """
    New proposed code for yarp/yarpecule/graph/smiles.py.

    Difference:
    - atom_info is now a dict keyed by atom index.
    - new H records are dicts.
    - return shape is unchanged.
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
        valence_electrons = el_valence.get(element, None)

        if valence_electrons is None:
            print(f"Warning: Element '{element}' is not recognized or has an undefined valence.")
            hydrogens_to_add.append(0)
            continue
        elif element in el_metals:
            hydrogens_to_add.append(0)
            continue

        bonds = sum(adjmat[atom])

        if element in smiles2adjmat.aromatics:
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
                "mass": ATOM_MASS["h"],
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


def mol_parse_new(mol, rdmolfiles, np, ATOM_MASS):
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
        mass = float(isotope) if isotope else ATOM_MASS[atom.GetSymbol().lower()]

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


def xyz_from_smiles_new(smiles, mode, smiles2adjmat, BondType, MolFromSmiles, rdchem, Atom, el_to_an, el_n_expand_octet, el_expand_octet, OctetError, AddHs, AllChem, np, ATOM_MASS):
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
        mass = float(isotope) if isotope else ATOM_MASS[atom.GetSymbol().lower()]
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


def yarpecule_read_structure_new(self, mol, mode, strict, np, xyz_parse, table_generator, xyz_q_parse, mol_parse, xyz_from_smiles, ATOM_MASS):
    """
    New proposed code for yarp/yarpecule/yarpecule.py.

    Difference:
    - adds call-site `strict=False` fallback behavior.
    - every yarpecule gets full _atom_info.
    - xyz path creates atom_info after adjacency generation.
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
        self._atom_info = {
            i: {
                "atom_index": i,
                "atom_map": None,
                "element": self._elements[i].lower(),
                "formal_charge": 0,
                "mass": ATOM_MASS[self._elements[i].lower()],
                "stereo": {"atom": None, "bonds": {}},
                "aromatic_input": False,
            }
            for i in range(len(self._elements))
        }
    elif len(mol) > 4 and mol[-4:] == ".xyz":
        self._elements, self._geo = xyz_parse(mol)
        self._adj_mat = table_generator(self._elements, self._geo)
        self._q = xyz_q_parse(mol)
        self._atom_info = {
            i: {
                "atom_index": i,
                "atom_map": None,
                "element": self._elements[i].lower(),
                "formal_charge": 0,
                "mass": ATOM_MASS[self._elements[i].lower()],
                "stereo": {"atom": None, "bonds": {}},
                "aromatic_input": False,
            }
            for i in range(len(self._elements))
        }
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
    self._masses = np.array([ATOM_MASS[_] for _ in self._elements])


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
    - only missing maps are generated after canonical reorder.
    """

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

            used = {self._atom_info[i]["atom_map"] for i in self._atom_info if self._atom_info[i]["atom_map"] is not None}
            next_map = 0
            for i in self._atom_info:
                if self._atom_info[i]["atom_map"] is None:
                    while next_map in used:
                        next_map += 1
                    self._atom_info[i]["atom_map"] = next_map
                    used.add(next_map)
                    next_map += 1
    else:
        self._atom_hashes = np.array([atom_hash(_, self._adj_mat, self._masses) for _ in range(len(self._elements))])


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


def xyz_write_new(name, elements, geo, comment=None):
    """
    New proposed code for yarp/util/write_files.py.

    Difference:
    - adds optional comment line content.
    - comment should hold the RDKit canonical smiles.
    """

    out = open(name, 'w+')
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
    - mapped smiles uses _atom_info atom_map values.
    - verbose prints full RDKit mol dumps and atom_index -> atom_map table.
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
