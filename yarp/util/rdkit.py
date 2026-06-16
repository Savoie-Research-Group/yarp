"""
Helper functions for integration with RDKit
"""
from rdkit import Chem
from rdkit.Chem import rdchem
import numpy as np
import warnings

BOND_MAP = {
    1: rdchem.BondType.SINGLE,
    2: rdchem.BondType.DOUBLE,
    3: rdchem.BondType.TRIPLE,
    4: rdchem.BondType.QUADRUPLE,
    5: rdchem.BondType.QUINTUPLE,
    6: rdchem.BondType.HEXTUPLE,
}
# TODO: How to handle dative bonds for organometallics?


def yarpecule_to_rdmol(elements, adj, bond_orders, atom_info=None, sanitize=True):
    """
    Convert yarpecule graph data into an RDKit mol object.

    Parameters:
    -----------
    elements : list
        List of element symbols (length N).

    adj : ndarray (N x N)
        Adjacency matrix with 1 for connected atoms and 0 for non-connected atoms.

    bond_orders : ndarray (N x N)
        Bond order / bond-electron matrix for the molecule.

    atom_info : dict, optional
        Optional atom metadata dictionary. If present and atom maps are available,
        map labels are attached to output atoms.

    sanitize : bool, default=True
        Whether RDKit sanitization should be applied prior to return.
    """
    N = len(elements)
    if adj.shape != (N, N):
        raise ValueError(f"adj shape {adj.shape} does not match ({N}, {N})")
    if bond_orders.shape != (N, N):
        raise ValueError(f"bond_orders shape {bond_orders.shape} does not match ({N}, {N})")

    element_lower = [el.lower() for el in elements]
    elements = [el.upper() for el in element_lower]
    formal_charges = np.zeros(N, dtype=int)
    is_radical = np.diag(bond_orders) % 2 == 1

    # Use Lewis-derived formal charges when available (imported lazily to avoid
    # importing yarpecule internals at module import time).
    try:
        from yarp.yarpecule.lewis.bem_score import return_formals
        formal_charges = np.array(return_formals(bond_orders, element_lower), dtype=int)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        warnings.warn(
            "Unable to infer formal charges from bond matrix; defaulting to zeros. "
            f"Error: {type(exc).__name__}: {exc}"
        )
        formal_charges = np.zeros(N, dtype=int)

    rw = Chem.RWMol()
    for idx, el in enumerate(elements):
        atom = Chem.Atom(el)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
        atom.SetFormalCharge(int(formal_charges[idx]))
        atom.SetNumRadicalElectrons(int(is_radical[idx]))
        if atom_info is not None and idx in atom_info:
            record = atom_info[idx]
            if record.get("atom_map") is not None:
                atom.SetProp("molAtomMapNumber", str(record["atom_map"]))
            stereo = record.get("stereo", {}).get("atom")
            if stereo == "@":
                atom.SetChiralTag(rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
            elif stereo == "@@":
                atom.SetChiralTag(rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
        rw.AddAtom(atom)

    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j]:
                bo = int(round(bond_orders[i, j]))
                if bo == 0:
                    warnings.warn(
                        f"Connected atom pair ({i}, {j}) has zero bond order; falling back to single bond."
                    )
                    bo = 1
                btype = BOND_MAP.get(bo)
                if btype is None:
                    raise ValueError(f"Unknown bond order value at {i},{j}: {bond_orders[i, j]}")
                rw.AddBond(i, j, btype)

    mol = rw.GetMol()
    for atom in mol.GetAtoms():
        atom.UpdatePropertyCache(strict=False)
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except (rdchem.KekulizeException, rdchem.AtomValenceException, ValueError, RuntimeError) as e:
            warnings.warn(
                f"Sanitization failed for elements {''.join(elements)}: {type(e).__name__}: {e}"
            )

    return mol


def freeze_implicit_hydrogens(mol, elements, adj, only_charged=True):
    """
    Prevent RDKit from introducing implicit hydrogens not present in yarpecule.

    Parameters:
    -----------
    mol : RDKit Mol
        Molecule to update in place.
    elements : list
        Yarpecule element labels in atom-index order.
    adj : ndarray
        Yarpecule adjacency matrix.
    only_charged : bool, default=True
        If True, only charged atoms are modified.
    """
    h_neighbors = {
        i for i in range(len(elements))
        if any(adj[i, j] and elements[j].lower() == "h" for j in range(len(elements)))
    }

    for idx, atom in enumerate(mol.GetAtoms()):
        if idx >= len(elements):
            continue
        if elements[idx].lower() == "h":
            continue
        if idx in h_neighbors:
            continue
        if only_charged and atom.GetFormalCharge() == 0:
            continue
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
        atom.UpdatePropertyCache(strict=False)

    return mol

def graph_to_rdmol(elements, adj, bond_orders, sanitize=True):
    """
    Convert graph output (from yarpecule) into an RDKit mol object.
    Currently, will not handle radicals correctly!!!

    Parameters:
    -----------
    elements : list
        List of element symbols (length N). Will be converted into upper case for
        use in RDKit
    
    adj : ndarray (N x N)
        Adjacency matrix with 1 for connected atoms and 0 for non-connected atoms
    
    bond_orders : ndarray (N x N)
        Bond order matrix. Currently supports single, double, and triple bonds.
        Dative bonds not handled.
    
    sanitize : bool (default = True)
        If true, RDKit will sanitize the mol object before returning it.

    Returns:
    --------
    mol : RDKit mol object
    """
    return yarpecule_to_rdmol(elements, adj, bond_orders, sanitize=sanitize)