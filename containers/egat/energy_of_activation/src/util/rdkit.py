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
    3: rdchem.BondType.TRIPLE
}
# TODO: How to handle dative bonds for organometallics?

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
    N = len(elements)
    assert adj.shape == (N, N)
    assert bond_orders.shape == (N, N)

    # Make elements uppercase
    elements = [el.upper() for el in elements]

    # Count up hydrogens and unpaired electrons
    num_h = elements.count("H")
    num_rad = np.diag(bond_orders) % 2 == 1

    rw = Chem.RWMol()
    # add atoms
    for el in elements:
        a = Chem.Atom(el)
        rw.AddAtom(a)

    # add bonds (only i < j to avoid duplicates)
    for i in range(N):
        for j in range(i+1, N):
            if adj[i, j]:
                bo = bond_orders[i, j]
                # map to RDKit bond type
                btype = BOND_MAP.get(bo)
                if btype is None:
                    # allow integerish strings
                    try:
                        btype = BOND_MAP.get(float(bo))
                    except Exception:
                        raise ValueError(f"Unknown bond order value at {i},{j}: {bo}")
                rw.AddBond(i, j, btype)

    mol = rw.GetMol()  # immutable Mol copy
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            warnings.warn(f"Sanitization failed: {e}")

    return mol