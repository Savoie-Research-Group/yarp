"""
Helper functions for integration with RDKit
"""
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
from rdkit.Geometry import Point3D
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


def yarpecule_to_rdmol(elements, adj, bond_orders, atom_info=None, geo=None, sanitize=True):
    """
    Convert yarpecule graph data directly into an RDKit mol object.

    This is the single conversion path shared by SMILES generation and bond
    matrix drawing. Hydrogens are added explicitly (one atom per element entry)
    and implicit-H perception is disabled, so RDKit never invents "ghost"
    hydrogens that are absent from the yarpecule bond-electron matrix.

    Parameters:
    -----------
    elements : list
        List of element symbols (length N).

    adj : ndarray (N x N)
        Adjacency matrix with 1 for connected atoms and 0 for non-connected atoms.

    bond_orders : ndarray (N x N)
        Bond order / bond-electron matrix for the molecule. Off-diagonal entries
        are bond orders; diagonal entries encode lone/unpaired electrons.

    atom_info : dict, optional
        Optional atom metadata dictionary keyed by atom index. If present, the
        per-atom ``atom_map`` is attached as ``molAtomMapNumber``.

    geo : ndarray (N x 3), optional
        Optional 3D coordinates. When supplied, a conformer is attached and
        stereochemistry (atom chirality and double-bond E/Z) is perceived from
        the geometry, matching the behavior of the legacy MOL-file round trip.

    sanitize : bool, default=True
        Whether RDKit sanitization should be applied prior to return.

    Returns:
    --------
    mol : RDKit mol object
    """
    N = len(elements)
    if adj.shape != (N, N):
        raise ValueError(f"adj shape {adj.shape} does not match ({N}, {N})")
    if bond_orders.shape != (N, N):
        raise ValueError(f"bond_orders shape {bond_orders.shape} does not match ({N}, {N})")

    element_lower = [el.lower() for el in elements]
    elements = [el.upper() for el in element_lower]
    is_radical = np.diag(bond_orders) % 2 == 1

    # Use Lewis-derived formal charges so that charged atoms get the correct
    # explicit valence (imported lazily to avoid a circular import at module
    # load time).
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
        rw.AddAtom(atom)

    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j]:
                bo = int(round(bond_orders[i, j]))
                if bo == 0:
                    # dative / metal bonds are stored with zero order; draw as single
                    # ERM: there *is* an rdchem.BondType.DATIVE option,
                    # but I'm not sure if that's what we want.
                    # Should consult with Zhao down the road on this question.
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

    # Perceive stereochemistry from the 3D geometry when it is available.
    if geo is not None:
        geo = np.asarray(geo, dtype=float)
        conf = Chem.Conformer(N)
        for idx in range(N):
            conf.SetAtomPosition(idx, Point3D(float(geo[idx][0]), float(geo[idx][1]), float(geo[idx][2])))
        conf.Set3D(True)
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)
        Chem.AssignStereochemistryFrom3D(mol)
        # AssignStereochemistryFrom3D tags every center with a definable
        # handedness, including symmetric (non-stereogenic) ones. Re-run the
        # CIP assignment with cleanIt=True so spurious tags on atoms/bonds that
        # are not real stereocenters are stripped, matching the legacy MOL-file path.
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    # Attach atom-map labels AFTER stereo perception. Map numbers make otherwise
    # symmetric substituents (e.g. the three H of a methyl group) distinguishable,
    # which would make RDKit treat non-stereogenic centers as stereocenters.
    if atom_info is not None:
        for idx, atom in enumerate(mol.GetAtoms()):
            if idx in atom_info and atom_info[idx].get("atom_map") is not None:
                atom.SetProp("molAtomMapNumber", str(atom_info[idx]["atom_map"]))

    return mol

def geom_from_rdmol(mol, conf_index=0):
    """
    Extract 3D conformer data from RDKit mol object

    Parameters:
    -----------
    mol : RDKit mol object
        The molecule to convert.

    Returns:
    --------
    geo : ndarray (N x 3) or None
        3D coordinates if a conformer is present, otherwise None.
    """
    geo = None
    N = mol.GetNumAtoms()
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer(conf_index)
        geo = np.zeros((N, 3), dtype=float)
        for idx in range(N):
            pos = conf.GetAtomPosition(idx)
            geo[idx] = [pos.x, pos.y, pos.z]

    return geo

def rdkit_ff_opt(ypcule, lot='uff', maxiter=200):
    '''
    Perform low-level level geometry optimization of yarpecule geometry
    via RDKit mol object.

    Parameters:
    ----------
    ypcule : yarpecule object
        molecule to be optimized

    lot : string
        Level of theory used for quick optimization
        ERM: mmff94 has a tendency to reform the reactant geometry
        when used to generate initial geom of products post product enumeration

    Returns:
    --------
    opt_geom : nd array (N x 3)
        optimized geometry
    '''

    rdmol = yarpecule_to_rdmol(elements=ypcule.elements, adj=ypcule.adj_mat, bond_orders=ypcule.bond_mats[0],
                               atom_info=ypcule._atom_info, geo=ypcule.geo)

    if lot == "uff":
        opt = AllChem.UFFOptimizeMolecule(rdmol, maxIters=maxiter, ignoreInterfragInteractions=False)
    elif lot == "mmff94":
        opt = AllChem.MMFFOptimizeMolecule(rdmol, maxIters=maxiter, ignoreInterfragInteractions=False)

    opt_geom = geom_from_rdmol(rdmol)

    return opt_geom