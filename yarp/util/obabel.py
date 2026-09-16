import os
import numpy as np
from openbabel import pybel, openbabel as ob

from yarp.util.write_files import mol_write_yp

# A 3-atom molecule used only to invalidate the force field's cached setup.
# Built from an XYZ string rather than make3D() so that constructing it does
# not itself touch the force field machinery.
_SENTINEL_XYZ = """3

O   0.00000   0.00000   0.00000
H   0.95700   0.00000   0.00000
H  -0.23980   0.92700   0.00000
"""
_sentinel = None


def _ensure_setup_needed(mol, lot):
    """
    Guarantee Open Babel re-parameterizes for `mol` on the next Setup() call.

    pybel caches one OBForceField per force field at module level and
    `Molecule.localopt` reuses it. `OBForceField::Setup` skips
    re-parameterization when `IsSetupNeeded()` is False, keeping the cached
    molecule's atom types, charges and calculation list and merely copying in
    the new coordinates. Per the Open Babel header, that check compares only
    three things against the force field's cached molecule: atom count, bond
    count, and atomic numbers.

    Products enumerated from a common parent share the atom count and the
    element order exactly, and only take a handful of distinct bond counts
    (break-n/form-n can raise a bond order instead of adding an edge). On the
    37 KHP cycle-1 products, 20 of 36 consecutive pairs collide on all three,
    so the second product of the pair is optimized with the first one's
    parameters -- worth 2.33 A on product 24.

    There is no way to force `IsSetupNeeded()` to True: it is a read-only query
    over the force field's private cached molecule, and nothing in the public
    API clears it. The only lever is to set up a molecule that differs in one
    of the three compared properties. Creating a fresh force field instead is
    not an option -- `MakeNewInstance()` followed by `Setup()` segfaults.

    Returns True if `mol` is now guaranteed to trigger a full setup.
    """
    global _sentinel

    ff = ob.OBForceField.FindForceField(lot.lower())
    if ff is None:
        return False

    if ff.IsSetupNeeded(mol.OBMol):
        return True

    if _sentinel is None:
        _sentinel = pybel.readstring("xyz", _SENTINEL_XYZ)

    ff.Setup(_sentinel.OBMol)
    return ff.IsSetupNeeded(mol.OBMol)

def obabel_joint_opt(source, target_bem, target_adj, lot="uff", maxiter=500):
    '''
    Attempt to bias a geometry toward a target bond-electron matrix (BEM)
    using Open Babel.

    Passing a molecule's own BEM and adjacency matrix performs a plain
    force-field relaxation under its existing bonding, which is what the
    former `obabel_ff_opt` did.

    Parameters
    ----------
    source : conformer or yarpecule object
        Supplies the starting coordinates. Only `.elements` and `.geo` are
        read, so any object carrying those works.

    target_bem : nd array (N x N)
        target bond-electron matrix to bias the geometry toward

    target_adj : nd array (N x N)
        adjacency matrix derived from target_bem

    lot : string
        Force field used for optimization

    maxiter : int
        Maximum number of optimization steps

    Returns
    -------
    opt_geom : nd array (N x 3) or None
        optimized geometry, or None if Open Babel could not set up/optimize a
        force field for the imposed target bonding

    Notes
    -----
    Supported force fields (`lot`):
        - 'uff' : Universal Force Field, general-purpose, works for most elements
        - 'mmff94' : Merck Molecular Force Field, better for organics
        - 'ghemical' : simpler/faster, less accurate

    Writes a temporary mol file with the imposed target bonding and optimizes
    it via pybel's `localopt`.
    '''
    mol_file = '.tmp_joint.mol'
    try:
        mol_write_yp(mol_file, source.elements, source.geo, target_bem, target_adj)

        mol = next(pybel.readfile("mol", mol_file))

        # The cached force field may still be holding a molecule this one is
        # indistinguishable from, in which case localopt would optimize it with
        # the previous molecule's parameters.
        _ensure_setup_needed(mol, lot)

        mol.localopt(forcefield=lot, steps=maxiter)

        opt_geo = np.zeros_like(source.geo)
        for count_i, i in enumerate(opt_geo):
            opt_geo[count_i] = mol.atoms[count_i].coords

        return opt_geo
    except (ValueError, RuntimeError):
        return None
    finally:
        if os.path.exists(mol_file):
            os.remove(mol_file)
