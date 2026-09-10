import os
import numpy as np
from openbabel import pybel

from yarp.util.write_files import mol_write_yp

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
