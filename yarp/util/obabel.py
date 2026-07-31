import os
import numpy as np
from openbabel import pybel

from yarp.util.write_files import mol_write_yp

def obabel_ff_opt(molecule, lot="uff", maxiter=500):
    '''
    Perform low-level level geometry optimization on yarpecule using openbabel.

    Parameters
    ----------
    molecule : yarpecule object
        molecule to be optimized 

    lot : string
        Level of theory used for quick optimization

    maxiter : int
        Maximum number of optimization steps
        

    Returns
    -------
    opt_geom : nd array (N x 3)
        optimized geometry

    Notes
    -----
    Supported force fields (`lot`):
        - 'uff' : Universal Force Field, general-purpose, works for most elements
        - 'mmff94' : Merck Molecular Force Field, better for organics
        - 'ghemical' : simpler/faster, less accurate
    '''

    # Write yarpecule object to a temporary mol file
    mol_file = '.tmp.mol'
    mol_write_yp(mol_file, molecule.elements, molecule.geo,
                 molecule.bond_mats[0], molecule.adj_mat)

    # Use openbabel to perform geometry optimization
    mol = next(pybel.readfile("mol", mol_file))
    mol.localopt(forcefield=lot, steps=maxiter)

    # Delete temporary mol file
    os.system("rm {}".format(mol_file))

    # Collect optimized geometry coordinates
    opt_geom = np.zeros_like(molecule.geo)
    for count_i, i in enumerate(opt_geom):
        opt_geom[count_i] = mol.atoms[count_i].coords

    return opt_geom

def obabel_joint_opt(conformer, target_bem, target_adj, lot="uff", maxiter=500):
    '''
    Attempt to bias conformer geometry toward a target bond-electron matrix
    (BEM) using Open Babel.

    Parameters
    ----------
    conformer : conformer object
        conformer whose geometry is biased toward target_bem

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
    Mirrors quick_geom_opt's Open Babel fallback (obabel_ff_opt): writes a
    temporary mol file with the imposed target bonding and optimizes it via
    pybel's `localopt`, so both fallbacks behave identically instead of
    diverging on which part of the Open Babel API they happen to call.
    '''
    mol_file = '.tmp_joint.mol'
    try:
        mol_write_yp(mol_file, conformer.elements, conformer.geo, target_bem, target_adj)

        mol = next(pybel.readfile("mol", mol_file))
        mol.localopt(forcefield=lot, steps=maxiter)

        opt_geo = np.zeros_like(conformer.geo)
        for count_i, i in enumerate(opt_geo):
            opt_geo[count_i] = mol.atoms[count_i].coords

        return opt_geo
    except (ValueError, RuntimeError):
        return None
    finally:
        if os.path.exists(mol_file):
            os.remove(mol_file)
