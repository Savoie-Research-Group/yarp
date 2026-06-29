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
