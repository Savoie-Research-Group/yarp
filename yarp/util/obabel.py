import subprocess
import sys
import tempfile

import numpy as np

from yarp.util.write_files import mol_write_yp
from yarp.yarpecule.input_parsers import xyz_parse


def run_obabel_local_optimization(elements, geo, bond_mat, adj_mat, lot, maxiter):
    """Optimize one imposed molecular graph with an isolated Open Babel call.

    Parameters
    ----------
    elements : sequence
        Atomic element labels for the molecule.
    geo : ndarray (N x 3)
        Starting Cartesian coordinates.
    bond_mat : ndarray (N x N)
        Bond-electron matrix to write to the Open Babel MOL input.
    adj_mat : ndarray (N x N)
        Adjacency matrix corresponding to ``bond_mat``.
    lot : str
        Open Babel force field, for example ``"uff"``.
    maxiter : int
        Maximum number of local-optimization steps.

    Returns
    -------
    ndarray (N x 3) or None
        Optimized Cartesian coordinates, or ``None`` when Open Babel cannot
        produce a readable optimized geometry.

    Notes
    -----
    The optimization is run through ``obabel_worker.py`` in a fresh Python
    process. This prevents native Open Babel force-field state from carrying
    over between independent YARP candidates.
    """
    with tempfile.TemporaryDirectory(prefix="yarp_obabel_") as tmp_dir:
        input_mol = f"{tmp_dir}/input.mol"
        output_xyz = f"{tmp_dir}/optimized.xyz"

        # mol_write_yp expects a string, not a pathlib.Path.
        mol_write_yp(
            str(input_mol),
            elements,
            geo,
            bond_mat,
            adj_mat,
        )

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "yarp.util.obabel_worker",
                    input_mol,
                    output_xyz,
                    str(lot),
                    str(maxiter),
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError:
            return None

        if result.returncode != 0:
            return None

        try:
            out_elements, opt_geom = xyz_parse(output_xyz, multiple=False)
        except Exception:
            return None

        if len(out_elements) != len(elements):
            return None

        return np.asarray(opt_geom, dtype=float)


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
    return run_obabel_local_optimization(
        molecule.elements,
        molecule.geo,
        molecule.bond_mats[0],
        molecule.adj_mat,
        lot,
        maxiter,
    )


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
    return run_obabel_local_optimization(
        conformer.elements,
        conformer.geo,
        target_bem,
        target_adj,
        lot,
        maxiter,
    )
