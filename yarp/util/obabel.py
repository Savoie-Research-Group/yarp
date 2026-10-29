import inspect
import json
import os
import subprocess
import sys
import numpy as np
from openbabel import pybel, openbabel as ob

from yarp.util.properties import el_to_an
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

def obabel_embed(elements, bond_mat, fc, lot="uff", maxiter=500, seed=0xf00d):
    '''
    Generate a 3D geometry for a molecular graph using Open Babel.

    This is the fallback for molecules that RDKit cannot embed (e.g., strained
    rings with cumulated double bonds). Each call runs the build in a fresh
    process (`_obabel_embed_worker()`) with Open Babel's random seed fixed, so the
    same graph always gets the same geometry, independent of earlier builds.

    Parameters
    ----------
    elements : list
        element labels, indexed to the atoms of the molecule

    bond_mat : nd array (N x N)
        bond orders on the off-diagonal (a bond-electron matrix works)

    fc : list
        formal charge of each atom

    lot : string
        Force field used to relax the built geometry

    maxiter : int
        Maximum number of optimization steps

    seed : int
        Seed for Open Babel's builder (set through the OB_RANDOM_SEED environment variable)

    Returns
    -------
    geo : nd array (N x 3)
        generated geometry, indexed to `elements`
    '''
    payload = {"atomic_numbers": [el_to_an[_.lower()] for _ in elements],
               "bond_mat": np.asarray(bond_mat).tolist(),
               "fc": [int(_) for _ in fc], "lot": lot, "maxiter": maxiter}

    # Open Babel's builder draws from one random generator per process, seeded once from
    # OB_RANDOM_SEED; a fresh process per build makes each build that generator's first use.
    # The worker runs from its own source in isolated mode (-I), so modules in the current
    # directory cannot shadow its imports, and yarp itself is not imported.
    worker = inspect.getsource(_obabel_embed_worker) + "\n_obabel_embed_worker()\n"
    proc = subprocess.run([sys.executable, "-I", "-c", worker],
                          input=json.dumps(payload), capture_output=True, text=True,
                          env={**os.environ, "OB_RANDOM_SEED": str(seed)})
    if proc.returncode != 0:
        # a worker that dies without a traceback (e.g., killed) leaves stderr empty
        error = (proc.stderr.strip().splitlines() or [f"worker exited with code {proc.returncode}"])[-1]
        raise RuntimeError(f"Open Babel could not build a geometry: {error}")

    return np.array(json.loads(proc.stdout.strip().splitlines()[-1])["geo"])


def _obabel_embed_worker():
    '''
    Run one Open Babel build for `obabel_embed()` in its own process: reads the graph as
    JSON on stdin and writes the geometry as JSON on the last line of stdout.
    Self-contained (its own imports, no yarp), since `obabel_embed()` runs it from its source.
    '''
    import json
    import sys
    from openbabel import openbabel as ob, pybel

    payload = json.loads(sys.stdin.read())

    mol = pybel.Molecule(ob.OBMol())
    for count_i, i in enumerate(payload["atomic_numbers"]):
        atom = mol.OBMol.NewAtom()
        atom.SetAtomicNum(i)
        atom.SetFormalCharge(payload["fc"][count_i])
    for i in range(len(payload["atomic_numbers"])):
        for j in range(i + 1, len(payload["atomic_numbers"])):
            if payload["bond_mat"][i][j] > 0:
                mol.OBMol.AddBond(i + 1, j + 1, int(payload["bond_mat"][i][j]))

    if not ob.OBBuilder().Build(mol.OBMol):
        raise RuntimeError("Open Babel could not build a geometry.")

    # A fresh process has no cached force field, so _ensure_setup_needed() is not needed here
    mol.localopt(forcefield=payload["lot"], steps=payload["maxiter"])

    print(json.dumps({"geo": [atom.coords for atom in mol.atoms]}))


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
