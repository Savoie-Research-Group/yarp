"""
Builds a 3D geometry for a molecular graph with Open Babel, run as a process of its own.

Open Babel's builder draws its random bond directions from a single generator per process,
seeded once, on first use, from the OB_RANDOM_SEED environment variable (else from
std::random_device). The generator cannot be reset afterwards, so in one process the
geometry of a molecule would depend on every build before it. `obabel_embed()`
(yarp/util/obabel.py) therefore runs this file in a fresh process for every build, with
OB_RANDOM_SEED set, so the same graph always gets the same geometry.

Only openbabel and numpy are imported, to keep the start-up of each process short.
Reads {"atomic_numbers", "bond_mat", "fc", "lot", "maxiter"} as JSON on stdin and writes
{"geo": [[x, y, z], ...]} as JSON on the last line of stdout.
"""
import json
import sys

import numpy as np
from openbabel import openbabel as ob, pybel


def build(atomic_numbers, bond_mat, fc, lot, maxiter):
    """
    Place the atoms with Open Babel's builder and relax them with the `lot` force field.
    The geometry follows the order of `atomic_numbers`.
    """
    mol = pybel.Molecule(ob.OBMol())
    for count_i, i in enumerate(atomic_numbers):
        atom = mol.OBMol.NewAtom()
        atom.SetAtomicNum(i)
        atom.SetFormalCharge(fc[count_i])
    for i in range(len(atomic_numbers)):
        for j in range(i + 1, len(atomic_numbers)):
            if bond_mat[i][j] > 0:
                mol.OBMol.AddBond(i + 1, j + 1, int(bond_mat[i][j]))

    if not ob.OBBuilder().Build(mol.OBMol):
        raise RuntimeError("Open Babel could not build a geometry.")

    # A fresh process has no cached force field, so no stale parameters can be reused here
    # (compare _ensure_setup_needed() in obabel.py).
    mol.localopt(forcefield=lot, steps=maxiter)

    return np.array([atom.coords for atom in mol.atoms])


if __name__ == "__main__":
    geo = build(**json.loads(sys.stdin.read()))
    print(json.dumps({"geo": geo.tolist()}))
