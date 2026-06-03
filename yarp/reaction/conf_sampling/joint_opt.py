import copy
import numpy as np
from openbabel import openbabel as ob
from yarp.util.write_files import mol_write_yp

def ob_joint_optimize(conformer, target_bem, ff_name="uff"):
    """
    Applies constraints based on the target BEM and runs an OpenBabel FF optimization.
    Returns a NEW conformer object with the biased geometry.
    """
    obMol = build_ob_mol(conformer.elements, conformer.geo, target_bem)
    ff = ob.OBForceField.FindForceField(ff_name) or ob.OBForceField.FindForceField("uff")
    if ff is None:
        raise RuntimeError(f"OpenBabel force field not found: {ff_name} or uff")

    if not ff.Setup(obMol):
        ff = ob.OBForceField.FindForceField("uff")
        if ff is None or not ff.Setup(obMol):
            raise RuntimeError("Failed to set up OpenBabel force field for joint optimization")

    ff.ConjugateGradients(500)
    ff.GetCoordinates(obMol)

    biased_conf = copy.deepcopy(conformer)
    biased_conf.geo = np.array([[obMol.GetAtom(i).GetX(), obMol.GetAtom(i).GetY(), obMol.GetAtom(i).GetZ()]
                                for i in range(1, obMol.NumAtoms() + 1)])
    biased_conf.type = f"biased_{conformer.type}"

    return biased_conf


def bondmat_to_adjmat(bond_mat):
    adj_mat = copy.deepcopy(bond_mat)
    for count_i, i in enumerate(bond_mat):
        for count_j, j in enumerate(i):
            if j and count_i != count_j:
                adj_mat[count_i][count_j] = 1.0
            if count_i == count_j:
                adj_mat[count_i][count_i] = 0.0
    return adj_mat


def build_ob_mol(elements, coords, bond_mat):
    """
    Builds an OBMol object using explicit BEM bonding info.
    """
    adj_mat = bondmat_to_adjmat(bond_mat)
    obMol = ob.OBMol()
    conv = ob.OBConversion()
    conv.SetInFormat("mol")

    import os
    import tempfile

    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mol", delete=False) as tmp:
            tmp_file = tmp.name
        mol_write_yp(tmp_file, elements, coords, bond_mat, adj_mat)
        conv.ReadFile(obMol, tmp_file)
    finally:
        if tmp_file is not None:
            os.unlink(tmp_file)

    return obMol
