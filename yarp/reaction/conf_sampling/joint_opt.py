import copy
import numpy as np
from openbabel import openbabel as ob

def ob_joint_optimize(conformer, target_bem, ff_name="uff"):
    """
    Applies constraints based on the target BEM and runs an OpenBabel FF optimization.
    Returns a NEW conformer object with the biased geometry.
    """
    obMol = build_ob_mol(conformer.elements, conformer.geo)
    ff = ob.OBForceField.FindForceField(ff_name)
    if not ff:
        ff = ob.OBForceField.FindForceField("uff")
    
    ff.Setup(obMol)
    
    # Mock constraints based on Target BEM
    constraints = ob.OBFFConstraints()
    num_atoms = len(conformer.elements)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if target_bem[i][j] > 0:
                constraints.AddDistanceConstraint(i + 1, j + 1, 1.5) 
                
    ff.SetConstraints(constraints)
    ff.ConjugateGradients(500)
    ff.GetCoordinates(obMol)
    
    biased_conf = copy.deepcopy(conformer)
    biased_conf.geo = np.array([[obMol.GetAtom(i).GetX(), obMol.GetAtom(i).GetY(), obMol.GetAtom(i).GetZ()] 
                                for i in range(1, obMol.NumAtoms() + 1)])
    biased_conf.type = f"biased_{conformer.type}"
    return biased_conf


def build_ob_mol(elements, coords):
    """Converts a Yarpecule/Conformer geometry to an OpenBabel OBMol."""
    obMol = ob.OBMol()
    for el, coord in zip(elements, coords):
        obAtom = obMol.NewAtom()
        obAtom.SetAtomicNum(ob.GetAtomicNum(el))
        obAtom.SetVector(*coord)
    return obMol

