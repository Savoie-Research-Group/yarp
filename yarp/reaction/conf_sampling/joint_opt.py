import copy
import os
import numpy as np
from openbabel import pybel
from rdkit.Chem import AllChem

from yarp.yarpecule.graph.adjacency import compare_adjacency
from yarp.util.rdkit import rdkit_joint_opt
from yarp.util.obabel import obabel_joint_opt


def joint_optimize(conformer, target_bem, lot="uff"):
    """
    Biases a conformer's geometry toward a target BEM via low-level force
    field optimization. Returns a NEW conformer object with the biased
    geometry, or None if no optimizer could produce a geometry consistent
    with the target BEM's connectivity.

    Mirrors the RDKit-first / Open Babel fallback pattern used by
    yarp.reaction.generate_rxns.quick_geom_opt: RDKit is tried first, and its
    result is only kept if the resulting geometry's connectivity matches
    target_bem; otherwise Open Babel is tried as a fallback, and if that also
    fails to reproduce the target connectivity, None is returned so the
    caller can skip the pair.
    """
    target_adj = bondmat_to_adjmat(target_bem)

    # First, attempt to bias the geometry with RDKit
    rd_opt_g = rdkit_joint_opt(conformer, target_bem, target_adj, lot=lot)

    # Check if optimization reproduced the target connectivity
    rd_ok = rd_opt_g is not None and compare_adjacency(conformer.elements, rd_opt_g, target_adj)[0]

    if not rd_ok:
        # If RDKit generated a garbage geom (or failed outright), try Open Babel
        ob_opt_g = obabel_joint_opt(conformer, target_bem, target_adj, lot=lot)

        # If Open Babel fails too, we return None
        if ob_opt_g is None:
            return None
        if not compare_adjacency(conformer.elements, ob_opt_g, target_adj)[0]:
            return None

        opt_geo = ob_opt_g

    # Otherwise, if RDKit gave a valid geom, use that one
    else:
        opt_geo = rd_opt_g

    biased_conf = copy.deepcopy(conformer)
    biased_conf.geo = opt_geo
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
