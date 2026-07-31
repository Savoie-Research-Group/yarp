"""Reaction featurization for the enthalpy-of-reaction EGAT model.

Self-contained copy of the training-time featurizer
(EGAT-JEPA Source/clean_rxn/featurize.py). A reaction is turned into two graphs
(reactant + product) that share the same node set and the same union edge set.

The atomic number is emitted separately as an integer index `Z` consumed by an
nn.Embedding in the model; all other descriptors are small float features.
"""

import numpy as np

from rxn_egat._graph_core import return_matrix, return_reactive
from rxn_egat.rings import adjmat_to_adjlist, return_rings
from rxn_egat.elements import el_to_an

# element symbol (capitalized) -> atomic number, e.g. "C" -> 6
ELEMENT_TO_Z = {k.capitalize(): v for k, v in el_to_an.items()}

# Embedding table size (index 0 = padding). Matches the model's nn.Embedding.
MAX_Z = 100

BOND_ORDERS = [0, 1, 2, 3]
NODE_FEAT_DIM = 5          # [formal_charge, degree/4, n_H/4, in_ring, is_reactive]
EDGE_FEAT_DIM = len(BOND_ORDERS) + 1   # bond-order one-hot + in_ring flag


def _ring_atom_set(adj):
    rings = return_rings(adjmat_to_adjlist(adj), max_size=20, remove_fused=True)
    ring_atoms = set()
    for r in rings:
        ring_atoms.update(r)
    return ring_atoms


def _node_features(elements, adj, fc, reactive_atoms, ring_atoms):
    n = len(elements)
    feats = np.zeros((n, NODE_FEAT_DIM), dtype=np.float32)
    for i in range(n):
        neighbors = np.nonzero(adj[i])[0]
        degree = len(neighbors)
        n_h = sum(1 for j in neighbors if elements[j] == "H")
        feats[i, 0] = fc[i]
        feats[i, 1] = degree / 4.0
        feats[i, 2] = n_h / 4.0
        feats[i, 3] = 1.0 if i in ring_atoms else 0.0
        feats[i, 4] = 1.0 if i in reactive_atoms else 0.0
    return feats


def _edge_features(bond_mat, u, v, ring_atoms):
    feats = np.zeros((len(u), EDGE_FEAT_DIM), dtype=np.float32)
    for e, (i, j) in enumerate(zip(u, v)):
        order = int(bond_mat[i, j])
        if order in BOND_ORDERS:
            feats[e, BOND_ORDERS.index(order)] = 1.0
        else:
            feats[e, BOND_ORDERS.index(3)] = 1.0
        if i in ring_atoms and j in ring_atoms:
            feats[e, len(BOND_ORDERS)] = 1.0
    return feats


def featurize_reaction(r_smiles, p_smiles):
    """Atom-mapped reactant/product SMILES -> graph tensors dict.

    Keys: Z (N,), u/v (E,), x_R/x_P (N, NODE_FEAT_DIM), e_R/e_P (E, EDGE_FEAT_DIM).
    Raises ValueError when the two sides cannot be aligned.
    """
    el_R, adj_R, bond_R, fc_R = return_matrix(r_smiles)
    el_P, adj_P, bond_P, fc_P = return_matrix(p_smiles)

    if el_R != el_P:
        raise ValueError("reactant/product element ordering mismatch")

    elements = el_R
    n = len(elements)
    if n == 0:
        raise ValueError("empty molecule")

    Z = np.array([ELEMENT_TO_Z[e] for e in elements], dtype=np.int64)

    u, v = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if adj_R[i, j] > 0 or adj_P[i, j] > 0:
                u.append(i)
                v.append(j)
    if len(u) == 0:
        raise ValueError("no bonds in reaction")
    u = np.array(u, dtype=np.int64)
    v = np.array(v, dtype=np.int64)

    _, reactive_atoms, *_ = return_reactive(elements, bond_R, bond_P)
    reactive_atoms = set(reactive_atoms)

    ring_atoms = _ring_atom_set(adj_R) | _ring_atom_set(adj_P)

    return {
        "Z": Z,
        "u": u,
        "v": v,
        "x_R": _node_features(elements, adj_R, fc_R, reactive_atoms, ring_atoms),
        "x_P": _node_features(elements, adj_P, fc_P, reactive_atoms, ring_atoms),
        "e_R": _edge_features(bond_R, u, v, ring_atoms),
        "e_P": _edge_features(bond_P, u, v, ring_atoms),
    }
