"""Minimal bond-electron matrix helpers for sieve."""
import numpy as np


def return_e(bond_mat):
    """Valence electrons per atom: sum(2*bond_mat, axis=1) - diag(bond_mat)."""
    return np.sum(2 * bond_mat, axis=1) - np.diag(bond_mat)
