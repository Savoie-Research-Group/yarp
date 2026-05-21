"""
Tests for EGAT atom-map normalization helpers.
"""

import pytest

from yarp.reaction.ml_barrier import normalize_reaction_smiles_for_egat


def test_sparse_maps_are_densified_zero_based():
    rsmiles, psmiles = normalize_reaction_smiles_for_egat(
        '[C:3]([O:5][H:11])([H:6])([H:9])[H:10]',
        '[C:3]([H:6])([H:9])([H:10])[H:11].[O:5]',
    )

    assert ':0]' in rsmiles
    assert ':5]' in rsmiles
    assert ':6]' not in rsmiles
    assert set(rsmiles.split(':')) != set()
    assert sorted({int(part.split(']')[0]) for part in rsmiles.split(':')[1:]}) == [0, 1, 2, 3, 4, 5]
    assert sorted({int(part.split(']')[0]) for part in psmiles.split(':')[1:]}) == [0, 1, 2, 3, 4, 5]


def test_mismatched_maps_raise():
    with pytest.raises(ValueError, match="atom-map sets differ"):
        normalize_reaction_smiles_for_egat('[C:1][O:3]', '[C:1]')
