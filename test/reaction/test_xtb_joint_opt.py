import numpy as np
import pytest

pytest.importorskip("openbabel")

from yarp.reaction.external.joint_opt import bem_to_distance_constraints


def test_bem_to_distance_constraints_uses_one_indexed_bonds():
    constraints = bem_to_distance_constraints(
        ["C", "H", "O"],
        np.array(
            [
                [0, 1, 0],
                [1, 0, 2],
                [0, 2, 0],
            ]
        ),
    )

    assert [tuple(c) for c in constraints] == [
        (1, 2, pytest.approx(1.147)),
        (2, 3, pytest.approx(1.048)),
    ]


def test_bem_to_distance_constraints_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        bem_to_distance_constraints(["C"], np.zeros((2, 2)))
