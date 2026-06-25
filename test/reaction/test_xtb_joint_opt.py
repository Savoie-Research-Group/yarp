import numpy as np
import pytest

pytest.importorskip("openbabel")

from yarp.reaction.conf_sampling.joint_opt import _build_xtb_command
from yarp.reaction.conf_sampling.joint_opt import bem_to_distance_constraints
from yarp.reaction.conf_sampling.joint_opt import write_xtb_xcontrol


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


def test_write_xtb_xcontrol_preserves_legacy_constraint_format(tmp_path):
    constraints = bem_to_distance_constraints(["C", "H"], np.array([[0, 1], [1, 0]]))
    xcontrol = tmp_path / "joint_opt.xcontrol"

    write_xtb_xcontrol(xcontrol, constraints, force_constant=1.0)

    assert xcontrol.read_text() == (
        "$constrain\n"
        "force constant=1.0\n"
        "distance: 1, 2, 1.1470\n"
        "$\n\n"
    )


def test_build_xtb_command_maps_multiplicity_to_uhf():
    cmd = _build_xtb_command(
        "input.xyz",
        "joint_opt.xcontrol",
        namespace="joint_opt",
        lot="gfn2",
        charge=-1,
        multiplicity=2,
        n_cpus=4,
        scf_iters=500,
    )

    assert cmd == [
        "xtb",
        "input.xyz",
        "--iterations",
        "500",
        "--chrg",
        "-1",
        "--uhf",
        "1",
        "--namespace",
        "joint_opt",
        "--opt",
        "--parallel",
        "4",
        "--input",
        "joint_opt.xcontrol",
        "--gfn",
        "2",
    ]
