#!/usr/bin/env python3
"""Run exactly one Open Babel local optimization in an isolated process."""

import sys
from openbabel import pybel


def main(argv):
    # argv:
    #   0: this script
    #   1: input MOL file
    #   2: output XYZ file
    #   3: force field, e.g. "uff"
    #   4: maximum number of steps
    if len(argv) != 5:
        raise SystemExit(
            "usage: obabel_worker.py INPUT_MOL OUTPUT_XYZ FORCEFIELD STEPS"
        )

    input_mol = argv[1]
    output_xyz = argv[2]
    forcefield = argv[3]
    steps = int(argv[4])

    mol = next(pybel.readfile("mol", input_mol))
    mol.localopt(forcefield=forcefield, steps=steps)
    mol.write("xyz", output_xyz, overwrite=True)


if __name__ == "__main__":
    main(sys.argv)