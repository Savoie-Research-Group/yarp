"""One-off export of the three corpus-derived reaction-hash case types.

This development script reads the trusted, git-ignored prototype corpus and
turns its selected 200/100/100 cases into self-contained test pickles. Normal
pytest runs load only the resulting fixtures under test/pickles/.
"""

import pickle
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
PROTOTYPE = ROOT / "debug/reaction_hash_canonicalization"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROTOTYPE))

import test_atom_hash_ordering_stress as source  # noqa: E402
from yarp.yarpecule.hashes import reaction_hash  # noqa: E402


DESTINATION = ROOT / "test/pickles"


def write(name, cases):
    path = DESTINATION / name
    with path.open("wb") as stream:
        pickle.dump({"version": 1, "cases": cases}, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{path}: {len(cases)} cases")


def main():
    write(
        "reaction_hash_symmetry.pkl",
        [
            (source._reaction(case.source, case.key), case.permutation)
            for case in source.SYMMETRY_CASES
        ],
    )
    write(
        "reaction_hash_nonisomorphic.pkl",
        [
            (source._reaction(case.source, case.key), case.left, case.right)
            for case in source.NEGATIVE_CASES
        ],
    )
    write(
        "reaction_hash_direction.pkl",
        [source._reaction(case.source, case.key) for case in source.DIRECTION_CASES],
    )

    # Preserve the five reverse records removed by the offline network-fixture
    # migration, independently of whether the historical commit remains local.
    fixture = "test/pickles/khp2pp22_soergel_beam2_cyc3.pkl"
    original = pickle.loads(subprocess.check_output(
        ["git", "show", f"1017e90578354e3112abc79d54f9fce54b6f9a43:{fixture}"],
        cwd=ROOT,
    ))
    first_by_hash = {}
    pairs = []
    for later in original.values():
        key = reaction_hash(later)
        if key in first_by_hash:
            pairs.append((first_by_hash[key], later))
        else:
            first_by_hash[key] = later
    write("reaction_hash_network_reverses.pkl", pairs)


if __name__ == "__main__":
    main()
