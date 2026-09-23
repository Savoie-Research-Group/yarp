"""One-off migration of a YARP reaction pickle to the current hash scheme.

Usage: python -m yarp.helpers.rekey_reactions input.pkl output.pkl

The input is never overwritten. Duplicate current hashes keep the first
reaction encountered in the input dictionary and print a warning.
Only load pickle files from trusted sources: unpickling executes code.
"""

import argparse
import pickle
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yarp.reaction.reaction import reaction  # noqa: E402
from yarp.yarpecule.hashes import reaction_hash  # noqa: E402


def rekey_reactions(reactions):
    """Return a first-record-wins dictionary keyed by current reaction hashes."""
    output = {}
    for rxn in reactions.values():
        if not isinstance(rxn, reaction):
            raise TypeError("Input dictionary must contain only YARP reactions.")
        rxn.hash = reaction_hash(rxn)
        if rxn.hash in output:
            retained = output[rxn.hash]
            print(
                "WARNING: Deduplicated reaction "
                f"{rxn.id}; retaining earlier reaction {retained.id} "
                f"for hash {rxn.hash}."
            )
            continue
        output[rxn.hash] = rxn
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Trusted input reaction pickle")
    parser.add_argument("destination", type=Path, help="New output pickle")
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error("Destination must differ from source; keep the original as a backup.")
    if args.destination.exists():
        parser.error(f"Destination already exists: {args.destination}")
    with args.source.open("rb") as stream:
        reactions = pickle.load(stream)
    if not isinstance(reactions, dict):
        parser.error("Input pickle must contain a reaction dictionary.")
    migrated = rekey_reactions(reactions)
    with args.destination.open("xb") as stream:
        pickle.dump(migrated, stream)
    print(f"Wrote {len(migrated)} reactions to {args.destination}")


if __name__ == "__main__":
    main()
