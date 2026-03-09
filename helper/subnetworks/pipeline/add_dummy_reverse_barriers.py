#!/usr/bin/env python3
"""Add dummy reverse barriers to YARP reaction objects.

For each reaction object found in a reactions container, set `reverse_barrier` to a value that
is randomly either 20% lower or 20% higher than the forward barrier.
"""

import pickle
import random
from pathlib import Path


def is_reaction_like(obj):
    """Return True for objects that look like YARP reaction objects."""
    return hasattr(obj, "reverse_barrier") and (
        hasattr(obj, "forward_barrier") or hasattr(obj, "barrier")
    )


def iter_reaction_objects(container):
    """Yield reaction-like objects contained in nested dict/list/tuple/set."""
    seen_container_ids = set()
    stack = [container]

    while stack:
        current = stack.pop()
        obj_id = id(current)
        if obj_id in seen_container_ids:
            continue
        seen_container_ids.add(obj_id)

        if is_reaction_like(current):
            yield current
            continue

        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, (list, tuple, set)):
            stack.extend(current)


def build_reverse_barrier(forward):
    """Create a reverse barrier object from a forward barrier object."""
    factor = random.choice((0.8, 1.2))

    if isinstance(forward, dict):
        reverse = {}
        for key, value in forward.items():
            try:
                reverse[key] = float(value) * factor
            except (TypeError, ValueError):
                # Keep non-numeric values as-is.
                reverse[key] = value
        return reverse

    return float(forward) * factor


def add_dummy_reverse_barriers(
    reactions_payload,
    *,
    output_path=None,
    seed=None,
    verbose=False,
):
    """Update reverse barriers in an already-loaded reactions object.

    Parameters
    ----------
    reactions_payload
        Already-loaded pickle object (usually a reaction dictionary).
    output_path
        Optional path to save the updated payload as a pickle.
    seed
        Random seed for reproducible +/-20% assignments.
    verbose
        If True, print update/save summary.

    Returns
    -------
    (updated_payload, written_path, updated_count, skipped_count)
        updated_payload: The mutated reactions payload.
        written_path: Path where pickle was written, or None if not saved.
        updated_count: Number of reactions with reverse barrier assigned.
        skipped_count: Number of reactions skipped due to missing/invalid forward barrier.
    """
    if seed is not None:
        random.seed(seed)

    updated_payload = reactions_payload
    updated_count = 0
    skipped_count = 0

    for rxn in iter_reaction_objects(updated_payload):
        forward = getattr(rxn, "forward_barrier", None)
        if forward is None:
            forward = getattr(rxn, "barrier", None)

        if forward is None:
            skipped_count += 1
            continue

        try:
            rxn.reverse_barrier = build_reverse_barrier(forward)
            updated_count += 1
        except (TypeError, ValueError):
            skipped_count += 1

    written_path = None
    if output_path is not None:
        written_path = Path(output_path)
        written_path.parent.mkdir(parents=True, exist_ok=True)
        with written_path.open("wb") as f:
            pickle.dump(updated_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        print(f"Updated reactions: {updated_count}")
        print(f"Skipped reactions: {skipped_count}")
        if written_path is not None:
            print(f"Wrote pickle: {written_path}")

    return updated_payload, written_path, updated_count, skipped_count


def load_pickle_payload(pickle_path):
    """Load a pickle payload with an informative missing-module error."""
    path = Path(pickle_path)
    if not path.exists():
        raise FileNotFoundError(f"Input pickle not found: {path}")

    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to unpickle because a required module is missing. "
            "Activate the Python environment that has YARP installed, then retry."
        ) from exc
