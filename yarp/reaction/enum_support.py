"""Support functions for reaction enumeration."""

from collections import Counter
from dataclasses import dataclass
import numpy as np
from yarp.util.properties import el_valence
from copy import copy
from itertools import combinations
from typing import Iterable, Tuple
from yarp.yarpecule.lewis.bem_score import return_formals
from yarp.yarpecule.yarpecule import yarpecule
from yarp.util.misc import prepare_list, merge_arrays

@dataclass(frozen=True)
class SharedAtomB2F2Change:
    """One legacy shared-atom B2F2 bond/electron rearrangement."""

    bonds_to_form: tuple
    shared_atom: int
    electron_donor: int
    criterion: str


def legacy_shared_atom_b2f2_changes(n, formset, radicals, bond_mat, elements):
    """Return strict and loose legacy changes for a shared-atom B2F2 step.

    ``None`` means the two broken bonds are not a shared-atom special case. An
    empty tuple means that they are a shared-atom case, but no untouched
    neighbour satisfies either applicable legacy lone-pair criterion.

    ``elements`` is required because the loose criterion must distinguish
    hydrogens from heavy atoms and compare the donor and shared-atom elements.
    """
    if n != 2 or radicals:
        return None

    shared_atoms = [
        atom for atom, count in Counter(formset).items() if count > 1
    ]
    if len(shared_atoms) != 1:
        return None

    shared_atom = shared_atoms[0]
    non_shared_atoms = [atom for atom in formset if atom != shared_atom]
    if len(non_shared_atoms) != 2 or len(set(non_shared_atoms)) != 2:
        return ()

    changes = []
    for neighbor, bond_order in enumerate(bond_mat[shared_atom]):
        if neighbor == shared_atom or bond_order <= 0 or neighbor in formset:
            continue

        other_connections = [
            atom
            for atom, order in enumerate(bond_mat[neighbor])
            if atom not in (neighbor, shared_atom) and order > 0
        ]
        has_lone_electrons = bond_mat[neighbor, neighbor] > 0
        strict_match = not other_connections and has_lone_electrons

        heavy_connections = [
            atom
            for atom in other_connections
            if elements[atom].lower() != "h"
        ]
        loose_match = (
            len(other_connections) < 3
            and not heavy_connections
            and elements[neighbor].lower() != elements[shared_atom].lower()
            and has_lone_electrons
        )

        if strict_match or loose_match:
            if strict_match and loose_match:
                criterion = "strict+loose"
            elif strict_match:
                criterion = "strict"
            else:
                criterion = "loose"
            changes.append(
                SharedAtomB2F2Change(
                    bonds_to_form=(
                        frozenset(non_shared_atoms),
                        frozenset((shared_atom, neighbor)),
                    ),
                    shared_atom=shared_atom,
                    electron_donor=neighbor,
                    criterion=criterion,
                )
            )

    return tuple(changes)


def apply_legacy_shared_atom_b2f2(bond_mat, bonds_to_break, change):
    """Apply the complete legacy shared-atom B2F2 BEM transformation.

    The legacy implementation first returns one electron to each endpoint of
    every broken bond, consumes one electron from each endpoint of every formed
    bond, and finally transfers an electron pair from the untouched terminal
    neighbour to the shared atom.
    """
    product_bmat = np.array(bond_mat, copy=True)

    for bond in bonds_to_break:
        atom_i, atom_j = bond[:2]
        product_bmat[atom_i, atom_j] -= 1
        product_bmat[atom_j, atom_i] -= 1
        product_bmat[atom_i, atom_i] += 1
        product_bmat[atom_j, atom_j] += 1

    for bond in change.bonds_to_form:
        atom_i, atom_j = tuple(bond)
        product_bmat[atom_i, atom_j] += 1
        product_bmat[atom_j, atom_i] += 1
        product_bmat[atom_i, atom_i] -= 1
        product_bmat[atom_j, atom_j] -= 1

    product_bmat[change.electron_donor, change.electron_donor] -= 2
    product_bmat[change.shared_atom, change.shared_atom] += 2

    return product_bmat

def _reactive_maps_from_react(react):
    """
    Convert the public reactive-atoms value to a set of atom-map ids.
    """
    if react is None or react == []:
        return set()
    if isinstance(react, set):
        return set(react)
    if isinstance(react, tuple):
        return set(react)
    if isinstance(react, list) and len(react) == 1 and isinstance(react[0], (set, list, tuple)):
        return set(react[0])
    return set(react)


def atom_map_to_local_index(yarp_like):
    """
    Return {atom_map: local_index} for a yarpecule and reject duplicate maps.
    """
    by_map = {}
    for local_idx, info in yarp_like.atom_info.items():
        atom_map = info.get("atom_map")
        if atom_map is None:
            continue
        if atom_map in by_map:
            raise ValueError(f"Duplicate atom map {atom_map} found in reactant.")
        by_map[atom_map] = local_idx
    return by_map


def _resolve_reactive_atoms_for_candidate(yarp_like, react, verbose=False):
    """
    Resolve public reactive atom maps to candidate-local atom indices.

    The public YAML/API value is expressed as atom-map IDs because those are the
    only stable atom identifiers across pickle restarts, product separation, and
    local atom reordering. The low-level enumeration code still operates on
    candidate-local atom indices, so every candidate has to resolve the map IDs
    against its own current atom table immediately before enumeration.

    Missing maps are intentionally not errors. A candidate can be a separated
    product fragment, or just a molecule that does not contain this depth's
    requested reactive atom subset. In those cases we use the intersection of
    requested maps and candidate maps. If the intersection is empty, the caller
    skips this candidate cleanly by returning no products.
    """
    react_maps = _reactive_maps_from_react(react)
    if not react_maps:
        return [], [], []

    # ``by_map`` is the bridge from stable user-facing map IDs to the local
    # integer indices expected by adjacency/bond-matrix enumeration routines.
    by_map = atom_map_to_local_index(yarp_like)
    present = sorted(react_maps & set(by_map))
    missing = sorted(react_maps - set(by_map))

    if not present:
        if verbose:
            print("   + Molecule contains no atoms in reactive set; skipping enumeration.")
        return None, present, missing

    # Preserve the historical internal shape: a list of sets, where each set is
    # a candidate-local reactive atom group. The public representation remains
    # atom-map IDs; only enumeration internals see local indices.
    local_react = [set(by_map[_] for _ in present)]
    return local_react, present, missing

def return_radicals(y, all_bmats=False):
    """
    Returns the indices of the atoms that are radicals in the yarpecule.
    if all_bmats is True then all bond electron matrices are considered else only the first one
    """
    if all_bmats:
        return set([i for bmat in y.lewis.bond_mats for i in range(len(bmat)) if bmat[i][i] % 2 == 1])
    else:
        return set([i for i in range(len(y.lewis.bond_mats[0])) if y.lewis.bond_mats[0][i][i] % 2 == 1])

def return_bondtypes(yarpecules, b_inds=[]):
    """
    This function provides a shortcut for enumerating "break m form n" products without generating intermediate 
    zwitterionic/dangling bond species. The function returns a list of bonds for each yarpecule. Each bond is a tuple
    of the form (i,j,i_hash,j_hash,bond_order) where i and j are the indices of the atoms in the bond, i_hash and j_hash
    are the hashes of the atoms, and bond_order is the bond order of the bond taken from the bond_mat at the index supplied
    by b_inds.

    Parameters
    ----------
    yarpecules: list of yarpecules
                This list holds the yarpecules that should be reacted. 

    b_inds: list of indices
            This holds the index of the bond_mat that the user wants the return the bond orders for. 
            By default the first bond_mat is used. 
    """
    # Wrap yarpecules in a list if only one is supplied
    yarpecules = prepare_list(yarpecules)

    # Use the first bond_mat if no indices are supplied
    if len(b_inds) != len(yarpecules):
        b_inds = [0 for _ in range(len(yarpecules))]

    # tuple holds: bond between atoms i and j, with their hashes, and the bond order taken from the bond_mat at the index supplied by b_inds. This list of bonds is returned for each yarpecule.
    return [[(count_i, j, y._atom_hashes[count_i], y._atom_hashes[j], y.lewis.bond_mats[b_inds[count_y]][count_i][j]) for count_i, i in enumerate(return_adjlist(y)) for j in i if count_i <= j] for count_y, y in enumerate(yarpecules)]

def unique_set_partition_generator(seq: Iterable, group_size: int):
    """
    Yield all unique partitions of `seq` into groups of `group_size`.
    Generates each partition exactly once, in canonical order,
    without holding all previous results in memory.

    This function returns the unique partitionings of group_size of the elements of seq. The returned partitions
    are not distinguishable by ordering within partitions or the ordering between partitions. For example, 
    if seq = [1,2,3,4] and group_size=2, then [(1,2),(3,4)], [(2,1),(4,3)], and [(3,4),(1,2)] would all be considered
    the same partition. This function is used to generate all possible partitions of atoms that can form 
    bonds, so a (1,2) bond is the same as a (2,1) bond and a [(1,2),(3,4)] pair of bonds is the same as a 
    [(3,4),(1,2)] pair of bonds, etc.

    When len(seq) is not divisible by group_size, all possible subsets of size
    (groups_needed * group_size) are partitioned, so no valid grouping is missed.
    """
    seq = tuple(seq)                     # tuple => O(1) index lookup
    n = len(seq)                         # O(1) lookup

    # Needs to be at least 1 and not larger than seq, otherwise no partition is possible
    if group_size <= 0 or group_size > n:
        return

    groups_needed = n // group_size      # number of complete groups we can form

    def helper(available: tuple, accum: tuple):
        """
        Recursively build up `accum`, a tuple of grouped index-tuples.
        Canonical order is enforced by always anchoring the next group on
        the first element of `available` — there is no choice here, which
        is what prevents duplicate partitions from being generated.
        """
        if len(accum) == groups_needed:   # base case: complete partition
            # Map indices back to original elements exactly once:
            yield tuple(frozenset(seq[i] for i in grp) for grp in accum)
            return

        # Canonical anchor: first available index MUST start the next group
        first, *rest = available
        for combo in combinations(rest, group_size - 1):
            # Build the remaining available indices by excluding the chosen combo
            remaining = tuple(i for i in rest if i not in combo)
            yield from helper(remaining, accum + ((first,) + combo,))

    # When n is not divisible by group_size, we iterate over all subsets of
    # exactly (groups_needed * group_size) elements and partition each one.
    # This ensures every valid grouping is considered regardless of which
    # elements are left over.
    elements_needed = groups_needed * group_size
    seen = set()                          # tracks yielded partitions to avoid duplicates
    for subset in combinations(range(n), elements_needed):
        for partition in helper(subset, ()):
            # Different subsets can produce identical frozenset partitions,
            # so we deduplicate before yielding
            key = frozenset(partition)
            if key not in seen:
                seen.add(key)
                yield partition


def add_bonds(bond_mat, bonds, val=1):
    """
    Helper function for bnfn. Modifies the bond_mat in place.

    Parameters
    ----------
    bond_mat : numpy array or nested list
        2D bond matrix to modify
    bonds : iterable of sequences
        Each bond should be indexable (list, tuple, etc.) with at least 2 elements
    val : int or float
        Value to add to bond matrix elements
    """
    for b in bonds:
        bond_mat[b[0]][b[1]] += val
        bond_mat[b[1]][b[0]] += val
    return bond_mat

# GRAB THE BETTER ONE FROM UTILS
def return_adjlist(yarpecule):
    return [np.where(i)[0].tolist() for count_i, i in enumerate(yarpecule.adj_mat)]

# def unique_set_partition_generator_old(lst, n):
#     """
#     This function returns the unique choose n groupings of the elements of lst. The returned groupings
#     are not distinguishable by ordering within grouping or the ordering of groupings. For example,
#     is lst = [1,2,3,4] and n=2, then [(1,2),(3,4)], [(2,1),(4,3)], and [(3,4),(1,2)] would all be considered
#     the same subgroupings. This function is used to generate all possible groupings of atoms that can form
#     bonds, so a (1,2) bond is the same as a (2,1) bond and a [(1,2),(3,4)] pair of bonds is the same as a
#     [(3,4),(1,2)] pair of bonds, etc.

#     Parameters
#     ----------
#     lst: list of elements
#          for efficiency gains the algorithm assumes that the list is sortable.

#     n: float
#          The number of elements per subgroup

#     Returns
#     -------
#     groupings: lst of frozensets
#          The list of unique unordered groupings is returned in its totality after a recursion. Each grouping is
#          stored as a frozenset which is used because a hashable set is needed in the algorithm.
#     """

#     # Return empty list if lst cannot be evenly divided into groups of size n
#     if len(lst) % n != 0:
#         return []

#     lst = sorted(lst)
#     total_groupings = []

#     # Recursive helper function to generate unique groupings
#     def helper(available_elements, current_partition):
#         # Base case: if no elements are left, add the current partition to total groupings
#         if not available_elements:
#             total_groupings.append(current_partition)
#             return

#         # Always pick the first element to enforce ordering and avoid duplicates
#         first_element = available_elements[0]
#         rest_elements = available_elements[1:]

#         # Generate all combinations of size n-1 from the remaining elements
#         for comb in combinations(rest_elements, n - 1):
#             # Form a group by combining the first element with the current combination
#             group = frozenset([first_element] + list(comb))
#             # Determine the elements that haven't been grouped yet
#             remaining_elements = [e for e in rest_elements if e not in comb]

#             # Recursively build groupings with the remaining elements
#             helper(remaining_elements, current_partition + [group])

#     # Start the recursive process with the full list and an empty partition
#     helper(lst, [])

#     # Remove duplicates by converting partitions to a sorted tuple of sorted frozensets
#     unique_groupings_set = set()
#     for partition in total_groupings:
#         # Sort groups within the partition and convert them to tuples for hashability
#         sorted_partition = tuple(sorted([tuple(sorted(group)) for group in partition]))
#         unique_groupings_set.add(sorted_partition)

#     # Convert back to the desired output format (list of frozensets)
#     return [list(map(frozenset, grouping)) for grouping in unique_groupings_set]