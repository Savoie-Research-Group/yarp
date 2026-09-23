"""Self-contained, corpus-derived reaction hash regression tests.

The committed pickles contain 200 symmetry, 100 nonisomorphic, 100 direction,
and 5 dropped network reverse-pair cases. No external corpus is needed.
The independent graph oracle deliberately checks element/mass/BEM topology,
not stereochemistry, which remains upstream work for yarpecule hashing.
"""

from copy import deepcopy
from pathlib import Path
import pickle
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from yarp.yarpecule.hashes import reaction_hash


pytestmark = pytest.mark.filterwarnings(
    "ignore:Element-inconsistent atom maps detected:RuntimeWarning"
)


PICKLES = Path(__file__).resolve().parents[1] / "pickles"


def load_cases(name):
    with (PICKLES / name).open("rb") as stream:
        payload = pickle.load(stream)
    assert payload["version"] == 1
    return payload["cases"]


SYMMETRY = load_cases("reaction_hash_symmetry.pkl")
NONISOMORPHIC = load_cases("reaction_hash_nonisomorphic.pkl")
DIRECTION = load_cases("reaction_hash_direction.pkl")
NETWORK_REVERSES = load_cases("reaction_hash_network_reverses.pkl")


def remap_product(rxn, permutation):
    variant = deepcopy(rxn)
    anchor = variant.reactant.graph
    other = variant.product.graph
    maps = [anchor._atom_info[i]["atom_map"] for i in range(len(anchor.elements))]
    by_map = {atom_map: i for i, atom_map in enumerate(maps)}
    for i in range(len(other.elements)):
        old_map = other._atom_info[i]["atom_map"]
        other._atom_info[i]["atom_map"] = maps[permutation[by_map[old_map]]]
    return variant


def reaction_graph(rxn):
    """Independent exact labeled graph of both mapped endpoint BEMs."""
    reactant = rxn.reactant.graph
    product = rxn.product.graph
    maps = [reactant._atom_info[i]["atom_map"] for i in range(len(reactant.elements))]
    product_by_map = {
        product._atom_info[i]["atom_map"]: i for i in range(len(product.elements))
    }
    order = [product_by_map[atom_map] for atom_map in maps]
    r_bem = np.sum(np.asarray(reactant.bond_mats), axis=0)
    p_bem = np.sum(np.asarray(product.bond_mats), axis=0)[np.ix_(order, order)]
    graph = nx.Graph()
    for i, element in enumerate(reactant.elements):
        graph.add_node(
            i,
            label=(
                element,
                round(float(reactant._masses[i]), 6),
                round(float(r_bem[i, i]), 8),
                round(float(p_bem[i, i]), 8),
            ),
        )
    for i in range(len(reactant.elements)):
        for j in range(i + 1, len(reactant.elements)):
            label = (round(float(r_bem[i, j]), 8), round(float(p_bem[i, j]), 8))
            if label != (0.0, 0.0):
                graph.add_edge(i, j, label=label)
    return graph


def isomorphic(left, right):
    return nx.is_isomorphic(
        reaction_graph(left),
        reaction_graph(right),
        node_match=nx.algorithms.isomorphism.categorical_node_match("label", None),
        edge_match=nx.algorithms.isomorphism.categorical_edge_match("label", None),
    )


def test_corpus_case_counts():
    assert (len(SYMMETRY), len(NONISOMORPHIC), len(DIRECTION)) == (200, 100, 100)
    assert len(NETWORK_REVERSES) == 5


@pytest.mark.parametrize("case", SYMMETRY, ids=[f"symmetry-{i:03d}" for i in range(200)])
def test_symmetry_equivalents_share_hash(case):
    original, permutation = case
    variant = remap_product(original, permutation)
    assert isomorphic(original, variant)
    assert reaction_hash(original) == reaction_hash(variant)


@pytest.mark.parametrize("case", NONISOMORPHIC, ids=[f"nonisomorphic-{i:03d}" for i in range(100)])
def test_nonisomorphic_correspondences_have_distinct_hashes(case):
    original, left, right = case
    permutation = list(range(len(original.reactant.graph.elements)))
    permutation[left], permutation[right] = permutation[right], permutation[left]
    variant = remap_product(original, permutation)
    assert not isomorphic(original, variant)
    assert reaction_hash(original) != reaction_hash(variant)


@pytest.mark.parametrize("forward", DIRECTION, ids=[f"direction-{i:03d}" for i in range(100)])
def test_forward_reverse_share_hash(forward):
    reverse = SimpleNamespace(reactant=forward.product, product=forward.reactant)
    assert reaction_hash(forward) == reaction_hash(reverse)


@pytest.mark.parametrize(
    "pair", NETWORK_REVERSES, ids=[f"network-reverse-{i:02d}" for i in range(5)]
)
def test_dropped_network_records_are_exact_reverses(pair):
    first, later = pair
    reversed_later = SimpleNamespace(reactant=later.product, product=later.reactant)
    assert isomorphic(first, reversed_later)
    assert reaction_hash(first) == reaction_hash(later)
