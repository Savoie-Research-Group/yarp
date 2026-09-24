"""Self-contained, corpus-derived reaction hash regression tests.

The committed pickles contain 200 symmetry, 100 nonisomorphic, 100 direction,
and 5 dropped network reverse-pair cases. No external corpus is needed.
The independent graph oracle deliberately checks element/mass/BEM topology,
not stereochemistry, which remains upstream work for yarpecule hashing.
"""

from copy import deepcopy
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from yarp.reaction.reaction import reaction
from yarp.yarpecule.hashes import reaction_hash

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


def test_corpus_case_counts(
    reaction_hash_symmetry_cases,
    reaction_hash_nonisomorphic_cases,
    reaction_hash_direction_cases,
    reaction_hash_network_reverse_cases,
):
    assert (
        len(reaction_hash_symmetry_cases),
        len(reaction_hash_nonisomorphic_cases),
        len(reaction_hash_direction_cases),
        len(reaction_hash_network_reverse_cases),
    ) == (200, 100, 100, 5)


@pytest.mark.parametrize(
    "case_index", range(200), ids=[f"symmetry-{i:03d}" for i in range(200)]
)
def test_symmetry_equivalents_share_hash(case_index, reaction_hash_symmetry_cases):
    original, permutation = reaction_hash_symmetry_cases[case_index]
    variant = remap_product(original, permutation)
    assert isomorphic(original, variant)
    assert reaction_hash(original) == reaction_hash(variant)


@pytest.mark.parametrize(
    "case_index", range(100), ids=[f"nonisomorphic-{i:03d}" for i in range(100)]
)
def test_nonisomorphic_correspondences_have_distinct_hashes(
    case_index, reaction_hash_nonisomorphic_cases
):
    original, left, right = reaction_hash_nonisomorphic_cases[case_index]
    permutation = list(range(len(original.reactant.graph.elements)))
    permutation[left], permutation[right] = permutation[right], permutation[left]
    variant = remap_product(original, permutation)
    assert not isomorphic(original, variant)
    assert reaction_hash(original) != reaction_hash(variant)


@pytest.mark.parametrize(
    "case_index", range(100), ids=[f"direction-{i:03d}" for i in range(100)]
)
def test_forward_reverse_share_hash(case_index, reaction_hash_direction_cases):
    forward = reaction_hash_direction_cases[case_index]
    reverse = SimpleNamespace(reactant=forward.product, product=forward.reactant)
    assert reaction_hash(forward) == reaction_hash(reverse)


@pytest.mark.parametrize(
    "case_index", range(5), ids=[f"network-reverse-{i:02d}" for i in range(5)]
)
def test_dropped_network_records_are_exact_reverses(
    case_index, reaction_hash_network_reverse_cases
):
    first, later = reaction_hash_network_reverse_cases[case_index]
    reversed_later = SimpleNamespace(reactant=later.product, product=later.reactant)
    assert isomorphic(first, reversed_later)
    assert reaction_hash(first) == reaction_hash(later)


class TestReactionHashIntegration:
    def test_symmetry_equivalent_reactions(self, reaction_hash_symmetry_cases):
        original, permutation = reaction_hash_symmetry_cases[0]
        variant = remap_product(original, permutation)
        assert isomorphic(original, variant)
        first = reaction(original.reactant.graph, original.product.graph)
        second = reaction(variant.reactant.graph, variant.product.graph)
        assert first.hash == second.hash

    def test_distinct_correspondence_reactions(self, reaction_hash_nonisomorphic_cases):
        original, left, right = reaction_hash_nonisomorphic_cases[0]
        permutation = list(range(len(original.reactant.graph.elements)))
        permutation[left], permutation[right] = permutation[right], permutation[left]
        variant = remap_product(original, permutation)
        assert not isomorphic(original, variant)
        first = reaction(original.reactant.graph, original.product.graph)
        second = reaction(variant.reactant.graph, variant.product.graph)
        assert first.hash != second.hash

    def test_reverse_reactions(self, reaction_hash_direction_cases):
        forward = reaction_hash_direction_cases[0]
        first = reaction(forward.reactant.graph, forward.product.graph)
        second = reaction(forward.product.graph, forward.reactant.graph)
        assert first.hash == second.hash
