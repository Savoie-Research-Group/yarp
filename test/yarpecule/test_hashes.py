"""
Testing suite for functions contained in yarp/yarpecule/hashes.py
"""
from importlib import import_module
from copy import deepcopy
from math import fsum
from types import SimpleNamespace

import networkx as nx
import pytest
import numpy as np
from yarp.yarpecule.hashes import (
    bmat_hash,
    reaction_hash,
)
from yarp.yarpecule.yarpecule import yarpecule
from yarp.reaction.reaction import reaction

class TestBmatHash:
    def test_distinguish_mappings(self, haa_canon_smi, haa_full_map_smi):
        """
        Test that bond electron matrix hashes can distinguish between
        the same molecule with different mappings
        """
        haa_canon = yarpecule(haa_canon_smi, mode='yarp', canon=False)
        haa_map = yarpecule(haa_full_map_smi, mode='yarp', canon=False)

        haa_canon_bem_hash = bmat_hash(haa_canon.bond_mats[0])
        haa_map_bem_hash = bmat_hash(haa_map.bond_mats[0])

        assert haa_canon_bem_hash != haa_map_bem_hash

    def test_distinguish_charge(self, benzene_smi, benz_rad_cat_smi):
        """
        Test that bond electron matrix hashes distinguish between charge states
        """
        benz = yarpecule(benzene_smi)
        benz_cat = yarpecule(benz_rad_cat_smi)

        assert benz.elements == benz_cat.elements
        assert np.array_equal(benz.adj_mat, benz_cat.adj_mat)

        benz_bem_hash = bmat_hash(benz.bond_mats[0])
        benz_cat_bem_hash = bmat_hash(benz_cat.bond_mats[0])

        assert benz_bem_hash != benz_cat_bem_hash

class TestAtomHash:
    def test_distinguish_mappings(self, haa_canon_smi, haa_full_map_smi):
        """
        Test that atom hashes will be identical, but show up in different order,
        depending on mapping.
        """
        haa_canon = yarpecule(haa_canon_smi, mode='yarp', canon=False)
        haa_map = yarpecule(haa_full_map_smi, mode='yarp', canon=False)

        assert len(haa_canon.atom_hashes) == len(haa_map.atom_hashes)
        assert not np.array_equal(haa_canon.atom_hashes, haa_map.atom_hashes)

        canon_hash = set(tuple(haa_canon.atom_hashes))
        map_hash = set(tuple(haa_map.atom_hashes))

        assert canon_hash == map_hash

    def test_charge_blind(self, benzene_smi, benz_rad_cat_smi):
        """
        Test that atom hashes are blind to charge state
        """
        benz = yarpecule(benzene_smi)
        benz_cat = yarpecule(benz_rad_cat_smi)

        assert np.array_equal(benz.atom_hashes, benz_cat.atom_hashes)

class TestYpHash:
    def test_mapping_blind(self, haa_canon_smi, haa_full_map_smi):
        """
        Test that different mappings of same molecule produce
        identical yarpecule hashes
        """
        haa_canon = yarpecule(haa_canon_smi, mode='yarp', canon=False)
        haa_map = yarpecule(haa_full_map_smi, mode='yarp', canon=False)

        assert haa_canon.hash == haa_map.hash

    def test_distinguish_charge(self, benzene_smi, benz_rad_cat_smi):
        """
        Test that yarpecule hashes distinguish between charge states
        """
        benz = yarpecule(benzene_smi)
        benz_cat = yarpecule(benz_rad_cat_smi)

        assert benz.hash != benz_cat.hash

class TestRxnHash:
    def test_direct_hash_uses_all_bems_and_mapped_atom_hashes(
        self, cyclohexane_dehydrogenation, monkeypatch
    ):
        reactant = cyclohexane_dehydrogenation.reactant.graph
        product = cyclohexane_dehydrogenation.product.graph
        product_by_map = {
            product.atom_info[i]["atom_map"]: i
            for i in range(len(product.elements))
        }
        product_order = [
            product_by_map[reactant.atom_info[i]["atom_map"]]
            for i in range(len(reactant.elements))
        ]
        combined_bems = list(reactant.bond_mats) + [
            np.asarray(bem)[np.ix_(product_order, product_order)]
            for bem in product.bond_mats
        ]
        expected_bem = np.zeros_like(combined_bems[0])
        for bem in combined_bems:
            expected_bem += bem
        expected_atom_hashes = (
            np.asarray(reactant.atom_hashes)
            + np.asarray(product.atom_hashes)[product_order]
        )
        expected_hash = (
            cyclohexane_dehydrogenation.reactant.hash
            + cyclohexane_dehydrogenation.product.hash
            + np.round(
                fsum(
                    (
                        expected_bem
                        * np.outer(expected_atom_hashes, expected_atom_hashes)
                    ).flat
                ),
                7,
            )
        )

        def unexpected_yarpecule_hash(_):
            raise AssertionError("Reaction hashing must use the direct algebra")

        fsum_calls = []

        def checked_fsum(values):
            fsum_calls.append(True)
            return fsum(values)

        hashes_module = import_module("yarp.yarpecule.hashes")
        monkeypatch.setattr(
            hashes_module,
            "yarpecule_hash",
            unexpected_yarpecule_hash,
        )
        monkeypatch.setattr(hashes_module, "fsum", checked_fsum)
        assert reaction_hash(cyclohexane_dehydrogenation) == expected_hash
        assert fsum_calls == [True]

    def test_element_inconsistent_maps_warn_and_continue(self):
        reactant = yarpecule("[C:0][O:1]", canon=False)
        product = yarpecule("[C:1][O:0]", canon=False)

        with pytest.warns(RuntimeWarning, match="Element-inconsistent atom maps"):
            mapped_reaction = reaction(reactant, product)
        assert isinstance(mapped_reaction.hash, float)

    def test_direct_rehash_revalidates_changed_maps(self):
        """An existing reaction cannot be rehashed after its maps become ambiguous."""
        mapped_reaction = reaction(
            yarpecule("[C:0][O:1]", canon=False),
            yarpecule("[C:0][O:1]", canon=False),
        )
        for graph in (mapped_reaction.reactant.graph, mapped_reaction.product.graph):
            graph._atom_info[1]["atom_map"] = 0

        with pytest.raises(ValueError, match="identical unique atom-map sets"):
            reaction_hash(mapped_reaction)

    def test_reaction_rejects_missing_maps(self):
        reactant = yarpecule("[C:0][O:1]", canon=False)
        product = yarpecule("[C:0][O:1]", canon=False)
        mapped_reaction = reaction(reactant, product)

        # A restored object can bypass yarpecule construction checks.
        for graph in (mapped_reaction.reactant.graph, mapped_reaction.product.graph):
            for info in graph._atom_info.values():
                info["atom_map"] = None

        with pytest.raises(ValueError, match="identical unique atom-map sets"):
            mapped_reaction._validate_reaction()

    def test_atom_map_values_are_arbitrary_correspondence_labels(self):
        first = reaction(
            yarpecule("[C:0]([H:1])([H:2])([H:3])[H:4]", canon=False),
            yarpecule("[C:0]([H:1])([H:2])([H:3])[H:4]", canon=False),
        )
        relabeled = reaction(
            yarpecule(
                "[C:100]([H:101])([H:102])([H:103])[H:104]",
                canon=False,
            ),
            yarpecule(
                "[C:100]([H:101])([H:102])([H:103])[H:104]",
                canon=False,
            ),
        )

        assert first.hash == relabeled.hash

    def test_mapping_equivalence(self):
        """
        Test that chemically distinct mappings remain different while
        symmetry-equivalent mappings receive the same reaction hash.
        """

        # H2 elimination from H's attached to 2 C atoms
        r1 = yarpecule('[C:0]([C:1]([H:6])([H:7])[H:8])([O:2][H:3])([H:4])[H:5]', canon=False)
        p1 = yarpecule('[C:0](=[C:1]([H:6])[H:7])([O:2][H:3])[H:4].[H:5][H:8]', canon=False)
        rxn1 = reaction(r1, p1)

        # H2 elimination but now one H comes from O atom and the other C-H replaces O-H
        r2 = yarpecule('[C:0]([C:1]([H:6])([H:7])[H:8])([O:2][H:3])([H:4])[H:5]', canon=False)
        p2 = yarpecule('[C:0](=[C:1]([H:6])[H:7])([O:2][H:8])[H:4].[H:5][H:3]', canon=False)
        rxn2 = reaction(r2, p2)

        assert rxn1.id == rxn2.id
        assert rxn1.hash != rxn2.hash

        r3 = yarpecule('[C:0]([C:1](=[O:2])[H:3])([H:4])([H:5])[H:6]', canon=False)
        p3 = yarpecule('[C:0](=[C:1]([O:2][H:4])[H:3])([H:5])[H:6]', canon=False)
        rxn3 = reaction(r3, p3)

        r4 = yarpecule('[C:0]([C:1](=[O:2])[H:3])([H:4])([H:5])[H:6]', canon=False)
        p4 = yarpecule('[C:0](=[C:1]([O:2][H:6])[H:3])([H:5])[H:4]', canon=False)
        rxn4 = reaction(r4, p4)

        assert rxn3.id == rxn4.id
        assert rxn3.hash == rxn4.hash

    def test_order_invariance(self):
        """
        Test that two reactions with identical mappings, but scrampled atom ordering
        have identical reaction hashes.
        Test reaction: H2 elimination from ethanol (CCO) to form HAA (C=CO)
        """

        # H2 elimination from H's attached to 2 C atoms
        r1 = yarpecule('[C:0]([C:1]([H:6])([H:7])[H:8])([O:2][H:3])([H:4])[H:5]', canon=False)
        p1 = yarpecule('[C:0](=[C:1]([H:6])[H:7])([O:2][H:3])[H:4].[H:5][H:8]', canon=False)
        rxn1 = reaction(r1, p1)

        # Swap indexes between OH group
        r2 = yarpecule('[C:0]([C:1]([H:6])([H:7])[H:8])([O:3][H:2])([H:4])[H:5]', canon=False)
        p2 = yarpecule('[C:0](=[C:1]([H:6])[H:7])([O:3][H:2])[H:4].[H:5][H:8]', canon=False)
        rxn2 = reaction(r2, p2)

        assert rxn1.hash == rxn2.hash

        # Swap indexes of O and non-adjacent or connected H
        r3 = yarpecule('[C:0]([C:1]([H:6])([H:7])[H:8])([O:4][H:2])([H:3])[H:5]', canon=False)
        p3 = yarpecule('[C:0](=[C:1]([H:6])[H:7])([O:4][H:2])[H:3].[H:5][H:8]', canon=False)
        rxn3 = reaction(r3, p3)

        assert rxn1.hash == rxn3.hash

        # Consistently reindex atoms involved in the same reaction
        r4 = yarpecule('[C:1]([C:0]([H:3])([H:4])[H:5])([O:2][H:6])([H:7])[H:8]', canon=False)
        p4 = yarpecule('[C:1](=[C:0]([H:3])[H:4])([O:2][H:6])[H:7].[H:8][H:5]', canon=False)
        rxn4 = reaction(r4, p4)

        assert rxn1.hash == rxn4.hash


    def test_reverse_reaction(self):
        """Forward and reverse reactions have the same hash."""
        r1 = yarpecule('[C:0]([C:1](=[O:2])[H:3])([H:4])([H:5])[H:6]', canon=False)
        p1 = yarpecule('[C:0](=[C:1]([O:2][H:4])[H:3])([H:5])[H:6]', canon=False)
        rxn1 = reaction(r1, p1)

        r2 = yarpecule('[C:0](=[C:1]([O:2][H:4])[H:3])([H:5])[H:6]', canon=False)
        p2 = yarpecule('[C:0]([C:1](=[O:2])[H:3])([H:4])([H:5])[H:6]', canon=False)
        rxn2 = reaction(r2, p2)

        assert rxn1.id != rxn2.id
        assert rxn1.hash == rxn2.hash


def remap_product(rxn, permutation):
    """Change product correspondences while keeping both endpoint graphs intact."""
    variant = deepcopy(rxn)
    anchor = variant.reactant.graph
    other = variant.product.graph
    maps = [anchor.atom_info[i]["atom_map"] for i in range(len(anchor.elements))]
    by_map = {atom_map: i for i, atom_map in enumerate(maps)}
    for i in range(len(other.elements)):
        old_map = other.atom_info[i]["atom_map"]
        other._atom_info[i]["atom_map"] = maps[permutation[by_map[old_map]]]
    return variant


def reaction_graph(rxn):
    """Independent element/mass/BEM graph oracle, without stereo information."""
    reactant = rxn.reactant.graph
    product = rxn.product.graph
    maps = [reactant.atom_info[i]["atom_map"] for i in range(len(reactant.elements))]
    product_by_map = {
        product.atom_info[i]["atom_map"]: i for i in range(len(product.elements))
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
    """Compare mapped endpoint topology independently of the numeric hash."""
    return nx.is_isomorphic(
        reaction_graph(left),
        reaction_graph(right),
        node_match=nx.algorithms.isomorphism.categorical_node_match("label", None),
        edge_match=nx.algorithms.isomorphism.categorical_edge_match("label", None),
    )


class TestReactionHashCorpus:
    """Committed cases: 200 symmetry, 100 nonisomorphic, 100 direction, 5 network reverses."""

    @pytest.mark.parametrize(
        "case_index", range(200), ids=[f"symmetry-{i:03d}" for i in range(200)]
    )
    def test_symmetry_equivalents_share_hash(self, case_index, reaction_hash_symmetry_cases):
        """Symmetry-equivalent correspondences must deduplicate."""
        original, permutation = reaction_hash_symmetry_cases[case_index]
        variant = remap_product(original, permutation)
        assert isomorphic(original, variant)
        assert reaction_hash(original) == reaction_hash(variant)

    @pytest.mark.parametrize(
        "case_index", range(100), ids=[f"nonisomorphic-{i:03d}" for i in range(100)]
    )
    def test_nonisomorphic_correspondences_have_distinct_hashes(
        self, case_index, reaction_hash_nonisomorphic_cases
    ):
        """Distinct mapped transformations must retain separate hashes."""
        original, left, right = reaction_hash_nonisomorphic_cases[case_index]
        permutation = list(range(len(original.reactant.graph.elements)))
        permutation[left], permutation[right] = permutation[right], permutation[left]
        variant = remap_product(original, permutation)
        assert not isomorphic(original, variant)
        assert reaction_hash(original) != reaction_hash(variant)

    @pytest.mark.parametrize(
        "case_index", range(100), ids=[f"direction-{i:03d}" for i in range(100)]
    )
    def test_forward_reverse_share_hash(self, case_index, reaction_hash_direction_cases):
        """Forward and reverse forms of one transformation share a hash."""
        forward = reaction_hash_direction_cases[case_index]
        reverse = SimpleNamespace(reactant=forward.product, product=forward.reactant)
        assert reaction_hash(forward) == reaction_hash(reverse)

    @pytest.mark.parametrize(
        "case_index", range(5), ids=[f"network-reverse-{i:02d}" for i in range(5)]
    )
    def test_dropped_network_records_are_exact_reverses(
        self, case_index, reaction_hash_network_reverse_cases
    ):
        """Five real dropped network records are reverse duplicates, not collisions."""
        first, later = reaction_hash_network_reverse_cases[case_index]
        reversed_later = SimpleNamespace(reactant=later.product, product=later.reactant)
        assert isomorphic(first, reversed_later)
        assert reaction_hash(first) == reaction_hash(later)


class TestBemSumHash:
    """
    `bem_sum_hash` is the mapping-DEPENDENT counterpart to the yarpecule hash.

    The yarpecule hash weights the summed bond-electron matrix by
    `outer(atom_hashes, atom_hashes)`; atom hashes are graph invariants, so
    relabelling the atoms leaves it unchanged. That is correct for "same
    molecule" and wrong for "same molecule, indexed the same way" -- the
    question anything exchanging index-ordered geometries has to ask.

    Measured over the KHP cycle-2 collection: 1110 products span 712 distinct
    (graph, mapping) pairs but only 268 distinct yarpecule hashes.
    """

    def test_distinguishes_atom_mappings(self, khp_remapped_products):
        """The whole point: same molecule, different indexing, different hash."""
        for smi, mols in khp_remapped_products.items():
            hashes = {m.hash for m in mols}
            assert len(hashes) == 1, (
                f"{smi} group no longer shares a yarpecule hash; the fixture is "
                "not producing remapped duplicates any more"
            )

            bem_hashes = [m.bem_sum_hash for m in mols]
            assert len(set(bem_hashes)) == len(mols), (
                f"bem_sum_hash failed to separate {len(mols)} atom mappings of "
                f"{smi}: {bem_hashes}"
            )

    def test_identical_mapping_gives_identical_hash(self, khp_products):
        """Rebuilding the same molecule the same way must reproduce the hash."""
        for smi, product in khp_products.items():
            twin = yarpecule((product.adj_mat, product.geo, product.elements,
                              product.q, product.atom_info), canon=False)
            assert twin.bem_sum_hash == product.bem_sum_hash, (
                f"bem_sum_hash was not reproducible for {smi}"
            )

    def test_independent_of_bond_mat_ordering(self):
        """
        Summing the BEMs before hashing is what makes this independent of the
        order find_lewis returns resonance structures in. The minimum Lewis
        score is frequently tied, so bond_mats[0] is a tie-break rather than a
        well-defined choice.
        """
        benzene = yarpecule('c1ccccc1')
        assert len(benzene.bond_mats) > 1, "expected resonance structures"

        scores = list(benzene.bond_mat_scores)
        assert scores[0] == scores[1], (
            "expected the minimum Lewis score to be tied for benzene"
        )

        summed = np.zeros_like(benzene.bond_mats[0])
        for mat in benzene.bond_mats:
            summed += mat
        reversed_sum = np.zeros_like(benzene.bond_mats[0])
        for mat in reversed(benzene.bond_mats):
            reversed_sum += mat

        assert np.array_equal(summed, reversed_sum)
        assert benzene.bem_sum_hash == bmat_hash(summed)

    def test_carries_no_element_information(self):
        """
        A documented limitation, pinned so it is not mistaken for a species key.
        Acetate and nitromethane share a bond topology and differ only in the
        central atom, which the bond-electron matrix does not record.
        """
        acetate = yarpecule('CC(=O)[O-]')
        nitromethane = yarpecule('C[N+](=O)[O-]')

        assert acetate.bem_sum_hash == nitromethane.bem_sum_hash
        assert acetate.hash != nitromethane.hash, (
            "the yarpecule hash must still separate these, or the composite "
            "identity used by progress_yarp is not sound"
        )
