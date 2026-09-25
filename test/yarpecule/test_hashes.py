"""
Testing suite for functions contained in yarp/yarpecule/hashes.py
"""
from importlib import import_module
from math import fsum

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
            product._atom_info[i]["atom_map"]: i
            for i in range(len(product.elements))
        }
        product_order = [
            product_by_map[reactant._atom_info[i]["atom_map"]]
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

    def test_element_inconsistent_maps_warn_and_continue(self, capsys):
        reactant = yarpecule("[C:0][O:1]", canon=False)
        product = yarpecule("[C:1][O:0]", canon=False)

        mapped_reaction = reaction(reactant, product)
        captured = capsys.readouterr()

        assert "WARNING: Element-inconsistent atom maps detected" in captured.out
        assert isinstance(mapped_reaction.hash, float)

    @pytest.mark.parametrize("atom_map", [None, 0])
    def test_reaction_rejects_missing_or_duplicate_maps(self, atom_map):
        reactant = yarpecule("[C:0][O:1]", canon=False)
        product = yarpecule("[C:0][O:1]", canon=False)
        mapped_reaction = reaction(reactant, product)

        # A restored object can bypass yarpecule construction checks.
        for graph in (mapped_reaction.reactant.graph, mapped_reaction.product.graph):
            for info in graph._atom_info.values():
                info["atom_map"] = atom_map

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
