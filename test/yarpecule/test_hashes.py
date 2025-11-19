"""
Testing suite for functions contained in yarp/yarpecule/hashes.py
"""
import pytest
import numpy as np
from yarp.yarpecule.hashes import bmat_hash
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
    def test_distinguish_mapping(self):
        """
        Test that two reactions with same reactant/product connectivities,
        but different atom mappings can be distinguished via reaction hash.
        Test reaction: H2 elimination from ethanol (CCO) to form HAA (C=CO)
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
        assert rxn3.hash != rxn4.hash

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

        # Swap indexes involved in the reaction
        r4 = yarpecule('[C:1]([C:0]([H:3])([H:4])[H:5])([O:2][H:6])([H:7])[H:8]', canon=False)
        p4 = yarpecule('[C:1](=[C:0]([H:3])[H:4])([O:2][H:6])[H:7].[H:8][H:5]', canon=False)
        rxn4 = reaction(r4, p4)

        assert rxn1.hash != rxn4.hash # TODO: figure out if this *should* be equivalent or not
