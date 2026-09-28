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


    # def test_reverse_reaction(self):
    #     """
    #     """

    #     r1 = yarpecule('[C:0]([C:1](=[O:2])[H:3])([H:4])([H:5])[H:6]', canon=False)
    #     p1 = yarpecule('[C:0](=[C:1]([O:2][H:4])[H:3])([H:5])[H:6]', canon=False)
    #     rxn1 = reaction(r1, p1)

    #     r2 = yarpecule('[C:0](=[C:1]([O:2][H:4])[H:3])([H:5])[H:6]', canon=False)
    #     p2 = yarpecule('[C:0]([C:1](=[O:2])[H:3])([H:4])([H:5])[H:6]', canon=False)
    #     rxn2 = reaction(r2, p2)

    #     assert rxn1.id != rxn2.id
    #     assert rxn1.hash != rxn2.hash

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
                              product.q), canon=False)
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
