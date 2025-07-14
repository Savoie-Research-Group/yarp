"""
Testing suite for functions contained in yarp/yarpecule/hashes.py
"""
import pytest
import numpy as np
from yarp.yarpecule.hashes import atom_hash, bmat_hash, yarpecule_hash
from yarp.yarpecule.yarpecule import yarpecule

class TestChargeStates:
    def test_benzene(self, benzene_smi, benz_rad_cat_smi):
        benz = yarpecule(benzene_smi, mode='yarp', canon=False)
        benz_cat = yarpecule(benz_rad_cat_smi, mode='yarp', canon=False)

        # Basic sanity checks
        assert benz.elements == benz_cat.elements
        assert np.array_equal(benz.adj_mat, benz_cat.adj_mat)

        # Check atom hashes are equivalent
        assert np.array_equal(benz.atom_hashes, benz_cat.atom_hashes)

        # Check bond-electron matrix hashes are distinct
        assert not np.array_equal(benz.bond_mats, benz_cat.bond_mats)

        assert bmat_hash(benz.bond_mats[0]) == pytest.approx(1347.297557511056, rel=1e-8)
        assert bmat_hash(benz_cat.bond_mats[0]) == pytest.approx(1346.0362830816723, rel=1e-8)

        # Check yarpecule hashes are distinct
        assert benz.hash == 1998699.92691621
        assert benz_cat.hash == 1907809.53893439

class TestAtomMappings:
    def test_haa(self, haa_canon_smi, haa_full_map_smi):
        haa_canon = yarpecule(haa_canon_smi, mode='yarp', canon=False)
        haa_map = yarpecule(haa_full_map_smi, mode='yarp', canon=False)

        # Basic sanity checks
        assert len(haa_canon.elements) == len(haa_map.elements)
        assert haa_canon.elements != haa_map.elements
        assert not np.array_equal(haa_canon.adj_mat, haa_map.adj_mat)

        # Check atom hashes are distinct
        assert not np.array_equal(haa_canon.atom_hashes, haa_map.atom_hashes)

        # Check bond-electron matrix hashes are distinct
        assert not np.array_equal(haa_canon.bond_mats, haa_map.bond_mats)

        assert bmat_hash(haa_canon.bond_mats[0]) == pytest.approx(460.7013728234123, rel=1e-8)
        assert bmat_hash(haa_map.bond_mats[0]) == pytest.approx(503.01282466196966, rel=1e-8)

        # Check yarpecule hashes are equivalent
        assert haa_canon.hash == 992960.0629294
        assert haa_canon.hash == haa_map.hash



