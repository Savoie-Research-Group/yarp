"""
Testing suite for functions contained in yarp/reaction/enum.py
"""
import pytest
import numpy as np
from yarp.reaction.enum import bmfn
from yarp.yarpecule.yarpecule import yarpecule

class TestConcertedClosedShell:
    def test_haa_b2f2_exact(self):
        """
        Test if all concerted b2f2 products are recovered
        """
        haa = yarpecule('CC=O')
        prods = list(bmfn(haa, 2, 2, hashes={haa.hash}))

        assert len(prods) == 3

        expected = set([869617.763119, 1131723.29428773, 507132.22729761])
        for _ in prods:
            assert _.hash in expected

    def test_haa_b3f3_contains_b2f2(self):
        """
        Test that b3f3 products equal b2f2 products.
        HAA doesn't have enough unique bonds to form "true" b3f3 products.
        """
        haa = yarpecule('CC=O')

        b2f2_prods = list(bmfn(haa, 2, 2, hashes={haa.hash}))
        b3f3_prods = list(bmfn(haa, 2, 2, hashes={haa.hash}))

        assert len(b2f2_prods) == len(b3f3_prods)

        b2f2_set = set()
        b3f3_set = set()
        for i in range(len(b2f2_prods)):
            b2f2_set.add(b2f2_prods[i].hash)
            b3f3_set.add(b3f3_prods[i].hash)

        assert b2f2_set == b3f3_set
    
    def test_khp_b2f2_target_products(self):
        """
        Test previously reported KHP products are found during b2f2.
        """
        khp = yarpecule('O=CCCOO')

        khp_b2f2 = list(bmfn(khp, 2, 2, hashes={khp.hash}))

        khp_b2f2_hash = set()
        for _ in khp_b2f2:
            khp_b2f2_hash.add(_.hash)

        expected_prods = ['O=CC=C.OO', '[H][H].O=C=CCOO', 'O=CCC=O.O', 'O=COO.C=C', 'O=CC(OO)C', '[H][H].O=CC=COO', 'O=CCC(O)O', 'O=COOCC']

        expected_prods_hash = set()
        for smi in expected_prods:
            mol = yarpecule(smi)
            expected_prods_hash.add(mol.hash)

        found_hashes = set()
        for _ in khp_b2f2:
            if _.hash in expected_prods_hash:
                found_hashes.add(_.hash)

        assert len(found_hashes) == len(expected_prods)
    
    def test_khp_b3f3_contains_b2f2(self):
        """
        Test b3f3 products contain b2f2 products.
        Generating b3f3 products takes about 5 seconds,
        so this is probably the largest molecule we will ever want to run this test for - ERM
        """

        khp = yarpecule('O=CCCOO')

        khp_b2f2 = list(bmfn(khp, 2, 2, hashes={khp.hash}))
        khp_b2f2_hash = set()
        for _ in khp_b2f2:
            khp_b2f2_hash.add(_.hash)

        khp_b3f3 = list(bmfn(khp, 3, 3, hashes={khp.hash}))
        khp_b3f3_hash = set()
        for _ in khp_b3f3:
            khp_b3f3_hash.add(_.hash)

        khp_overlap = khp_b2f2_hash & khp_b3f3_hash

        assert khp_overlap == khp_b2f2_hash

    def test_da_b2f2_excludes_product(self):
        """
        Test that a b3f3 product does NOT show up in b2f2.
        Generating b3f3 products for DA reactants takes about 30 seconds,
        so I don't think it should be put in the test suite - ERM
        """
        dar = yarpecule('C=CC=C.C=C') # Diels-Alder reactants
        dap = yarpecule('C1C=CCCC1') # Diels-Alder product (from b3f3)

        dar_b2f2 = list(bmfn(dar, 2, 2, hashes={dar.hash}, lower_score=True))

        dar_b2f2_hash = set()
        for _ in dar_b2f2:
            dar_b2f2_hash.add(_.hash)

        assert dap.hash not in dar_b2f2_hash

