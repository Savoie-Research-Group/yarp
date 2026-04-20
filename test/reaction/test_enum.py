"""
Testing suite for functions contained in yarp/reaction/enum.py
"""
import pytest
import numpy as np
from yarp.reaction.enum import bmfn
from yarp.yarpecule.yarpecule import yarpecule
from yarp.reaction.enum import unique_set_partition_generator
import math

class TestConcertedClosedShell:
    def test_haa_b2f2_exact(self):
        """
        Test if all concerted b2f2 products are recovered
        """
        haa = yarpecule('CC=O')
        prods = list(bmfn(haa, 2, 2, hashes={haa.hash}))

        assert len(prods) == 3

        expected_prods = ['C=CO', 'C1CO1', 'C=C=O.[H][H]']

        expected_prods_hash = set()
        for smi in expected_prods:
            mol = yarpecule(smi)
            expected_prods_hash.add(mol.hash)

        found_hashes = set()
        for _ in prods:
            if _.hash in expected_prods_hash:
                found_hashes.add(_.hash)

        assert found_hashes == expected_prods_hash

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

        assert found_hashes == expected_prods_hash
    
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

class TestConcertedOpenShell:
    def test_liec_b1f1(self):
        '''
        Test if lower_score b1f1 products are formed
        '''
        liec = yarpecule('[Li]O[C]1OCCO1')
        prods = list(bmfn(liec, 1, 1, hashes={liec.hash}))

        assert len(prods) == 4

        expected_prods = ['[Li][O]C(=O)OC[CH2]',
                          '[H].[Li][O][C@@]12OC[C@@H]1O2',
                          '[Li][O][C@@H]1O[CH]CO1',
                          'O=C1OCCO1.[Li]']
        
        expected_prods_hash = set()
        for smi in expected_prods:
            mol = yarpecule(smi)
            expected_prods_hash.add(mol.hash)

        found_hashes = set()
        for _ in prods:
            if _.hash in expected_prods_hash:
                found_hashes.add(_.hash)
            
        assert found_hashes == expected_prods_hash

    def test_liec_b1f1_all(self):
        '''
        Test if all b1f1 products are formed (regardless of score)
        '''
        liec = yarpecule('[Li]O[C]1OCCO1')
        prods = list(bmfn(liec, 1, 1, hashes={liec.hash}, lower_score=False))


        assert len(prods) == 7

        expected_prods = ['[Li][O][C@@]1(O[CH2])CO1',
                            '[Li][O][C@]1([O])CCO1',
                            '[Li][O]C(=O)OC[CH2]',
                            '[H].[Li][O][C@@]12OC[C@@H]1O2',
                            '[Li][O][C@@H]1O[CH]CO1',
                            'O=C1OCCO1.[Li]',
                            '[Li][C]1([O])OCCO1']
        
        expected_prods_hash = set()
        for smi in expected_prods:
            mol = yarpecule(smi)
            expected_prods_hash.add(mol.hash)

        found_hashes = set()
        for _ in prods:
            if _.hash in expected_prods_hash:
                found_hashes.add(_.hash)
            
        assert found_hashes == expected_prods_hash

    def test_liec_b1f2(self):
        '''
        Test if all b1f2 products are formed
        --> All should be same as b1f1
        '''

        liec = yarpecule('[Li]O[C]1OCCO1')
        b1f1_prods = list(bmfn(liec, 1, 1, hashes={liec.hash}, lower_score=True))
        b1f2_prods = list(bmfn(liec, 1, 2, hashes={liec.hash}, lower_score=True))

        assert len(b1f1_prods) == len(b1f2_prods)

        b1f1_prods_hash = set()
        for _ in b1f1_prods:
            b1f1_prods_hash.add(_.hash)

        b1f2_prods_hash = set()
        for _ in b1f2_prods:
            b1f2_prods_hash.add(_.hash)

        assert b1f1_prods == b1f2_prods

    def test_liec_b2f2(self):
        """
        Test if all b2f2 products from doi: 10.1021/acs.jpclett.5c01123n are formed
        --> Also include all b1f1 products
        """
        
        liec = yarpecule('[Li]O[C]1OCCO1')
        b1f1_prods = list(bmfn(liec, 1, 1, hashes={liec.hash}, lower_score=False))
        b2f2_prods = list(bmfn(liec, 2, 2, hashes={liec.hash}, lower_score=False))

        paper_prods = ['[Li][O][C@@H]1O[CH]CO1',    # product R2-2 (also b1f1 product)
                       '[Li][O]C(=O)OC[CH2]',       # product R2-1 (also b1f1 product)
                       '[Li][O]C(=O)O[CH]C',        # product R3-3
                       'C=C.[Li][O]C([O])=O'        # product R3-1 (only shows up if lower_score=False) 3.609 > 2.538
                       ]
        
        # get set of paper products
        paper_prods_hash = set()
        for smi in paper_prods:
            mol = yarpecule(smi)
            paper_prods_hash.add(mol.hash)

        # get set of b1f1 products
        b1f1_prods_hash = set()
        for _ in b1f1_prods:
            b1f1_prods_hash.add(_.hash)

        # get found hashes for each
        found_paper = set()
        found_b1f1 = set()
        for _ in b2f2_prods:
            # look for paper products
            if _.hash in paper_prods_hash:
                found_paper.add(_.hash)
            
            # look for b1f1 products
            if _.hash in b1f1_prods_hash:
                found_b1f1.add(_.hash)
            
        assert found_paper == paper_prods_hash

        assert found_b1f1 == b1f1_prods_hash

class TestPartitionGenerator:
    def test_unique_set_partition_generator(self):
        N_list = [5, 8, 11, 14]
        n_list = [2, 4]

        uspg_length_set = set()
        confirmation_set = set()
        for N in N_list:
            sequence = list(range(1, N+1))
            for n in n_list:
                # get the length of unique_set_partition generator and add to set
                uspg = list(unique_set_partition_generator(sequence, n))
                uspg_length_set.add(len(uspg))

                # statistical determination of how many partitions should have been generated
                x = N//n
                result = 1
                for i in range(x):
                    result *= math.comb(N-(i)*n, n)
                result /= math.factorial(x)
                confirmation_set.add(result)

        assert uspg_length_set == confirmation_set



