"""
Testing suite for functions contained in yarp/yarpecule/graph/fragment.py
"""
import numpy as np
from yarp.yarpecule.graph.smiles import smiles2adjmat
from yarp.yarpecule.graph.fragment import return_rings


class TestReturnRings:
    def test_fused_rings(self, ring_cases):
        """
        return_rings finds every ring, and every ring it returns closes back on its first atom.
        """
        for smiles, sizes in ring_cases:
            adjmat, bemat, atom_info = smiles2adjmat(smiles)
            adj_list = [set(np.nonzero(row)[0]) for row in adjmat]
            rings = return_rings(adj_list, max_size=10, remove_fused=False)
            assert sorted(len(ring) for ring in rings) == sizes, smiles
            for ring in rings:
                assert all(ring[k - 1] in adj_list[ring[k]] for k in range(len(ring))), smiles
