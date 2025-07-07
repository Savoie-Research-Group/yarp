"""
Testing suite for the yarpecule class
"""
import pytest
from yarp.yarpecule.yarpecule import yarpecule as ypcule


class TestYarpeculeFromSMILES:
    def test_ethene_init(self, ethene_smi):
        mol_yarp = ypcule(ethene_smi, mode='yarp')
        assert mol_yarp.elements == ['c', 'c', 'h', 'h', 'h', 'h']

        mol_rdkit = ypcule(ethene_smi, mode='rdkit')
        assert mol_rdkit.elements == ['c', 'c', 'h', 'h', 'h', 'h']

    def test_haa_init(self, haa_canon_smi):
        mol_yarp = ypcule(haa_canon_smi, mode='yarp')
        assert mol_yarp.elements == ['c', 'c', 'o', 'o', 'h', 'h', 'h', 'h']

        mol_rdkit = ypcule(haa_canon_smi, mode='rdkit')
        assert mol_rdkit.elements == ['c', 'c', 'o', 'o', 'h', 'h', 'h', 'h']
