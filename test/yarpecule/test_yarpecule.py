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

class TestSMILESFromYarpecule:
    def test_haa_mapping(self, haa_full_map_smi):
        mol = ypcule(haa_full_map_smi, canon=False)
        assert mol.canon_smi == "O=CCO"
        assert mol.map_smi == "[C:0]([C:1]([O:3][H:4])([H:6])[H:7])(=[O:2])[H:5]"

    def test_aro_mapping(self, aromatic_full_map_smi):
        mol = ypcule(aromatic_full_map_smi, canon=False)
        assert mol.canon_smi == 'Oc1ccoc1'
        assert mol.map_smi == '[c:0]1([H:7])[c:2]([O:5][H:6])[c:3]([H:9])[c:1]([H:8])[o:4]1'

    def test_rad_mapping(self, rad_full_map_smi):
        mol = ypcule(rad_full_map_smi, canon=False)
        assert mol.canon_smi == '[CH2]C'
        assert mol.map_smi == '[C:0]([C:1]([H:2])[H:6])([H:3])([H:4])[H:5]' # revist if this is the correct output - ERM