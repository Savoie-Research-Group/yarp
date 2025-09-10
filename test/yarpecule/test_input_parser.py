"""
Testing suite for functions contained in yarp/yarpecule/input_parser.py
"""
import pytest
from yarp.yarpecule.input_parsers import xyz_parse
from yarp.yarpecule.input_parsers import xyz_q_parse
from yarp.yarpecule.input_parsers import mol_parse


class TestXYZParser:
    def test_single_molecule_no_types(self, ethene_xyz):
        elements, geo = xyz_parse(ethene_xyz)

        assert elements == ["C", "C", "H", "H", "H", "H"]
        assert geo.shape == (6, 3)
        assert geo[0, 0] == pytest.approx(0.01051, rel=1e-5)
        assert geo[0, 1] == pytest.approx(-0.00247, rel=1e-5)
        assert geo[0, 2] == pytest.approx(0.29143, rel=1e-5)
        assert geo[1, 0] == pytest.approx(-0.04372, rel=1e-5)
        assert geo[1, 1] == pytest.approx(-0.04563, rel=1e-5)
        assert geo[1, 2] == pytest.approx(1.61414, rel=1e-5)
        assert geo[2, 0] == pytest.approx(0.66926, rel=1e-5)
        assert geo[2, 1] == pytest.approx(-0.66369, rel=1e-5)
        assert geo[2, 2] == pytest.approx(-0.26622, rel=1e-5)
        assert geo[3, 0] == pytest.approx(-0.60221, rel=1e-5)
        assert geo[3, 1] == pytest.approx(0.69539, rel=1e-5)
        assert geo[3, 2] == pytest.approx(-0.27400, rel=1e-5)
        assert geo[4, 0] == pytest.approx(0.56900, rel=1e-5)
        assert geo[4, 1] == pytest.approx(-0.74348, rel=1e-5)
        assert geo[4, 2] == pytest.approx(2.17958, rel=1e-5)
        assert geo[5, 0] == pytest.approx(-0.70247, rel=1e-5)
        assert geo[5, 1] == pytest.approx(0.61559, rel=1e-5)
        assert geo[5, 2] == pytest.approx(2.17179, rel=1e-5)

class TestXYZQParser:
    def test_xyz_with_pos_q(self, ammonium_xyz):
        q = xyz_q_parse(ammonium_xyz)
        assert q == pytest.approx(1, rel=1e-5)
        
    def test_xyz_neg_q(self, nitrate_xyz):
        q = xyz_q_parse(nitrate_xyz)
        assert q == pytest.approx(-1, rel=1e-5)
                
    def test_xyz_no_q(self, ethene_xyz):
        q = xyz_q_parse(ethene_xyz)
        assert q == pytest.approx(0, rel=1e-5)

class TestMolParser:
    def test_single_mol_no_charge(self, ethanol_mol):
        elements, geo, adj_mat, q = mol_parse(ethanol_mol)
        
        # elements, heavy atoms only
        assert elements == ['C','C',"O"]
        
        # geometry, heavy atoms only
        assert geo.shape == (3, 3)
        assert geo[0, 0] == pytest.approx(-0.9254, rel=1e-5)
        assert geo[0, 1] == pytest.approx(0.0742, rel=1e-5)
        assert geo[0, 2] == pytest.approx(0.0328, rel=1e-5)
        assert geo[1, 0] == pytest.approx(0.5123, rel=1e-5)
        assert geo[1, 1] == pytest.approx(-0.4192, rel=1e-5)
        assert geo[1, 2] == pytest.approx(-0.0743, rel=1e-5)
        assert geo[2, 0] == pytest.approx(1.3778, rel=1e-5)

        # adjacency matrix
        assert adj_mat.shape == (3,3)
        assert adj_mat[0, 0] == 0
        assert adj_mat[0, 1] == 1
        assert adj_mat[0, 2] == 0
        assert adj_mat[1, 0] == 1
        assert adj_mat[1, 1] == 0
        assert adj_mat[1, 2] == 1
        assert adj_mat[2, 0] == 0
        assert adj_mat[2, 1] == 1
        assert adj_mat[2, 2] == 0
        
        # charge
        assert q == pytest.approx(0, rel=1e-5)