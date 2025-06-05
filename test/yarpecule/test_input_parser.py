"""
Testing suite for functions contained in yarp/yarpecule/input_parser.py
"""
import pytest
from yarp.yarpecule.input_parsers import xyz_parse


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
