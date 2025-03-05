import pytest
import numpy as np
import src.core.yarpecule as ypcule


@pytest.fixture(params=["ch2ch2_xyz", "ch2ch2_smi"])
def ethene_input(request):
    """Call this in test functions to iterate through all ethene input types."""
    return request.getfixturevalue(request.param)


class TestSmallOrganics:
    """
    Do we handle closed-shell organics properly?
    """

    def test_load_ethene(self, ethene_input):
        mol = ypcule(ethene_input)
        assert mol

    def test_elements_ethene(self, ethene_input):
        mol = ypcule(ethene_input)
        assert mol.elements == ['c', 'c', 'h', 'h', 'h', 'h']

    def test_geom_ethene(self, ethene_input):
        mol = ypcule(ethene_input)
        assert mol.geo.shape == (6, 3)

    def test_charge_ethene(self, ethene_input):
        mol = ypcule(ethene_input)
        assert mol.q == 0

    def test_adj_mat_ethene(self, ch2ch2_xyz):
        mol = ypcule(ch2ch2_xyz)
        adj = mol.adj_mat

        assert adj.shape == (6, 6)
        assert np.sum(adj) == 10.0
        # this is ERM being lazy, should add more rigorous checks

    def test_bond_mat_ethene(self, ch2ch2_xyz):
        mol = ypcule(ch2ch2_xyz)
        bond = mol.bond_mats[0]

        assert bond.shape == (6, 6)
        assert np.sum(bond) == 12.0
        # this is ERM being lazy, should add more rigorous checks


class TestResonance:
    """
    Do we handle resonating structures properly?
    """


class TestOrganometallics:
    """
    Do we handle dative bonds properly?
    """


class TestOrganicRadicals:
    """
    Do we handle open-shell organics properly?
    """
