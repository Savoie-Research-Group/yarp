"""
Testing suite for the network class
"""
import pickle
from yarp.network.network import network
from yarp.yarpecule.yarpecule import yarpecule as ypcule

class TestInitialization:

    def test_single_path(self, glucose_single_path):
        crn = network(glucose_single_path)

        assert crn.n_rxns == 5

    def test_multi_path(self, glucose_multi_path):
        crn = network(glucose_multi_path)

        assert crn.n_rxns == 10

class TestTerminalSpecies:
    def test_single_path(self, glucose_single_path):
        expected_mols = ['O/C=C(/O)CO', 'O=CCO', 'C=O']
        expected_hashes = set()
        for m in expected_mols:
            y = ypcule(m)
            expected_hashes.add(y.hash)

        crn = network(glucose_single_path)
        mols = crn.get_terminal_species()

        assert len(mols) == 3
        for _ in mols:
            assert _.hash in expected_hashes