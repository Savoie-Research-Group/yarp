"""
Testing suite for functions contained in yarp/yarpecule/graph/smiles.py
"""
import pytest
from yarp.yarpecule.graph.smiles import smiles2adjmat
import importlib

import numpy as np

from yarp.yarpecule.graph.fragment import return_rings

smiles_mod = importlib.import_module("yarp.yarpecule.graph.smiles")


class TestSmi2Adj:
    def test_ethene(self, ethene_smi):
        adjmat, bemat, atom_info = smiles2adjmat(ethene_smi)

        elements = [atom_info[i]["element"].capitalize() for i in atom_info]
        assert elements == ['C', 'C', 'H', 'H', 'H', 'H']

        assert adjmat.shape == (6, 6)
        assert bemat.shape == (6, 6)

        assert adjmat[0, 0] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[0, 1] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[0, 2] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[0, 3] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[0, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[0, 5] == pytest.approx(0.0, rel=1e-5)

        assert adjmat[1, 0] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[1, 1] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[1, 2] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[1, 3] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[1, 4] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[1, 5] == pytest.approx(1.0, rel=1e-5)

        assert adjmat[2, 0] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[2, 1] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[2, 2] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[2, 3] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[2, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[2, 5] == pytest.approx(0.0, rel=1e-5)

        assert adjmat[3, 0] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[3, 1] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[3, 2] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[3, 3] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[3, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[3, 5] == pytest.approx(0.0, rel=1e-5)

        assert adjmat[4, 0] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[4, 1] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[4, 2] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[4, 3] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[4, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[4, 5] == pytest.approx(0.0, rel=1e-5)

        assert adjmat[5, 0] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[5, 1] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[5, 2] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[5, 3] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[5, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[5, 5] == pytest.approx(0.0, rel=1e-5)

    def test_haa_canon(self, haa_canon_smi):
        adjmat, bemat, atom_info = smiles2adjmat(haa_canon_smi)

        elements = [atom_info[i]["element"].capitalize() for i in atom_info]
        assert elements == ['O', 'C', 'C', 'O', 'H', 'H', 'H', 'H']

        assert adjmat.shape == (8, 8)
        assert bemat.shape == (8, 8)

        assert adjmat[0, 0] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[0, 1] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[0, 2] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[0, 3] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[0, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[0, 5] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[0, 6] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[0, 7] == pytest.approx(0.0, rel=1e-5)

        assert adjmat[1, 0] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[1, 1] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[1, 2] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[1, 3] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[1, 4] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[1, 5] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[1, 6] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[1, 7] == pytest.approx(0.0, rel=1e-5)

        assert adjmat[2, 0] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[2, 1] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[2, 2] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[2, 3] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[2, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[2, 5] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[2, 6] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[2, 7] == pytest.approx(0.0, rel=1e-5)

        assert adjmat[3, 0] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[3, 1] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[3, 2] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[3, 3] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[3, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[3, 5] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[3, 6] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[3, 7] == pytest.approx(1.0, rel=1e-5)

        assert adjmat[4, 0] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[4, 1] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[4, 2] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[4, 3] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[4, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[4, 5] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[4, 6] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[4, 7] == pytest.approx(0.0, rel=1e-5)

        assert adjmat[5, 0] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[5, 1] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[5, 2] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[5, 3] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[5, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[5, 5] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[5, 6] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[5, 7] == pytest.approx(0.0, rel=1e-5)

        assert adjmat[6, 0] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[6, 1] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[6, 2] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[6, 3] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[6, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[6, 5] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[6, 6] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[6, 7] == pytest.approx(0.0, rel=1e-5)

        assert adjmat[7, 0] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[7, 1] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[7, 2] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[7, 3] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[7, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[7, 5] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[7, 6] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[7, 7] == pytest.approx(0.0, rel=1e-5)

    def test_haa_full_map(self, haa_full_map_smi):
        adjmat, bemat, atom_info = smiles2adjmat(haa_full_map_smi)

        elements = [atom_info[i]["element"].capitalize() for i in atom_info]
        assert elements == ['C', 'C', 'O', 'O', 'H', 'H', 'H', 'H']

        # TO-DO: test adjmat outputs in detail
        assert adjmat.shape == (8, 8)

    def test_haa_heavy_map(self, haa_heavy_map_smi):
        adjmat, bemat, atom_info = smiles2adjmat(haa_heavy_map_smi)

        elements = [atom_info[i]["element"].capitalize() for i in atom_info]
        assert elements == ['C', 'O', 'O', 'C', 'H', 'H', 'H', 'H']

        assert adjmat.shape == (8, 8)
        assert bemat.shape == (8, 8)

        assert adjmat[0, 0] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[0, 1] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[0, 2] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[0, 3] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[0, 4] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[0, 5] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[0, 6] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[0, 7] == pytest.approx(0.0, rel=1e-5)

        assert adjmat[1, 0] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[1, 1] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[1, 2] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[1, 3] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[1, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[1, 5] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[1, 6] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[1, 7] == pytest.approx(0.0, rel=1e-5)

        assert adjmat[2, 0] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[2, 1] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[2, 2] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[2, 3] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[2, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[2, 5] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[2, 6] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[2, 7] == pytest.approx(1.0, rel=1e-5)

        assert adjmat[3, 0] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[3, 1] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[3, 2] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[3, 3] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[3, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[3, 5] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[3, 6] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[3, 7] == pytest.approx(0.0, rel=1e-5)

        assert adjmat[4, 0] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[4, 1] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[4, 2] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[4, 3] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[4, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[4, 5] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[4, 6] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[4, 7] == pytest.approx(0.0, rel=1e-5)

        assert adjmat[5, 0] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[5, 1] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[5, 2] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[5, 3] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[5, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[5, 5] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[5, 6] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[5, 7] == pytest.approx(0.0, rel=1e-5)

        assert adjmat[6, 0] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[6, 1] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[6, 2] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[6, 3] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[6, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[6, 5] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[6, 6] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[6, 7] == pytest.approx(0.0, rel=1e-5)

        assert adjmat[7, 0] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[7, 1] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[7, 2] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[7, 3] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[7, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[7, 5] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[7, 6] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[7, 7] == pytest.approx(0.0, rel=1e-5)

    def test_haa_heavy_map_explicitH(self, haa_heavy_map_explicitH_smi):
        adjmat, bemat, atom_info = smiles2adjmat(haa_heavy_map_explicitH_smi)

        elements = [atom_info[i]["element"].capitalize() for i in atom_info]
        assert elements == ['C', 'O', 'O', 'C', 'H', 'H', 'H', 'H']

        assert adjmat.shape == (8, 8)
        assert bemat.shape == (8, 8)

        assert adjmat[0, 0] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[0, 1] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[0, 2] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[0, 3] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[0, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[0, 5] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[0, 6] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[0, 7] == pytest.approx(1.0, rel=1e-5)

        assert adjmat[1, 0] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[1, 1] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[1, 2] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[1, 3] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[1, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[1, 5] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[1, 6] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[1, 7] == pytest.approx(0.0, rel=1e-5)

        assert adjmat[2, 0] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[2, 1] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[2, 2] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[2, 3] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[2, 4] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[2, 5] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[2, 6] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[2, 7] == pytest.approx(0.0, rel=1e-5)

        assert adjmat[3, 0] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[3, 1] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[3, 2] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[3, 3] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[3, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[3, 5] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[3, 6] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[3, 7] == pytest.approx(0.0, rel=1e-5)

        assert adjmat[4, 0] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[4, 1] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[4, 2] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[4, 3] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[4, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[4, 5] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[4, 6] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[4, 7] == pytest.approx(0.0, rel=1e-5)

        assert adjmat[5, 0] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[5, 1] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[5, 2] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[5, 3] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[5, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[5, 5] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[5, 6] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[5, 7] == pytest.approx(0.0, rel=1e-5)

        assert adjmat[6, 0] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[6, 1] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[6, 2] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[6, 3] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[6, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[6, 5] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[6, 6] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[6, 7] == pytest.approx(0.0, rel=1e-5)

        assert adjmat[7, 0] == pytest.approx(1.0, rel=1e-5)
        assert adjmat[7, 1] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[7, 2] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[7, 3] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[7, 4] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[7, 5] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[7, 6] == pytest.approx(0.0, rel=1e-5)
        assert adjmat[7, 7] == pytest.approx(0.0, rel=1e-5)

    def test_rad_canon(self, rad_canon_smi):
        adjmat, bemat, atom_info = smiles2adjmat(rad_canon_smi)

        elements = [atom_info[i]["element"].capitalize() for i in atom_info]
        assert elements == ['C', 'C', 'H', 'H', 'H', 'H', 'H']

        assert adjmat.shape == (7, 7)
        assert bemat.shape == (7, 7)
        # To-do: add explicit adjmat element checks

    def test_rad_canon_map(self, rad_canon_map_smi):
        adjmat, bemat, atom_info = smiles2adjmat(rad_canon_map_smi)

        elements = [atom_info[i]["element"].capitalize() for i in atom_info]
        assert elements == ['C', 'C', 'H', 'H', 'H', 'H', 'H']

        assert adjmat.shape == (7, 7)
        assert bemat.shape == (7, 7)
        # To-do: add explicit adjmat element checks

    def test_rad_explicitH(self, rad_explicitH_smi):
        adjmat, bemat, atom_info = smiles2adjmat(rad_explicitH_smi)

        elements = [atom_info[i]["element"].capitalize() for i in atom_info]
        assert elements == ['C', 'C', 'H', 'H', 'H', 'H', 'H']

        assert adjmat.shape == (7, 7)
        assert bemat.shape == (7, 7)
        # To-do: add explicit adjmat element checks

    def test_rad_full_map(self, rad_full_map_smi):
        adjmat, bemat, atom_info = smiles2adjmat(rad_full_map_smi)

        elements = [atom_info[i]["element"].capitalize() for i in atom_info]
        assert elements == ['C', 'C', 'H', 'H', 'H', 'H', 'H']

        assert adjmat.shape == (7, 7)
        assert bemat.shape == (7, 7)
        # To-do: add explicit adjmat element checks

    def test_anion_canon_smi(self, anion_canon_smi):
        adjmat, bemat, atom_info = smiles2adjmat(anion_canon_smi)

        elements = [atom_info[i]["element"].capitalize() for i in atom_info]
        assert elements == ['C', 'C', 'O', 'O', 'H', 'H', 'H']

        assert adjmat.shape == (7, 7)
        assert bemat.shape == (7, 7)
        # To-do: add explicit adjmat element checks

    def test_anion_canon_map(self, anion_canon_map_smi):
        adjmat, bemat, atom_info = smiles2adjmat(anion_canon_map_smi)

        elements = [atom_info[i]["element"].capitalize() for i in atom_info]
        assert elements == ['O', 'C', 'C', 'O', 'H', 'H', 'H']

        assert adjmat.shape == (7, 7)
        assert bemat.shape == (7, 7)
        # To-do: add explicit adjmat element checks

    def test_anion_explicitH(self, anion_explicitH_smi):
        adjmat, bemat, atom_info = smiles2adjmat(anion_explicitH_smi)

        elements = [atom_info[i]["element"].capitalize() for i in atom_info]
        assert elements == ['O', 'C', 'O', 'C', 'H', 'H', 'H']

        assert adjmat.shape == (7, 7)
        assert bemat.shape == (7, 7)
        # To-do: add explicit adjmat element checks

    def test_anion_full_map(self, anion_full_map_smi):
        adjmat, bemat, atom_info = smiles2adjmat(anion_full_map_smi)

        elements = [atom_info[i]["element"].capitalize() for i in atom_info]
        assert elements == ['O', 'C', 'O', 'C', 'H', 'H', 'H']

        assert adjmat.shape == (7, 7)
        assert bemat.shape == (7, 7)
        # To-do: add explicit adjmat element checks

    def test_aromatic_canon(self, aromatic_canon_smi):
        adjmat, bemat, atom_info = smiles2adjmat(aromatic_canon_smi)

        elements = [atom_info[i]["element"].capitalize() for i in atom_info]
        assert elements == ['O', 'C', 'C', 'C', 'O', 'C', 'H', 'H', 'H', 'H']

        assert adjmat.shape == (10, 10)
        assert bemat.shape == (10, 10)
        # To-do: add explicit adjmat element checks

    def test_aromatic_full_map(self, aromatic_full_map_smi):
        adjmat, bemat, atom_info = smiles2adjmat(aromatic_full_map_smi)

        elements = [atom_info[i]["element"].capitalize() for i in atom_info]
        assert elements == ['C', 'C', 'C', 'C', 'O', 'O', 'H', 'H', 'H', 'H']

        assert adjmat.shape == (10, 10)
        assert bemat.shape == (10, 10)
        # To-do: add explicit adjmat element checks

    def test_aromatic_lactam(self):
        adjmat, bemat, atom_info = smiles2adjmat("Oc1ncc(c(=O)[nH]1)C")

        elements = [atom_info[i]["element"].capitalize() for i in atom_info]
        assert elements == ['O', 'C', 'N', 'C', 'C', 'C', 'O', 'N', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
        assert adjmat.shape == (15, 15)
        assert bemat.shape == (15, 15)


class TestSmi2AdjRegressions:
    def test_regressions(self, smiles2adjmat_cases, quiet, formula, failures):
        """
        Branch bookkeeping (a `)` followed by `(` left close_flag set), element tokens and
        bracket atoms: the parsed elements, the named bond orders, and no radicals on heavy atoms.
        """
        for smiles, expected_formula, expected_orders in smiles2adjmat_cases:
            with failures.case(smiles):
                (_, bemat, atom_info), _ = quiet(smiles2adjmat, smiles)
                idx = {atom_info[i]["atom_map"]: i for i in atom_info}

                assert formula(atom_info[i]["element"] for i in atom_info) == expected_formula
                for (map_i, map_j), order in expected_orders.items():
                    assert bemat[idx[map_i], idx[map_j]] == order, f"bond {map_i}-{map_j}"
                radicals = [i for i in atom_info if atom_info[i]["element"] != "h" and int(bemat[i, i]) % 2]
                assert radicals == [], "heavy-atom radicals"
        failures.report()


class TestNeedsAromaticDoubleBond:
    def test_lone_pair_donors(self, aromatic_double_bond_cases, monkeypatch, quiet, failures):
        """
        The decision for every aromatic atom of the parse, recorded where smiles2adjmat calls
        the function: lone-pair donors and atoms with an exocyclic double bond say False.
        """
        for smiles, expected in aromatic_double_bond_cases:
            with failures.case(smiles):
                real = smiles_mod.needs_aromatic_double_bond
                decisions = {}

                def spy(atom, *args):
                    decisions[atom] = real(atom, *args)
                    return decisions[atom]

                monkeypatch.setattr(smiles_mod, "needs_aromatic_double_bond", spy)
                quiet(smiles2adjmat, smiles, reorder_mapped=False)
                monkeypatch.undo()
                assert [decisions[i] for i in sorted(decisions)] == expected
        failures.report()


class TestReturnRings:
    def test_fused_rings(self, ring_cases, quiet, failures):
        """
        return_rings (yarp/yarpecule/graph/fragment.py; tested here, where its bug showed):
        every ring is found, and every returned path closes back to its first atom.
        """
        for smiles, expected_sizes in ring_cases:
            with failures.case(smiles):
                (adj_mat, _, _), _ = quiet(smiles2adjmat, smiles)
                adj_list = [set(np.nonzero(row)[0]) for row in adj_mat]

                rings = return_rings(adj_list, max_size=10, remove_fused=False)

                assert sorted(len(_) for _ in rings) == expected_sizes
                for ring in rings:
                    assert all(ring[(k + 1) % len(ring)] in adj_list[ring[k]] for k in range(len(ring))), \
                        f"ring {ring} does not close"
        failures.report()
