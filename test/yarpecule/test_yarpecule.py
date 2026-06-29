"""
Testing suite for the yarpecule class
"""
import pytest
import numpy as np
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
    # NOTE: Revisit if mapping indexes should start from 0 or from 1.
    # It would be easier to read in via RDKit if mappings started from 1, I think - ERM
    def test_haa_mapping(self, haa_full_map_smi):
        mol = ypcule(haa_full_map_smi, canon=False)
        mol.get_smiles()
        assert mol.canon_smi == "O=CCO"
        assert mol.map_smi == "[C:0]([C:1]([O:7][H:11])([H:13])[H:14])(=[O:6])[H:12]"

    def test_aro_mapping(self, aromatic_full_map_smi):
        mol = ypcule(aromatic_full_map_smi, canon=False)
        mol.get_smiles()
        assert mol.canon_smi == 'Oc1ccoc1'
        assert mol.map_smi == '[c:0]1([H:15])[c:4]([O:9][H:13])[c:5]([H:20])[c:1]([H:16])[o:6]1'

    def test_rad_mapping(self, rad_full_map_smi):
        mol = ypcule(rad_full_map_smi, canon=False)
        mol.get_smiles()
        assert mol.canon_smi == '[CH2]C'
        assert mol.map_smi == '[C:1]([C:2]([H:3])[H:7])([H:4])([H:5])[H:6]' # revist if this is the correct output - ERM

    def test_no_ghost_hydrogens_in_mapped_smiles(self):
        # Regression: charged atoms (here S+2) used to gain implicit "ghost"
        # hydrogens during the RDKit round trip that are absent from the
        # yarpecule bond-electron matrix. See get_smiles / yarpecule_to_rdmol.
        mol = ypcule(
            '[c:0]12[c:1]([H:7])[c:2]([H:8])[c:3]([H:9])[c:4]([H:10])[c:5]1[C-:12]([O-:11])[N:13]([H:6])[S+2:14]2',
            canon=False,
        )
        mol.get_smiles()

        assert "[SH2+2" not in mol.map_smi
        assert "[S+2:14]" in mol.map_smi
        assert "[SH2+2" not in mol.canon_smi

    def test_reactive_map_smi_display(self):
        mol = ypcule('[C:0]([O:1][H:2])([H:3])([H:4])[H:5]', canon=False)
        marked = mol.reactive_map_smi([set([0, 1])])

        assert '[C*:0]' in marked
        assert '[O*:1]' in marked
        assert '[H:2]' in marked

class TestInchiFromYarpecule:
    def test_haa(self, haa_full_map_smi):
        mol = ypcule(haa_full_map_smi)
        mol.get_inchi()
        assert mol.inchi == "WGCNASOHLSPBMP"

    def test_aro(self, aromatic_full_map_smi):
        mol = ypcule(aromatic_full_map_smi)
        mol.get_inchi()
        assert mol.inchi == "RNWIJLZQJSGBCU"

    def test_rad(self, rad_full_map_smi):
        mol = ypcule(rad_full_map_smi)
        mol.get_inchi()
        assert mol.inchi == "QUPDWYMUPZLYJZ"

class TestJoinYarpecules:
    def test_eth_h2o(self):
        ethene = ypcule('C=C')
        h2o = ypcule('O')

        joined = ethene.join(h2o)

        # Quick checks of elements and geometries
        # NOTE: Maybe more robust test should be added...
        # But it's good enough for now! - ERM
        assert joined.elements == ethene.elements + h2o.elements
        assert joined.geo.shape[0] == ethene.geo.shape[0] + h2o.geo.shape[0]

        # Check that all diagonal elements of adjacency matrix are zero
        diag = np.array(joined.adj_mat.diagonal())
        zeros = np.zeros_like(diag)
        np.testing.assert_array_equal(diag, zeros)

        # Check one set of off-diagonal adjacency matrix elements
        adj = joined.adj_mat

        assert adj[0, 1] == 1
        assert adj[0, 2] == 1
        assert adj[0, 3] == 1
        assert adj[0, 4] == 0
        assert adj[0, 5] == 0
        assert adj[0, 6] == 0
        assert adj[0, 7] == 0
        assert adj[0, 8] == 0

        assert adj[1, 2] == 0
        assert adj[1, 3] == 0
        assert adj[1, 4] == 1
        assert adj[1, 5] == 1
        assert adj[1, 6] == 0

        joined_maps = [joined._atom_info[i]["atom_map"] for i in joined._atom_info]
        assert len(joined_maps) == len(set(joined_maps))
        assert adj[1, 7] == 0
        assert adj[1, 8] == 0

        assert adj[2, 3] == 0
        assert adj[2, 4] == 0
        assert adj[2, 5] == 0
        assert adj[2, 6] == 0
        assert adj[2, 7] == 0
        assert adj[2, 8] == 0

        assert adj[3, 4] == 0
        assert adj[3, 5] == 0
        assert adj[3, 6] == 0
        assert adj[3, 7] == 0
        assert adj[3, 8] == 0

        assert adj[4, 5] == 0
        assert adj[4, 6] == 0
        assert adj[4, 7] == 0
        assert adj[4, 8] == 0

        assert adj[5, 6] == 0
        assert adj[5, 7] == 0
        assert adj[5, 8] == 0

        assert adj[6, 7] == 1
        assert adj[6, 8] == 1

        assert adj[7, 8] == 0

class TestSeparateYarpecules:
    def test_eth_h2o(self):
        eth_h2o = ypcule('C=C.O')

        sep_mols = eth_h2o.separate()

        assert sep_mols[0].elements == ['c', 'c', 'h', 'h', 'h', 'h']
        assert sep_mols[1].elements == ['o', 'h', 'h']

        assert sep_mols[0].geo.shape == (6, 3)
        assert sep_mols[1].geo.shape == (3, 3)

        assert sep_mols[0].adj_mat.shape == (6, 6)
        assert sep_mols[1].adj_mat.shape == (3, 3)
