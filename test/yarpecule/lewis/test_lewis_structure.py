"""
Integration testing for Lewis structure generation in the Yarpecule package.
Assumption: that the 20JAN (tag v0.1.0) version of the Find Lewis code in YARP is correct.
"""
import pytest 
import numpy as np
from yarp.yarpecule.yarpecule import yarpecule as ypcule

#Define a test function to apply to all test cases
def assert_invariants(yp_mol, tol=1e-4):
    """
    Test basic invariants of find-lewis outputs. Also tests that all attribute generate as not None.

    Assumes:
    - yp.bond_mats is a list of NxN float arrays
    - yp.atom_neighbors is a list of sets 
    - yp.fc formal charge on each atom, as integers
    - yp.q is total charge of the molecule as an integer
    """
    
    # --- attribute presence / non-null ---
    required_attrs = [
        "elements",
        "bond_mats",
        "atom_neighbors",
        "fc",
        "rings",
        "bond_mat_scores",
        "n_e_accept",
        "n_e_donate",
        "q"
    ]

    for attr in required_attrs:
        assert hasattr(yp_mol, attr), f"missing attribute yp_mol.{attr}"
        val = getattr(yp_mol, attr)
        assert val is not None, f"yp_mol.{attr} is None"

    # --- basic sizes ---
    n = len(yp_mol.elements)
    assert n > 0, "empty elements"
    assert len(yp_mol.fc) == n, "len(fc) != len(elements)"
    assert len(yp_mol.atom_neighbors) == n, "len(atom_neighbors) != len(elements)"

    # --- bond matrices (No Nan, square, symmetric) ---
    assert isinstance(yp_mol.bond_mats, (list, tuple)) and len(yp_mol.bond_mats) > 0, "bond_mats empty/non-list"
    for i, bm in enumerate(yp_mol.bond_mats):
        B = np.asarray(bm)
        assert B.ndim == 2 and B.shape[0] == B.shape[1], f"bond_mats[{i}] not square: {B.shape}"
        assert B.shape[0] == n, f"bond_mats[{i}] size {B.shape[0]} != n_atoms {n}"
        assert np.isfinite(B).all(), f"bond_mats[{i}] has NaN/Inf"
        assert np.allclose(B, B.T, atol=tol), f"bond_mats[{i}] not symmetric"
        assert (B >= -tol).all(), f"bond_mats[{i}] has negative entries"

    # --- neighbors consistency ---
    for i, neigh in enumerate(yp_mol.atom_neighbors):
        assert isinstance(neigh, (set, frozenset)), f"atom_neighbors[{i}] not a set"
        assert all(0 <= j < n for j in neigh), f"atom_neighbors[{i}] out-of-range index"

    # --- charge consistency ---
    assert hasattr(yp_mol, "q"), "missing total charge yp.q"
    total_fc = float(np.sum(np.asarray(yp_mol.fc)))
    assert int(round(total_fc)) == int(yp_mol.q), f"sum(fc)={total_fc} != q={yp_mol.q}"


#Define test class for Lewis structure generation
class TestLewisStructureGeneration:
    """========== Find Lewis Structures Tests for Ions =========="""
    def test_methyl_3_buteneium_xyz(self, methyl_3_buteneium_xyz):
        yp_mol = ypcule(methyl_3_buteneium_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.fc[2] == 1.0
        assert yp_mol.fc[3] == 0.0
        assert yp_mol.fc[13] == 0.0
        assert yp_mol.n_e_accept[3] == 2.0
        assert yp_mol.n_e_accept[4] == 2.0
        assert yp_mol.n_e_accept[5] == 0.0
        assert yp_mol.n_e_donate[3] == 2.0
        assert yp_mol.n_e_donate[4] == 2.0
        assert yp_mol.n_e_donate[5] == 0.0
        
        
    def test_methyl_4_chloro_3_buteneium_xyz(self, methyl_4_chloro_3_buteneium_xyz):
        yp_mol = ypcule(methyl_4_chloro_3_buteneium_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.n_e_donate[5] == 2.0
        assert yp_mol.n_e_donate[6] == 6.0
        assert yp_mol.n_e_donate[7] == 0.0
        assert yp_mol.atom_neighbors[2] == {2, 4, 14, 15, 16}
        assert yp_mol.atom_neighbors[3] == {0, 3, 5, 6}
        assert yp_mol.atom_neighbors[4] == {1, 2, 4, 5}
        
    def test_methyl_4_fluoro_3_buteneium_xyz(self, methyl_4_fluoro_3_buteneium_xyz):
        yp_mol = ypcule(methyl_4_fluoro_3_buteneium_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.atom_neighbors[1] == {1, 2, 11, 12, 13}
        assert yp_mol.atom_neighbors[8] == {8, 0}
        assert yp_mol.atom_neighbors[13] == {1, 13}
        assert yp_mol.bond_mats[0][0][10] == 1.0
        assert yp_mol.bond_mats[0][3][5] == 2.0
        assert yp_mol.bond_mats[0][13][12] == 0.0

    def test_propanenitrene_xyz(self, propanenitrene_xyz):
        yp_mol = ypcule(propanenitrene_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.fc[3] == -1.0
        assert yp_mol.fc[4] == 0.0
        assert yp_mol.fc[9] == 0.0
        assert yp_mol.bond_mat_scores[0] < 0.0
        
    def test_cp_anion_xyz(self, cp_anion_xyz):
        yp_mol = ypcule(cp_anion_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.fc[0] == -1.0
        assert yp_mol.fc[1] == 0.0
        assert yp_mol.fc[9] == 0.0
        assert yp_mol.bond_mat_scores[0] ==  yp_mol.bond_mat_scores[4]
        
    def test_cyclopropenium_xyz(self, cyclopropenium_xyz):
        yp_mol = ypcule(cyclopropenium_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert len(yp_mol.bond_mats) == 3
        assert yp_mol.bond_mat_scores[0] < 0.0
        assert yp_mol.bond_mat_scores[1] == yp_mol.bond_mat_scores[2]
        assert yp_mol.fc[0] == 1.0
        assert yp_mol.fc[1] == 0.0
        assert yp_mol.fc[5] == 0.0
        
    def test_ketonate_xyz(self, ketonate_xyz): 
        yp_mol = ypcule(ketonate_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.fc[3] == -1.0
        assert yp_mol.fc[4] == 0.0
        assert yp_mol.atom_neighbors[0] == {0, 1, 4, 5, 6}
        assert yp_mol.atom_neighbors[1] == {0, 1, 2, 3}
        assert yp_mol.atom_neighbors[2] == {8, 1, 2, 7}
        
    def test_methylmethyleneoxide_xyz(self, methylmethyleneoxide_xyz):
        yp_mol = ypcule(methylmethyleneoxide_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.bond_mats[0][2][2] == 2.0
        assert yp_mol.n_e_accept[1] == 2.0
        assert yp_mol.n_e_accept[2] == 2.0
        assert yp_mol.n_e_donate[1] == 2.0
        assert yp_mol.n_e_donate[2] == 4.0
        assert yp_mol.bond_mat_scores[0] > 0.0
        assert yp_mol.fc[2] == 1.0
        
    def test_pyrylium_xyz(self, pyrylium_xyz):
        yp_mol = ypcule(pyrylium_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.rings[0][0] == 0
        assert yp_mol.rings[0][1] == 2
        assert yp_mol.rings[0][2] == 4
        assert yp_mol.rings[0][3] == 3
        assert yp_mol.rings[0][4] == 1
        assert yp_mol.rings[0][5] == 5
        assert yp_mol.n_e_accept[0] == 2.0
        assert yp_mol.n_e_accept[1] == 2.0
        assert yp_mol.n_e_accept[2] == 2.0
        
    def test_toluenium_xyz(self, toluenium_xyz):
        yp_mol = ypcule(toluenium_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.fc[6] == 1.0
        assert yp_mol.fc[7] == 0.0
        assert yp_mol.bond_mat_scores[0] == yp_mol.bond_mat_scores[1]
        assert yp_mol.bond_mat_scores[0] > 0.0
        assert yp_mol.n_e_accept[6] == 2.0
        assert yp_mol.n_e_accept[7] == 0.0
    
    """=========== Find Lewis Structures Tests for Radicals ==========="""
    def test_methyl_3_butene_radical_xyz(self, methyl_3_butene_radical_xyz):
        yp_mol = ypcule(methyl_3_butene_radical_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert len(yp_mol.bond_mats) == 2
        assert yp_mol.bond_mat_scores[0] > 0.0
        assert yp_mol.bond_mat_scores[1] > 0.0
        #Heuristic for organic radicals
        assert yp_mol.bond_mat_scores[0] < 5.0
        assert yp_mol.bond_mat_scores[1] < 5.0
        assert yp_mol.fc[2] == 0.0
        assert yp_mol.fc[3] == 0.0
        assert yp_mol.fc[4] == 0.0
        assert yp_mol.fc[5] == 0.0
        assert np.trace(yp_mol.bond_mats[0]) == 1.0
        assert np.trace(yp_mol.bond_mats[1]) == 1.0
        
    def test_propene_radical_xyz(self, propene_radical_xyz):
        yp_mol = ypcule(propene_radical_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.bond_mat_scores[0] > 0.0
        assert yp_mol.bond_mat_scores[1] > 0.0
        #Heuristic for organic radicals
        assert yp_mol.bond_mat_scores[0] < 5.0
        assert yp_mol.bond_mat_scores[1] < 5.0
        assert yp_mol.atom_neighbors[0] == {0, 1, 2, 3}
        assert yp_mol.atom_neighbors[1] == {0, 1, 6, 7}
        assert yp_mol.atom_neighbors[2] == {0, 2, 4, 5}
        assert yp_mol.atom_neighbors[3] == {0, 3}
        assert yp_mol.atom_neighbors[4] == {2, 4}
        assert yp_mol.atom_neighbors[5] == {2, 5}
        assert yp_mol.atom_neighbors[6] == {1, 6}
        assert yp_mol.atom_neighbors[7] == {1, 7}
        assert np.trace(yp_mol.bond_mats[0]) == 1.0
        assert np.trace(yp_mol.bond_mats[1]) == 1.0
        
    """=========== Find Lewis Structures Tests for Ring Structures ==========="""
    def test_adamantum_xyz(self, adamantum_xyz):
        yp_mol = ypcule(adamantum_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.rings[0][0] == 0
        assert yp_mol.rings[0][1] == 3
        assert yp_mol.rings[0][2] == 2
        assert yp_mol.rings[1][0] == 0
        assert yp_mol.rings[1][1] == 3
        assert yp_mol.rings[1][2] == 2
        assert yp_mol.rings[1][3] == 1
        assert yp_mol.rings[1][4] == 7
        assert yp_mol.rings[1][5] == 6
        assert yp_mol.rings[2][0] == 0
        assert yp_mol.rings[2][1] == 4
        assert yp_mol.rings[2][2] == 5
        assert yp_mol.bond_mat_scores[0] < 0.1
        assert yp_mol.fc[19] == 0.0
        assert yp_mol.fc[20] == 0.0
        assert yp_mol.fc[21] == 0.0
        assert yp_mol.n_e_accept[19] == 0.0
        assert yp_mol.n_e_accept[20] == 0.0
        assert yp_mol.n_e_accept[21] == 0.0
        assert yp_mol.n_e_donate[9] == 0.0
        assert yp_mol.n_e_donate[10] == 0.0
        assert yp_mol.n_e_donate[11] == 0.0
        assert yp_mol.atom_neighbors[7] == {1, 6, 7, 19, 20}
        assert yp_mol.atom_neighbors[8] == {8, 0}
        assert yp_mol.atom_neighbors[9] == {9, 1}
        
    def test_anthracene_xyz(self, anthracene_xyz):
        yp_mol = ypcule(anthracene_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.rings[0][0] == 0
        assert yp_mol.rings[0][1] == 2
        assert yp_mol.rings[0][2] == 5
        assert yp_mol.rings[0][3] == 1
        assert yp_mol.rings[0][4] == 3
        assert yp_mol.rings[0][5] == 4
        assert yp_mol.rings[1][0] == 0
        assert yp_mol.rings[1][1] == 2
        assert yp_mol.rings[1][2] == 7
        assert yp_mol.rings[1][3] == 11
        assert yp_mol.rings[1][4] == 10
        assert yp_mol.rings[1][5] == 6
        assert yp_mol.rings[2][0] == 1
        assert yp_mol.rings[2][1] == 3
        assert yp_mol.rings[2][2] == 9
        assert yp_mol.rings[2][3] == 13
        assert yp_mol.rings[2][4] == 12
        assert yp_mol.rings[2][5] == 8
        assert len(yp_mol.bond_mats) == 2
        assert yp_mol.bond_mat_scores[0] == yp_mol.bond_mat_scores[1]
        assert len(yp_mol.rings) == 3
        
    def test_azulene_xyz(self, azulene_xyz):
        yp_mol = ypcule(azulene_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert len(yp_mol.bond_mats) == 1
        assert yp_mol.rings[0][0] == 0
        assert yp_mol.rings[0][1] == 1
        assert yp_mol.rings[0][2] == 3
        assert yp_mol.rings[0][3] == 6
        assert yp_mol.rings[1][2] == 5
        assert yp_mol.rings[1][3] == 7
        assert yp_mol.rings[1][4] == 9
        assert yp_mol.rings[1][5] == 8
        assert yp_mol.rings[1][6] == 4
        assert yp_mol.n_e_accept[9] == 2.0
        assert yp_mol.n_e_accept[10] == 0.0
        assert yp_mol.n_e_donate[9] == 2.0
        assert yp_mol.n_e_donate[10] == 0.0
        assert len(yp_mol.rings) == 2
        
        
    def test_benzene_xyz(self, benzene_xyz):
        yp_mol = ypcule(benzene_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.fc[8] == 0.0
        assert yp_mol.fc[9] == 0.0
        assert yp_mol.fc[10] == 0.0
        assert yp_mol.bond_mat_scores[0] == yp_mol.bond_mat_scores[1]
        assert len(yp_mol.bond_mats) == 2
        assert yp_mol.bond_mats[0][1][7] == 1.0
        assert yp_mol.bond_mats[1][5][1] == 2.0
        assert yp_mol.bond_mats[1][5][2] == 0.0
        #Confirm only 1 ring
        assert yp_mol.rings[0][0] == 0
        assert yp_mol.rings[0][1] == 1
        assert yp_mol.rings[0][2] == 5
        assert yp_mol.rings[0][3] == 4
        assert yp_mol.rings[0][4] == 3
        assert yp_mol.rings[0][5] == 2
        assert len(yp_mol.rings) == 1
        
    def test_biphenylene_xyz(self, biphenylene_xyz):
        yp_mol = ypcule(biphenylene_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.rings[1][1] == 1
        assert yp_mol.rings[1][2] == 5
        assert yp_mol.rings[1][3] == 8
        assert yp_mol.rings[2][1] == 3
        assert yp_mol.rings[2][4] == 10
        assert yp_mol.rings[2][5] == 6
        assert yp_mol.n_e_accept[0] == 2.0
        assert yp_mol.n_e_accept[1] == 2.0
        assert yp_mol.n_e_accept[19] == 0.0
        assert len(yp_mol.rings) == 3
        
        
    def test_bis_cyclohexane_xyz(self, bis_cyclohexane_xyz):
        yp_mol = ypcule(bis_cyclohexane_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.fc[16] == 0.0
        assert yp_mol.fc[17] == 0.0
        assert yp_mol.fc[18] == 0.0
        assert yp_mol.bond_mat_scores[0] < 0.5
        assert yp_mol.rings[0][0] == 0
        assert yp_mol.rings[1][0] == 0
        assert yp_mol.rings[1][3] == 10
        assert yp_mol.n_e_accept[13] == 0.0
        assert yp_mol.n_e_accept[14] == 0.0
        assert yp_mol.n_e_accept[15] == 0.0
        assert yp_mol.n_e_donate[23] == 0.0
        assert yp_mol.n_e_donate[24] == 0.0
        assert yp_mol.n_e_donate[25] == 0.0
        assert yp_mol.atom_neighbors[8] == {3, 8, 10, 25, 26}
        assert yp_mol.atom_neighbors[9] == {5, 6, 9, 27, 28}
        assert yp_mol.atom_neighbors[10] == {7, 8, 10, 29, 30}
        assert yp_mol.bond_mats[0][1][19] == 0.0
        assert yp_mol.bond_mats[0][1][20] == 0.0
        assert yp_mol.bond_mats[0][30][26] == 0.0
        assert len(yp_mol.bond_mats) == 1
        assert len(yp_mol.rings) == 2
        
    def test_napthalene_xyz(self, napthalene_xyz):
        yp_mol = ypcule(napthalene_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.fc[7] == 0.0
        assert yp_mol.fc[8] == 0.0
        assert yp_mol.fc[9] == 0.0
        assert yp_mol.bond_mat_scores[0] < 0.0
        assert len(yp_mol.rings) == 2
        
    def test_o_xylene_xyz(self, o_xylene_xyz):
        yp_mol = ypcule(o_xylene_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.fc[8] == 0.0
        assert yp_mol.fc[9] == 0.0
        assert yp_mol.fc[10] == 0.0
        assert yp_mol.bond_mat_scores[0] == yp_mol.bond_mat_scores[1]
        assert yp_mol.bond_mat_scores[0] < 0.0
        assert len(yp_mol.bond_mats) == 2
        assert len(yp_mol.rings) == 1
        
    def test_pyridine_aldehyde_bond_matrix(self, pyridine_aldehyde_xyz):
        yp_mol = ypcule(pyridine_aldehyde_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.n_e_accept[5] == 2.0
        assert yp_mol.n_e_accept[6] == 2.0
        assert yp_mol.n_e_accept[8] == 0.0
        assert yp_mol.atom_neighbors[4] == {0, 11, 4, 5}
        assert yp_mol.atom_neighbors[5] == {12, 3, 4, 5}
        assert yp_mol.atom_neighbors[6] == {2, 3, 6}
        assert len(yp_mol.rings) == 1
        
    def test_thiopehene_bond_matrix(self, thiophene_xyz):
        yp_mol = ypcule(thiophene_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.n_e_donate[3] == 2.0
        assert yp_mol.n_e_donate[4] == 4.0
        assert yp_mol.n_e_donate[5] == 0.0
        assert yp_mol.atom_neighbors[2] == {0, 2, 3, 7}
        assert yp_mol.atom_neighbors[3] == {8, 1, 2, 3}
        assert yp_mol.atom_neighbors[4] == {0, 1, 4}
        assert len(yp_mol.rings) == 1
        
    """=========== Find Lewis Structures Tests for Zwitterions ==========="""
    def test_co_xyz(self, co_xyz):
        yp_mol = ypcule(co_xyz, mode='yarp')        
        assert_invariants(yp_mol)
        assert yp_mol.fc[0] == 1.0
        assert yp_mol.fc[1] == -1.0
        assert yp_mol.bond_mat_scores[0] > 0.0
        assert yp_mol.atom_neighbors[0] == {0, 1}
        assert yp_mol.atom_neighbors[1] == {0, 1}
        assert yp_mol.n_e_donate[0] == 6.0
        assert yp_mol.n_e_donate[1] == 6.0
        assert yp_mol.n_e_accept[0] == 2.0
        assert yp_mol.n_e_accept[1] == 2.0
        
    def test_cyclopenteneamine_xyz(self, cyclopenteneamine_xyz):
        yp_mol = ypcule(cyclopenteneamine_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.n_e_accept[8] == 2.0
        assert yp_mol.n_e_accept[9] == 0.0
        assert yp_mol.n_e_accept[10] == 0.0
        assert yp_mol.bond_mats[4][5][11] == 1.0
        assert yp_mol.bond_mats[4][5][14] == 0.0
        assert len(yp_mol.bond_mats) == 5
        
    def test_diazomethane_xyz(self, diazomethane_xyz):
        yp_mol = ypcule(diazomethane_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.fc[1] == 1.0
        assert yp_mol.fc[2] == -1.0
        assert yp_mol.atom_neighbors[0] == {0, 1, 3, 4}
        assert yp_mol.atom_neighbors[1] == {0, 1, 2}
        assert len(yp_mol.bond_mats) == 1
        assert yp_mol.bond_mats[0][2][0] == 0.0
        assert yp_mol.bond_mats[0][2][1] == 2.0
        assert yp_mol.bond_mats[0][2][2] == 4.0
        
    def test_sulfoxide_xyz(self, sulfoxide_xyz):
        yp_mol = ypcule(sulfoxide_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.fc[2] == 1.0
        assert yp_mol.fc[3] == -1.0
        assert yp_mol.n_e_accept[2] == 2.0
        assert yp_mol.n_e_accept[3] == 0.0
        assert yp_mol.n_e_donate[2] == 2.0
        assert yp_mol.n_e_donate[3] == 6.0
        
    def test_trimethyl_phosphine_oxide_xyz(self, trimethyl_phosphine_oxide_xyz):
        yp_mol = ypcule(trimethyl_phosphine_oxide_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.fc[0] == 1.0
        assert yp_mol.fc[2] == 0.0
        assert yp_mol.fc[4] == -1.0
        assert yp_mol.n_e_accept[0] == 2.0
        assert yp_mol.n_e_accept[1] == 0.0
        assert yp_mol.n_e_donate[4] == 6.0
        assert yp_mol.n_e_donate[5] == 0.0

    def test_amide_smi(self):
        yp_mol = ypcule('CC(N)=O')
        assert_invariants(yp_mol)
        assert len(yp_mol.bond_mats) == 2
        assert yp_mol.fc[0] == 0.0
        assert yp_mol.fc[1] == 0.0
        assert yp_mol.fc[2] == 1.0
        assert yp_mol.fc[3] == -1.0
        assert yp_mol.n_e_accept[0] == 0.0
        assert yp_mol.n_e_accept[1] == 2.0
        assert yp_mol.n_e_accept[2] == 2.0
        assert yp_mol.n_e_accept[3] == 0.0
        assert yp_mol.n_e_donate[0] == 0.0
        assert yp_mol.n_e_donate[1] == 2.0
        assert yp_mol.n_e_donate[2] == 2.0
        assert yp_mol.n_e_donate[3] == 6.0

    """=========== Find Lewis Structures Tests for Neutral Structures ==========="""
    def test_chloroform_xyz(self, chloroform_xyz):
        yp_mol = ypcule(chloroform_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.n_e_accept[0] == 0.0
        assert yp_mol.n_e_accept[1] == 2.0
        assert yp_mol.n_e_accept[2] == 2.0
        assert yp_mol.n_e_donate[2] == 6.0
        assert yp_mol.n_e_donate[3] == 6.0
        assert yp_mol.n_e_donate[4] == 0.0
        assert yp_mol.atom_neighbors[0] == {0, 1, 2, 3, 4}
        
    def test_ec_xyz(self, ec_xyz):
        yp_mol = ypcule(ec_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.fc[2] == 0.0
        assert yp_mol.fc[3] == 0.0
        assert yp_mol.rings[0][0] == 0
        assert yp_mol.rings[0][1] == 1
        assert yp_mol.rings[0][2] == 4
        assert yp_mol.rings[0][3] == 2
        assert yp_mol.rings[0][4] == 3
        assert yp_mol.n_e_accept[2] == 2.0
        assert yp_mol.n_e_accept[3] == 0.0
        assert yp_mol.n_e_accept[4] == 0.0
        assert yp_mol.n_e_donate[3] == 4.0
        assert yp_mol.n_e_donate[4] == 4.0
        assert yp_mol.n_e_donate[5] == 6.0
        
    def test_ester_xyz(self, ester_xyz):
        yp_mol = ypcule(ester_xyz, mode='yarp')
        assert_invariants(yp_mol)
        assert yp_mol.fc[5] == 0.0
        assert yp_mol.fc[6] == 0.0
        assert yp_mol.fc[7] == 0.0
        assert yp_mol.atom_neighbors[1] == {1, 4, 8, 9, 10}
        assert yp_mol.atom_neighbors[2] == {0, 2, 11, 12, 13}
        assert yp_mol.atom_neighbors[3] == {0, 3, 4, 5}
        assert len(yp_mol.bond_mats) == 1
        
        
    """=========== Find Lewis Structures Tests for Comparing B_Mat Scores (Relative) ==========="""
    def test_relative_bmat_scores(self, benzene_xyz, propene_radical_xyz):
        benzene = ypcule(benzene_xyz, mode='yarp')
        radical = ypcule(propene_radical_xyz, mode='yarp')
        
        benzene_score = benzene.bond_mat_scores[0]
        radical_score = radical.bond_mat_scores[0]
        
        assert benzene_score < radical_score
        assert benzene_score < 0
        assert radical_score > 0
        
    """=========== Find Lewis Structures Tests for Multi-Molecular Structures ==========="""
    def test_multi_structure_xyz(self,  bimole1_far_xyz, 
                                        bimole1_one_xyz, 
                                        bimole1_two_xyz):
        both = ypcule(bimole1_far_xyz, mode='yarp')
        one = ypcule(bimole1_one_xyz, mode='yarp')
        two = ypcule(bimole1_two_xyz, mode='yarp')
        assert_invariants(one)
        assert_invariants(both)
        assert_invariants(two)

        # --- 1) Size Check ---
        assert len(one.elements)  == 9
        assert len(two.elements)  == 10
        assert len(both.elements) == len(one.elements) + len(two.elements)

        # --- 2) Element placement check ---
        assert one.elements.count("n")  == 1
        assert two.elements.count("n")  == 0
        assert both.elements.count("n") == one.elements.count("n") + two.elements.count("n")

        # --- 3) Ring count and placement ---
        assert len(one.rings)  == 1
        assert len(two.rings)  == 0
        assert len(both.rings) == len(one.rings) + len(two.rings)

        # --- 4) confirm two disconnected fragments in `both` by checking cross-block zeros ---
        # Atoms 0-9 belong to fragment "two", atoms 10-18 belong to fragment "one"
        B = both.bond_mats[0]
        assert B[0][10] == 0.0
        assert B[2][12] == 0.0
        assert B[3][11] == 0.0
        assert B[9][18] == 0.0
        assert B[10][0] == 0.0
        assert B[12][2] == 0.0

        # --- 5) confirm there ARE bonds inside each fragment (not all zeros) ---
        assert B[0][1] > 0.0
        assert B[2][3] > 0.0
        assert B[10][11] > 0.0
        assert B[11][12] > 0.0

        