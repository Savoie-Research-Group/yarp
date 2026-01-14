"""
Integration testing for Lewis structure generation in the Yarpecule package.
"""
from conftest import chloroform_xyz, ec_xyz, ester_xyz, methyl_3_butene_radical_xyz, methylmethyleneoxide_xyz, o_xylene_xyz, propene_radical_xyz
import pytest 
import numpy as np
from yarp.yarpecule.yarpecule import yarpecule as ypcule

#Define test class for Lewis structure generation
class TestLewisStructureGeneration:
    """========== Find Lewis Structures Tests for Ions =========="""
    def test_methyl_3_buteneium_xyz(self, methyl_3_buteneium_xyz):
        yp_mol = ypcule(methyl_3_buteneium_xyz, mode='yarp')
        
    def test_methyl_4_chloro_3_buteneium_xyz(self, methyl_4_chloro_3_buteneium_xyz):
        yp_mol = ypcule(methyl_4_chloro_3_buteneium_xyz, mode='yarp')

    def test_methyl_4_fluoro_3_buteneium_xyz(self, methyl_4_fluoro_3_buteneium_xyz):
        yp_mol = ypcule(methyl_4_fluoro_3_buteneium_xyz, mode='yarp')
        
    def test_propanenitrene_xyz(self, propanenitrene_xyz):
        yp_mol = ypcule(propanenitrene_xyz, mode='yarp')
        
    def test_cp_anion_xyz(self, cp_anion_xyz):
        yp_mol = ypcule(cp_anion_xyz, mode='yarp')
        
    def test_cyclopropenium_xyz(self, cyclopropenium_xyz):
        yp_mol = ypcule(cyclopropenium_xyz, mode='yarp')
        
    def test_ketonate_xyz(self, ketonate_xyz): 
        yp_mol = ypcule(ketonate_xyz, mode='yarp')

    def test_methylmethyleneoxide_xyz(self, methylmethyleneoxide_xyz):
        yp_mol = ypcule(methylmethyleneoxide_xyz, mode='yarp')

    def test_pyrylium_xyz(self, pyrylium_xyz):
        yp_mol = ypcule(pyrylium_xyz, mode='yarp')
        
    def test_toluenium_xyz(self, toluenium_xyz):
        yp_mol = ypcule(toluenium_xyz, mode='yarp')
        
    
    """=========== Find Lewis Structures Tests for Radicals ==========="""
    def test_methyl_3_butene_radical_xyz(self, methyl_3_butene_radical_xyz):
        yp_mol = ypcule(methyl_3_butene_radical_xyz, mode='yarp')
        
    def test_propene_radical_xyz(self, propene_radical_xyz):
        yp_mol = ypcule(propene_radical_xyz, mode='yarp')

        
    """=========== Find Lewis Structures Tests for Ring Structures ==========="""
    def test_adamantum_xyz(self, adamantum_xyz):
        yp_mol = ypcule(adamantum_xyz, mode='yarp')

    def test_anthracene_xyz(self, anthracene_xyz):
        yp_mol = ypcule(anthracene_xyz, mode='yarp')

    def test_azulene_xyz(self, azulene_xyz):
        yp_mol = ypcule(azulene_xyz, mode='yarp')

    def test_benzene_xyz(self, benzene_xyz):
        yp_mol = ypcule(benzene_xyz, mode='yarp')

    def test_biphenylene_xyz(self, biphenylene_xyz):
        yp_mol = ypcule(biphenylene_xyz, mode='yarp')

    def test_bis_cyclohexane_xyz(self, bis_cyclohexane_xyz):
        yp_mol = ypcule(bis_cyclohexane_xyz, mode='yarp')

    def test_napthalene_xyz(self, napthalene_xyz):
        yp_mol = ypcule(napthalene_xyz, mode='yarp')
        
    def test_o_xylene_xyz(self, o_xylene_xyz):
        yp_mol = ypcule(o_xylene_xyz, mode='yarp')
    
    def test_pyridine_aldehyde_bond_matrix(self, pyridine_aldehyde_xyz):
        yp_mol = ypcule(pyridine_aldehyde_xyz, mode='yarp')
        
    def test_thiopehene_bond_matrix(self, thiophene_xyz):
        yp_mol = ypcule(thiophene_xyz, mode='yarp')
        
    """=========== FinD Lewis Structures Tests for Zwitterions ==========="""
    def test_co_xyz(self, co_xyz):
        yp_mol = ypcule(co_xyz, mode='yarp')        
        
    def test_cyclopenteneamine_xyz(self, cyclopenteneamine_xyz):
        yp_mol = ypcule(cyclopenteneamine_xyz, mode='yarp')
        
    def test_diazomethane_xyz(self, diazomethane_xyz):
        yp_mol = ypcule(diazomethane_xyz, mode='yarp')
        
    def test_sulfoxide_xyz(self, sulfoxide_xyz):
        yp_mol = ypcule(sulfoxide_xyz, mode='yarp')
        
    def test_trimethyl_phosphine_oxide_xyz(self, trimethyl_phosphine_oxide_xyz):
        yp_mol = ypcule(trimethyl_phosphine_oxide_xyz, mode='yarp')
        
    """=========== Find Lewis Structures Tests for Neutral Structures ==========="""
    def test_chloroform_xyz(self, chloroform_xyz):
        yp_mol = ypcule(chloroform_xyz, mode='yarp')
    
    def test_ec_xyz(self, ec_xyz):
        yp_mol = ypcule(ec_xyz, mode='yarp')
    
    def test_ester_xyz(self, ester_xyz):
        yp_mol = ypcule(ester_xyz, mode='yarp')
    
    """=========== Find Lewis Structures Tests for Multi-Molecular Structures ==========="""
    def test_bimole1_far_xyz(self, bimole1_far_xyz):
        yp_mol = ypcule(bimole1_far_xyz, mode='yarp')

    def test_bimole1_one_xyz(self, bimole1_one_xyz):
        yp_mol = ypcule(bimole1_one_xyz, mode='yarp')

    def test_bimole1_two_xyz(self, bimole1_two_xyz):
        yp_mol = ypcule(bimole1_two_xyz, mode='yarp')

    def test_bimole1_xyz(self, bimole1_xyz):
        yp_mol = ypcule(bimole1_xyz, mode='yarp')
        
    """=========== Find Lewis Structures Tests for Comparing B_Mat Scores (Relative) ==========="""
    def test_relative_bmat_scores(self, benzene_xyz, propene_radical_xyz):
        benzene = ypcule(benzene_xyz, mode='yarp')
        radical = ypcule(propene_radical_xyz, mode='yarp')
        
        benzene_score = benzene.bond_mat_scores[0]
        radical_score = radical.bond_mat_scores[0]
        
        assert benzene_score < radical_score
        assert benzene_score < 0
        assert radical_score > 0