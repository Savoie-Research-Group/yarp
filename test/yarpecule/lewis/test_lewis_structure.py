"""
Integration testing for Lewis structure generation in the Yarpecule package.
"""
import pytest 
import numpy as np
from yarp.yarpecule.yarpecule import yarpecule as ypcule

#Define test class for Lewis structure generation
class TestLewisStructureGeneration:
    #Bond Matrix Tests
    def test_pyridine_aldehyde_bond_matrix(self, pyridine_aldehyde_xyz):
        yp_mol = ypcule(pyridine_aldehyde_xyz, mode='yarp')
        b_mat = yp_mol.bond_mats
        assert b_mat[0][1] == pytest.approx(1, rel=1e-5)
        

    
    #Formal Charge Tests
    
    #Lewis Structure Score Tests
    
    #Rings and Aromaticity Tests
    
    #Electron Acceptor and Donor Tests
    
    #Atom Neighbors Tests
    
    #Multiple Molecules Tests
    
    
