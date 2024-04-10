import sys, itertools, timeit, os, copy                                                                                                                                                               
from openbabel import pybel
from openbabel import openbabel as ob
from collections import Counter
import numpy as np
import yarp as yp
from yarp.taffi_functions import graph_seps,table_generator,return_rings,adjmat_to_adjlist,canon_order
from yarp.properties import el_to_an,an_to_el,el_mass, el_radii
from yarp.find_lewis import find_lewis,return_formals,return_n_e_accept,return_n_e_donate,return_formals,return_connections,return_bo_dict
from yarp.hashes import atom_hash,yarpecule_hash
from yarp.input_parsers import xyz_parse,xyz_q_parse,xyz_from_smiles, mol_parse
from yarp.misc import merge_arrays, prepare_list
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import EnumerateStereoisomers, AllChem, TorsionFingerprints, rdmolops, rdDistGeom
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.ML.Cluster import Butina
from math import cos, sin
from wrappers.reaction import *
from utils import *
def return_model_rxn(rxn, depth=1):
    # This function is written by Hsuan-Hao Hsu (hsu205@purdue.edu)
    # Read in a true reaction and return a reaction class of model reaction
    elements=rxn.reactant.elements
    R_geo=rxn.reactant.geo
    P_geo=rxn.product.geo
    R_adj=rxn.reactant.adj_mat
    P_adj=rxn.product.adj_mat
    R_bond=rxn.reactant.bond_mats[0]
    P_bond=rxn.product.bond_mats[0]
    BE_change=P_bond-R_bond
    bn, fm, reactive_atoms=return_bnfm(BE_change)
    bond_change=bn+fm
    gs=graph_seps(R_adj)
    keep_idx=[]
    edge_idx=[] # the atom we need to add hydrogens
    for i in bond_change:
        if i[0] not in keep_idx: keep_idx.append(i[0])
        if i[1] not in keep_idx: keep_idx.append(i[1])
        for count_j, j in enumerate(gs[i[0]]):
            if j>0 and j<=depth and count_j not in keep_idx: keep_idx.append(count_j)
            if j>0 and j==depth and count_j not in reactive_atoms and count_j not in edge_idx: edge_idx.append(count_j)
        for count_j, j in enumerate(gs[i[1]]):
            if j>0 and j<=depth and count_j not in keep_idx: keep_idx.append(count_j)
            if j>0 and j==depth and count_j not in reactive_atoms and count_j not in edge_idx: edge_idx.append(count_j)
    # keep_idx stores the info of the atoms we want to keep
    # next step is adding hydrogens at the edge atom
    new_R_E, new_R_geo, new_P_geo=return_model_geo(elements, R_geo, R_bond, BE_change, keep_idx, edge_idx)
    xyz_write(".tmp_R.xyz", new_R_E, new_R_geo)
    reactant=yp.yarpecule(".tmp_R.xyz", canon=False)
    os.system("rm .tmp_R.xyz")
    xyz_write(".tmp_P.xyz", new_R_E, new_P_geo)
    product=yp.yarpecule(".tmp_P.xyz", canon=False)
    os.system("rm .tmp_P.xyz")
    R=reaction(reactant, product, args=rxn.args, opt=False)
    return R

def return_model_geo(elements, geo, bondmat, BE_change, keep_idx, edge_idx):
    # this function will generate the geometry for model rxn
    new_E, new_geo, new_edge, new_bondmat, numbond, new_BE_change=[], [], [], [], [], []
    for count_i, i in enumerate(elements):
        tmp=0
        if count_i in keep_idx:
            if count_i in edge_idx: new_edge.append(len(new_E))
            new_E.append(i)
            new_geo.append(geo[count_i])
            new_bondmat.append([j for count_j, j in enumerate(bondmat[count_i]) if count_j in keep_idx])
            new_BE_change.append([j for count_j, j in enumerate(BE_change[count_i]) if count_j in keep_idx])
            for count_j, j in enumerate(bondmat[count_i]):
                if count_j != count_i: tmp+=j
            numbond.append(tmp)
    for i in new_edge:
        # add hydrogen to the edge atoms
        tot_bond=0
        for count_j, j in enumerate(new_bondmat[i]):
            if count_j != i: tot_bond+=j
        num_add_hydrogen=int(numbond[i]-tot_bond)
        if num_add_hydrogen > 0:
            bond_length=el_radii[new_E[i]]+el_radii["H"]
        for j in range(num_add_hydrogen):
            new_E.append("H")
            const=2*3.1415926/num_add_hydrogen*float(j)
            new_geo.append(new_geo[i]+[bond_length*cos(const), bond_length*sin(const), 0.0])
            for count_k, k in enumerate(new_bondmat):
                if count_k != i:
                    new_bondmat[count_k].append(0)
                    new_BE_change[count_k].append(0)
                elif count_k == i:
                    new_bondmat[count_k].append(1)
                    new_BE_change[count_k].append(0)
            bond_h=[]
            change=[]
            for count_k, k in enumerate(new_bondmat[0]):
                if count_k != i:
                    bond_h.append(0)
                    change.append(0)
                else:
                    bond_h.append(1)
                    change.append(0)
            new_bondmat.append(bond_h)
            new_BE_change.append(change)
    new_bondmat=np.asarray(new_bondmat)
    new_geo=opt_geo(new_E, new_geo, new_bondmat)
    new_BE_change=np.asarray(new_BE_change)
    new_bondmat=new_bondmat+new_BE_change
    new_change_geo=opt_geo(new_E, new_geo, new_bondmat) 
    return new_E, new_geo, new_change_geo

def return_bnfm(bondmat):
    break_bond=[]
    form_bond=[]
    reactive_atoms=[]
    for i in range(len(bondmat)):
        for j in range(len(bondmat)):
            if i>j:
                if bondmat[i][j]>0: 
                    form_bond+=[(i, j)]
                    if i not in reactive_atoms: reactive_atoms.append(i)
                    if j not in reactive_atoms: reactive_atoms.append(j)
                elif bondmat[i][j]<0:
                    break_bond+=[(i, j)]
                    if i not in reactive_atoms: reactive_atoms.append(i)
                    if j not in reactive_atoms: reactive_atoms.append(j)
    return break_bond, form_bond, reactive_atoms
