import sys, itertools, timeit, os, copy, math
from itertools import combinations
from openbabel import pybel
from openbabel import openbabel as ob
from collections import Counter
import numpy as np
import yarp as yp
import yaml, fnmatch, pickle
import scipy
# from sklearn.preprocessing import normalize
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))
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

class MODEL(object):
    def __init__(self, reaction, depth=1, ff='mmff94'):
        self.reaction=reaction
        self.depth=depth
        self.ff=ff
        self.model=None
    def return_model_rxn(self):
        rxn=self.reaction
        depth=self.depth
        elements=rxn.reactant.elements
        R_geo=rxn.reactant.geo
        P_geo=rxn.product.geo
        R_adj=rxn.reactant.adj_mat
        P_adj=rxn.product.adj_mat
        R_bond=rxn.reactant.bond_mats[0]
        P_bond=rxn.product.bond_mats[0]
        BE_change=P_bond-R_bond
        adj_change=P_adj-R_adj
        bond_change, reactive_atoms=return_adj_change(adj_change)
        gs=graph_seps(R_adj)
        keep_idx=list(reactive_atoms)
        edge_idx=[]
        for i in bond_change:
            if i[0] not in keep_idx: keep_idx.append(i[0])
            if i[1] not in keep_idx: keep_idx.append(i[1])
            for count_j, j in enumerate(gs[i[0]]):
                if j>0 and j<=depth and count_j not in keep_idx: keep_idx.append(count_j)
                if j>0 and j==depth and reactive_atoms and count_j not in edge_idx: edge_idx.append(count_j)
            for count_j, j in enumerate(gs[i[1]]):
                if j>0 and j<=depth and count_j not in keep_idx: keep_idx.append(count_j)
                if j>0 and j==depth and reactive_atoms and count_j not in edge_idx: edge_idx.append(count_j)
        tmp_E=[]
        for i in tmp_E:
            if i != "H" and i!="h": tmp_E.append(tmp_E)
        if len(tmp_E)==len(keep_idx):
            print(f"This reaction is a model reaction with depth {depth}.")
            self.model=rxn
            return
        new_R_E, new_R_geo, new_P_geo=self.return_model_geo(elements, R_geo, R_bond, BE_change, keep_idx, edge_idx)
        if len(new_P_geo)==0 or len(new_R_geo)==0:
            print(f"This reaction is failed to optimized by {self.ff}.")
            self.model=None 
            return
        xyz_write(".tmp_R.xyz", new_R_E, new_R_geo)
        reactant=yp.yarpecule(".tmp_R.xyz", canon=False)
        os.system("rm .tmp_R.xyz")
        xyz_write(".tmp_P.xyz", new_R_E, new_P_geo)
        product=yp.yarpecule(".tmp_P.xyz", canon=False)
        os.system("rm .tmp_P.xyz")
        self.model=reaction(reactant, product, args=rxn.args, opt=True)
        return
    def return_model_geo(self, elements, geo, bondmat, BE_change, keep_idx, edge_idx):
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

        #for count_, _ in enumerate(new_E):
        #    print(f"{_} {new_geo[count_][0]} {new_geo[count_][1]} {new_geo[count_][2]}")

        for i in new_edge:
            tot_bond=0
            double_bond=0
            for count_j, j in enumerate(new_bondmat[i]):
                if count_j != i: tot_bond+=j
                if j>1: double_bond+=(j-1)
            num_add_hydrogen=numbond[i]-tot_bond
            if num_add_hydrogen!=0:
                bond_length=el_radii[new_E[i]]+el_radii["H"]
                connect_ids=[count_k for count_k, k in enumerate(new_bondmat[i]) if count_k!=i and k>=1]
                if numbond[i]-double_bond==1:
                    # A-H condition: just add the hydrogen randomly
                    new_coord=new_geo[i]+bond_length*np.array([1.02, 0.0, 0.0])
                    new_E.append("H")
                    new_geo.append(new_coord)
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
                elif numbond[i]-double_bond==2:
                    if num_add_hydrogen==1:
                        # B-A-H condition: add hydrogen along AB vector
                        vec=[new_geo[i][0]-new_geo[connect_ids[0]][0], new_geo[i][1]-new_geo[connect_ids[0]][1], new_geo[i][2]-new_geo[connect_ids[0]][2]]
                        vec=vec/np.linalg.norm(vec)
                        new_coord=new_geo[i]+vec*bond_length*1.02
                        new_E.append("H")
                        new_geo.append(new_coord)
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
                    elif num_add_hydrogen==2:
                        # H-A-H condition: add hydrogen randomly for first hydrogen and add another one along vecton A-H1
                        # First H
                        new_coord=new_geo[i]+bond_length*np.array([1.02, 0.0, 0.0])
                        new_E.append("H")
                        new_geo.append(new_coord)
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
                        # Second H
                        new_coord=new_geo[i]+bond_length*np.array([-1.02, 0.0, 0.0])
                        new_E.append("H")
                        new_geo.append(new_coord)
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
                elif numbond[i]-double_bond==3:
                    if num_add_hydrogen==1:
                        # B-A(H)-C condition: find the middle point between BC.
                        # find vector A and the point. Then, locate hydrogen.
                        vec_BA=[new_geo[i][0]-new_geo[connect_ids[0]][0], new_geo[i][1]-new_geo[connect_ids[0]][1], new_geo[i][2]-new_geo[connect_ids[0]][2]]
                        vec_CA=[new_geo[i][0]-new_geo[connect_ids[1]][0], new_geo[i][1]-new_geo[connect_ids[1]][1], new_geo[i][2]-new_geo[connect_ids[1]][2]]
                        vec_BA=vec_BA/np.linalg.norm(vec_BA)
                        vec_CA=vec_CA/np.linalg.norm(vec_CA)
                        cross_vec=np.cross(vec_BA, vec_CA)
                        if np.linalg.norm(cross_vec)<1E-5: # angle (BAC) is nearly 180 deg. Contruct the plane perpendcular to vec_BA
                            # the plane would be vec[0]*x+vec[1]*y+vec[2]*z=vec*A
                            dot=np.dot(vec_BA, new_geo[i])
                            vec=[abs(vec_BA[0]), abs(vec_BA[1]), abs(vec_BA[2])]
                            max_idx=vec.index(max(vec))
                            if max_idx==0:
                               point=[dot/vec_BA[0], 0.0, 0.0]
                            elif max_idx==1:
                               point=[0.0, dot/vec_BA[1], 0.0]
                            else:
                               point=[0.0, 0.0, dot/vec_BA[2]]
                            vec=np.array([point[0]-new_geo[i][0], point[1]-new_geo[i][1], point[2]-new_geo[i][2]])
                        else:
                            middle=[(new_geo[connect_ids[0]][0]+new_geo[connect_ids[1]][0])/2.0, (new_geo[connect_ids[0]][1]+new_geo[connect_ids[1]][1])/2.0, (new_geo[connect_ids[0]][2]+new_geo[connect_ids[1]][2])/2.0]
                            vec=np.array([new_geo[i][0]-middle[0], new_geo[i][1]-middle[1], new_geo[i][2]-middle[2]])
                        new_coord=new_geo[i]+(vec/np.linalg.norm(vec))*bond_length*1.02
                        new_E.append("H")
                        new_geo.append(new_coord)
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
                    elif num_add_hydrogen==2:
                        # H-A(H)-C condition: add first one by vector AC and insert another one follow the rule above..
                        vec=[new_geo[i][0]-new_geo[connect_ids[0]][0], new_geo[i][1]-new_geo[connect_ids[0]][1], new_geo[i][2]-new_geo[connect_ids[0]][2]]
                        vec=np.array(vec)/np.linalg.norm(vec)
                        new_coord=new_geo[i]+vec*bond_length*1.02
                        new_E.append("H")
                        new_geo.append(new_coord)
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
                        dot=np.dot(vec, new_geo[i])
                        tmp=[abs(vec[0]), abs(vec[1]), abs(vec[2])]
                        max_idx=tmp.index(max(tmp))
                        if max_idx==0:
                            point=[dot/vec[0], 0.0, 0.0]
                        elif max_idx==1:
                            point=[0.0, dot/vec[1], 0.0]
                        else:
                            point=[0.0, 0.0, dot/vec[2]]
                        vec=np.array([point[0]-new_geo[i][0], point[1]-new_geo[i][1], point[2]-new_geo[i][2]])
                        vec=np.array(vec)/np.linalg.norm(vec)
                        new_coord=new_geo[i]+vec*bond_length*1.02
                        new_E.append("H")
                        new_geo.append(new_coord)
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
                    elif num_add_hydrogen==3:
                        # AH3 condition: use regular triangle to construct this molecule.
                        new_coord=[new_geo[i][0], new_geo[i][1]+1.0*bond_length*1.02, new_geo[i][2]]
                        new_E.append("H")
                        new_geo.append(new_coord)
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
                        new_coord=[new_geo[i][0]-1.732/2.0*bond_length*1.02, new_geo[i][1]-1.0/2.0*bond_length*1.02, new_geo[i][2]]
                        new_E.append("H")
                        new_geo.append(new_coord)
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
                        new_coord=[new_geo[i][0]+1.732/2.0*bond_length*1.02, new_geo[i][1]-1.0/2.0*bond_length*1.02, new_geo[i][2]]
                        new_E.append("H")
                        new_geo.append(new_coord) 
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
                elif numbond[i]-double_bond==4:
                    if num_add_hydrogen==1:
                        # (B)A(C)(D)H condition
                        middle=[(new_geo[connect_ids[0]][0]+new_geo[connect_ids[1]][0]+new_geo[connect_ids[2]][0])/3.0,\
                                (new_geo[connect_ids[0]][1]+new_geo[connect_ids[1]][1]+new_geo[connect_ids[2]][1])/3.0,\
                                (new_geo[connect_ids[0]][2]+new_geo[connect_ids[1]][2]+new_geo[connect_ids[2]][2])/3.0]
                        vec=[new_geo[i][0]-middle[0], new_geo[i][1]-middle[1], new_geo[i][2]-middle[2]]
                        if np.linalg.norm(vec)<1E-5: # point A is to close to the plane
                            vec1=[new_geo[connect_ids[0]][0]-new_geo[connect_ids[1]][0],\
                                  new_geo[connect_ids[0]][1]-new_geo[connect_ids[1]][1],\
                                  new_geo[connect_ids[0]][2]-new_geo[connect_ids[1]][2]]
                            vec2=[new_geo[connect_ids[0]][0]-new_geo[connect_ids[2]][0],\
                                  new_geo[connect_ids[0]][1]-new_geo[connect_ids[2]][1],\
                                  new_geo[connect_ids[0]][2]-new_geo[connect_ids[2]][2]]
                            vec=np.cross(vec1, vec2)
                        new_coord=new_geo[i]+(vec/np.linalg.norm(vec))*bond_length*1.02
                        new_E.append("H")
                        new_geo.append(new_coord)
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
                    elif num_add_hydrogen==2:
                        # (B)AH2(C)
                        vec_BA=[new_geo[i][0]-new_geo[connect_ids[0]][0], new_geo[i][1]-new_geo[connect_ids[0]][1], new_geo[i][2]-new_geo[connect_ids[0]][2]]
                        vec_CA=[new_geo[i][0]-new_geo[connect_ids[1]][0], new_geo[i][1]-new_geo[connect_ids[1]][1], new_geo[i][2]-new_geo[connect_ids[1]][2]]
                        vec_BA=vec_BA/np.linalg.norm(vec_BA)
                        vec_CA=vec_CA/np.linalg.norm(vec_CA)
                        cross_vec=np.cross(vec_BA, vec_CA)
                        if np.linalg.norm(cross_vec)<1E-5: # angle (BAC) is nearly 180 deg. Contruct the plane perpendcular to vec_BA
                            # the plane would be vec[0]*x+vec[1]*y+vec[2]*z=vec*A
                            dot=np.dot(vec_BA, new_geo[i])
                            vec=[abs(vec_BA[0]), abs(vec_BA[1]), abs(vec_BA[2])]
                            max_idx=vec.index(max(vec))
                            if max_idx==0:
                               point=[dot/vec_BA[0], 0.0, 0.0]
                            elif max_idx==1:
                               point=[0.0, dot/vec_BA[1], 0.0]
                            else:
                               point=[0.0, 0.0, dot/vec_BA[2]]
                            vec=np.array([point[0]-new_geo[i][0], point[1]-new_geo[i][1], point[2]-new_geo[i][2]])
                        else:
                            middle=[(new_geo[connect_ids[0]][0]+new_geo[connect_ids[1]][0])/2.0, (new_geo[connect_ids[0]][1]+new_geo[connect_ids[1]][1])/2.0, (new_geo[connect_ids[0]][2]+new_geo[connect_ids[1]][2])/2.0]
                            vec=np.array([new_geo[i][0]-middle[0], new_geo[i][1]-middle[1], new_geo[i][2]-middle[2]])
                        new_coord=new_geo[i]+(vec/np.linalg.norm(vec))*bond_length*1.02
                        new_E.append("H")
                        new_geo.append(new_coord)
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
                        new_coord=new_geo[i]-(vec/np.linalg.norm(vec))*bond_length*1.02
                        new_E.append("H")
                        new_geo.append(new_coord)
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
                    elif num_add_hydrogen==3:
                        # BAH3 condition
                        vec=[new_geo[i][0]-new_geo[connect_ids[0]][0], new_geo[i][1]-new_geo[connect_ids[0]][1], new_geo[i][2]-new_geo[connect_ids[0]][2]]
                        new_coord=new_geo[i]+(vec/np.linalg.norm(vec))*bond_length*1.02
                        new_E.append("H")
                        new_geo.append(new_coord)
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
                        dot=np.dot(vec, new_geo[i])
                        tmp=[abs(vec[0]),abs(vec[1]), abs(vec[2])]
                        max_idx=tmp.index(max(tmp))
                        if max_idx==0:
                            point=[dot/vec[0], 0.0, 0.0]
                        elif max_idx==1:
                            point=[0.0, dot/vec[1], 0.0]
                        else:
                            point=[0.0, 0.0, dot/vec[2]]
                        vec=np.array([point[0]-new_geo[i][0], point[1]-new_geo[i][1], point[2]-new_geo[i][2]])
                        new_coord=new_geo[i]+(vec/np.linalg.norm(vec))*bond_length*1.02
                        new_E.append("H")
                        new_geo.append(new_coord)
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
                        new_coord=new_geo[i]-(vec/np.linalg.norm(vec))*bond_length*1.02
                        new_E.append("H")
                        new_geo.append(new_coord)
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
        try:
            #for count_i, i in enumerate(new_E):
            #    print(f"{i} {new_geo[count_i][0]} {new_geo[count_i][1]} {new_geo[count_i][2]}")
            #print(new_bondmat)
            new_geo=opt_geo(new_E, new_geo, new_bondmat)
            
        except:
            print("Model reactant is failed to optimize.")
            return [], [], []
        try:
            new_BE_change=np.asarray(new_BE_change)
            new_bondmat=new_bondmat+new_BE_change
            new_change_geo=opt_geo(new_E, new_geo, new_bondmat)
        except:
            print("Model product is failed to optimize.")
            return [], [], []
        return new_E, new_geo, new_change_geo

def return_adj_change(adjmat):
    keep_idx=[]
    reactive_atoms=[]
    for i in range(len(adjmat)):
        for j in range(len(adjmat)):
            if i > j:
                if adjmat[i][j]!=0:
                    keep_idx+=[(i, j)]
                    reactive_atoms.append(i)
                    reactive_atoms.append(j)
    return keep_idx, reactive_atoms
