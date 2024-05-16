# this program handles the model reaction problems and is created by Hsuan-Hao Hsu (hsu205@purdue.edu).
import sys, itertools, timeit, os, copy                                                                                                                                                               
from openbabel import pybel
from openbabel import openbabel as ob
from collections import Counter
import numpy as np
import yarp as yp
import yaml, fnmatch, pickle
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
from main_xtb import initialize

def main(args:dict):
    args, logger=initialize(args)
    if os.path.isfile(args["input"]): # Read smiles in
        mol=[i.split('\n')[0] for i in open(args["input"], 'r+').readlines()]
    else:
        mol=[args["input"]+"/"+i for i in os.listdir(args["input"]) if fnmatch.fnmatch(i, '*.xyz') or fnmatch.fnmatch(i, '*.mol')]
    radical=[]
    for i in mol:
        reactant=yp.yarpecule(i)
        # find the reactant is radical or not
        bmat=reactant.bond_mats[0]
        is_rad=0
        for j in range(len(bmat)):
            is_rad+=int(bmat[j, j])%2
        if is_rad: radical.append(reactant)
        else:
            tmp=generate_uniradical(reactant)
            for j in tmp: radical.append(j)
    # running enumerations
    reactions=[]
    for i in radical:
        break_mol=list(yp.break_bonds(i, n=args["n_break"]))
        products=yp.form_n_bonds(break_mol, n=args["n_break"], def_only=True)
        products=[_ for _ in products if _.bond_mat_scores[0]<=1.0]
        for j in products: reactions.append(reaction(i, j, args=args, opt=True))
    
    with open("true_rxns.p", "wb") as f:
        pickle.dump(reactions, f)
    
    MR_rxns, MR_dict=create_model_reactions(reactions)
    with open(args["reaction_data"], "wb") as f:
        pickle.dump(MR_rxns, f)
    with open("MR_dict.p", "wb") as f:
        pickle.dump(MR_dict, f)
    return MR_rxns, MR_dict

def create_model_reactions(reaction):
    # this function is given the set of true reaction and return a set of model reactions.
    # this function will generate one list and one dictionary.
    # the list will store all model reactions
    # the dictionary will store the infomation between model reaction and true reaction.
    MR_list=[]
    MR_dict=dict()
    for rxn in reaction:
        MR=return_model_rxn(rxn)
        if MR==[]: continue
        if MR.hash not in MR_dict.keys():
            MR_dict[MR.hash]=[rxn.hash]
            MR_list.append(MR)
        else:
            MR_dict[MR.hash].append(rxn.hash)
    return MR_list, MR_dict

def generate_uniradical(reactant):
    # give a neutral species, this function will remove a hydrogen atom and generate a list of uni-radical
    # find hydrogen atom first
    hydrogen=[]
    for count, atom in enumerate(reactant.elements):
        if atom.lower()=='h': hydrogen.append(count)
    # print(hydrogen)
    product=list(yp.break_bonds(reactant, n=1, react=[hydrogen]))
    # remove the hydrogen radical in reactant and product list
    radicals=[]
    for prod in product:
        elements=[]
        P_geo=[]
        prod=geometry_opt(prod)
        for count_i, i in enumerate(range(len(prod.adj_mat))):
            if prod.adj_mat[count_i].sum()==0: # the seperate hydrogen atom
               continue
            else:
               elements.append(prod.elements[count_i])
               P_geo.append(prod.geo[count_i])
        out=open(".tmp_P.xyz", "w+")
        out.write(f"{len(elements)}\n\n")
        for count_i, i in enumerate(P_geo):
            out.write(f"{elements[count_i]} {i[0]} {i[1]} {i[2]}\n")
        out.close()
        P=yp.yarpecule(".tmp_P.xyz", canon=False)
        os.system("rm .tmp_P.xyz")
        radicals.append(P)
    return radicals

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
    if new_P_geo==[] or new_R_geo==[]:
        print("Failed optimize geometry.")
        return []
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
            argu=1
            cycle=0
            while argu or cycle<10:
                if cycle<3: vec=[bond_length*cos(const), bond_length*sin(const), 0.0]
                elif cycle<6: vec=[0.0, bond_length*cos(const), bond_length*sin(const)]
                else: vec=[bond_length*cos(const), 0.0, bond_length*sin(const)]
                new_coord=new_geo[i]+[np.random.random()*0.01, np.random.random()*0.01, np.random.random()*0.01]+vec
                argu=0
                cycle=cycle+1
                for old_coord in new_geo:
                    dist=(old_coord[0]-new_coord[0])**2.0+(old_coord[1]-new_coord[1])**2.0+(old_coord[2]-new_coord[2])**2.0
                    dist=dist**0.5
                    if dist<1.0: argu=1
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
    if len(new_E)<=1:
        return [], [], []
    new_bondmat=np.asarray(new_bondmat)
    #print(new_E)
    #print(new_bondmat)
    try:
        new_geo=opt_geo(new_E, new_geo, new_bondmat)
    except:
        return [], [], []
    try:
        new_BE_change=np.asarray(new_BE_change)
        new_bondmat=new_bondmat+new_BE_change
        new_change_geo=opt_geo(new_E, new_geo, new_bondmat) 
    except:
        return [], [], []
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

if __name__=="__main__":
    parameters = sys.argv[1]
    parameters = yaml.load(open(parameters, "r"), Loader=yaml.FullLoader)
    main(parameters)
