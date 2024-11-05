# this program handles the model reaction problems and is created by Hsuan-Hao Hsu (hsu205@purdue.edu).
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
    #args, logger=initialize(args)
    args, logger, logging_queue=initialize(args)
    
    if os.path.isfile(args["input"]) and fnmatch.fnmatch(args["input"], '*.smi'): # Read smiles in
        mol=[i.split('\n')[0] for i in open(args["input"], 'r+').readlines()]
    elif os.path.isfile(args["input"]) and fnmatch.fnmatch(args["input"], '*.xyz'):
        mol=[args["input"]+"/"+i for i in os.listdir(args["input"])]
    else:
        mol=[args["input"]+"/"+i for i in os.listdir(args["input"]) if fnmatch.fnmatch(i, '*.xyz') or fnmatch.fnmatch(i, '*.mol')]
    radical=[]
    #print(mol)
    
    for i in mol:
        print("Generate uniradical for {}".format(i))
        reactant=yp.yarpecule(i)
        # find the reactant is radical or not
        heavy=0
        CHNO=1
        for _ in reactant.elements:
            if _!="H" and _!="h": heavy+=1
            if _!="H" and _!="h" and _!="C" and _!="c" and _!="N" and _!="n" and _!="O" and _!="o": CHNO=0
        if CHNO==0: continue
        if heavy>20: continue
        bmat=reactant.bond_mats[0]
        is_rad=0
        for j in range(len(bmat)):
            is_rad+=int(bmat[j, j])%2
        if sum(np.abs(reactant.fc))>0:
            print("Ionic species. Removing....")
            continue
        elif is_rad==1:
            print("{} is a uni-radical.".format(i))
            radical.append(reactant)
        elif is_rad==0:
            tmp=generate_uniradical(reactant)
            for _ in tmp: radical.append(_)
    # running enumerations
    print(f"We have {len(radical)} radicals.")
    with open("radicals.p", "wb") as f:
        pickle.dump(radical, f)
    
    
    radical=pickle.load(open("radicals.p", "rb"))
    
    #radicals=[]
    #for _ in radical:
    #    heavy=0
    #    for i in _.elements:
    #        if i!="H" or i!="h": heavy+=1
    #    if heavy<20:
    #        radicals.append(_)
    #radical=[]
    #radical=radicals
    #reactions=[]
    # print(radical)
    # print(len(radical))
    # exit()
    """
    count=0
    break_mol=list(yp.break_bonds(radical[0], n=1))
    products=yp.form_n_bonds(break_mol, n=1, def_only=True)
    products=[_ for _ in products if _.bond_mat_scores[0]<=0.0]
    for j in products: reactions.append(reaction(radical[0], j, args=args, opt=True))
    reactions=[i for i in reactions if "Error" not in i.product_inchi and "Error" not in i.reactant_inchi]
    MR_rxns, MR_dict=create_model_reactions(reactions)
    print(len(reactions))
    print(len(MR_rxns))
    exit()
    
    count=0
    
    reactions=[]
    for count_i, i in enumerate(radical):
        break_mol=list(yp.break_bonds(i, n=args["n_break"]))
        products=yp.form_n_bonds(break_mol, n=args["n_break"], def_only=True)
        products=[_ for _ in products if _.bond_mat_scores[0]<=0.0]
        for j in products:
            reactions.append(reaction(i, j, args=args, opt=True))
    reactions=[_ for _ in reactions if "Error" not in i.product_inchi and "Error" not in i.reactant_inchi]
    with open("All_true.p", "wb") as f: pickle.dump(reactions, f)
    MR_rxns, MR_dict=create_model_reactions(reactions)
    # with open("All_true.p", "wb") as f: pickle.dump(reactions, f)
    with open("All_model.p", "wb") as f: pickle.dump(MR_rxns, f)
    with open("All_dict.p", "wb") as f: pickle.dump(MR_dict, f)
    """
    count=0
    while 1:
        reactions=[]
        # if count>=5: break
        if count+10>len(radical): bound=len(radical)
        else: bound=count+10
        if 0:
        #if os.path.isfile(f"true_{count}_{bound}.p") is True:
            reactions=pickle.load(open(f"true_{count}_{bound}.p", "rb"))
        else:
            for i in range(count, bound):
                #print("RADICAL")
                #for count_e, _ in enumerate(radical[i].geo):
                #    print(f"{radical[i].elements[count_e]} {_[0]} {_[1]} {_[2]}")
                react=[]
                rad=[]
                for count_b, _ in enumerate(radical[i].bond_mats[0]):
                    if _[count_b]%2==0: react.append(count_b)
                    else: rad.append(count_b)
                neb=[]
                for _ in rad:
                    for count_bd, bd in enumerate(radical[i].adj_mat[_]):
                        if bd: neb.append(count_bd)
                react=[_ for _ in react if _ not in neb] 
                break_mol=list(break_H_bonds(radical[i], react=react))
                #break_mol=list(yp.break_bonds(radical[i], n=args["n_break"]))
                #products=yp.form_n_bonds(break_mol, n=args["n_break"], def_only=True)
                products=form_radical_bonds(break_mol)
                products=[_ for _ in products if _.bond_mat_scores[0]<=0.0]
                for j in products:
                    print("Create reaction class....")
                    if np.sum(abs(radical[i].adj_mat-j.adj_mat))==0: continue
                    product_geo=opt_geo(j.elements, j.geo, j.bond_mats[0])
                    if product_geo!=[]:
                        j.geo=product_geo
                        reactions.append(reaction(radical[i], j, args=args, opt=False))
            reactions=[i for i in reactions if "ERROR" not in i.product_inchi and "ERROR" not in i.reactant_inchi]
            reactions=[i for i in reactions if i.product_inchi!=i.reactant_inchi]
            with open(f"MR/true_{count}_{bound}.p", "wb") as f:
                print(f"MR/true_{count}_{bound}.p")
                pickle.dump(reactions,f)
        print("Create model reaction....")
        MR_rxns, MR_dict=create_model_reactions(reactions)
        #print(len(reactions))
        #print(len(MR_rxns))
        #print(f"model_{count}_{bound}.p")
        with open(f"MR/model_{count}_{bound}.p", "wb") as f:
            pickle.dump(MR_rxns, f)
        with open(f"MR/MRdict_{count}_{bound}.p", "wb") as f:
            pickle.dump(MR_dict, f)
        if count+10>=len(radical): break
        count=count+10
        #break
        #if count>=len(radical):break
    """
    for i in radical:
        break_mol=list(yp.break_bonds(i, n=args["n_break"]))
        products=yp.form_n_bonds(break_mol, n=args["n_break"], def_only=True)
        products=[_ for _ in products if _.bond_mat_scores[0]<=3.0]
        for j in products: reactions.append(reaction(i, j, args=args, opt=True))
    with open("true_rxns.p", "wb") as f:
        pickle.dump(reactions, f)
    MR_rxns, MR_dict=create_model_reactions(reactions)
    #for rxns in MR_rxns:
    #    for count_i, i in enumerate(rxns.reactant.elements):
    #        print(f"{i} {rxns.reactant.geo[count_i][0]} {rxns.reactant.geo[count_i][1]} {rxns.reactant.geo[count_i][2]}")
    #    print("\n")
    with open(args["reaction_data"], "wb") as f:
        pickle.dump(MR_rxns, f)
    with open("MR_dict.p", "wb") as f:
        pickle.dump(MR_dict, f)
    """
    return

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
    #print(hydrogen)
    product=list(break_H_bonds(reactant, react=hydrogen))
    # remove the hydrogen radical in reactant and product list
    #print(len(product))
    #print(len(product))
    #for prod in product:
    #    print(len(prod.elements))
    #    for count, i in enumerate(prod.geo):
    #        print(f"{prod.elements[count]} {i[0]} {i[1]} {i[2]}")
    #    print("\n")
    radicals=[]
    for prod in product:
        elements=[]
        P_geo=[]
        #prod=geometry_opt(prod)
        #print(prod.adj_mat)
        for count_i, i in enumerate(range(len(prod.adj_mat))):
            if prod.adj_mat[count_i].sum()==0: # the separate hydrogen atom
               continue
            else:
               elements.append(prod.elements[count_i])
               P_geo.append(prod.geo[count_i])
        out=open(".tmp_P.xyz", "w+")
        out.write(f"{len(elements)}\n\n")
        for count_i, i in enumerate(P_geo):
            out.write(f"{elements[count_i]} {i[0]} {i[1]} {i[2]}\n")
            #print(f"{elements[count_i]} {i[0]} {i[1]} {i[2]}")
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
    adj_change=P_adj-R_adj
    bn, fm, _=return_bnfm(BE_change)
    bond_change, reactive_atoms=return_adj_change(adj_change)
    gs=graph_seps(R_adj)
    keep_idx=list(reactive_atoms)
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
    # print(keep_idx)
    # print(edge_idx)
    new_R_E, new_R_geo, new_P_geo=return_model_geo(elements, R_geo, R_bond, BE_change, keep_idx, edge_idx)
    if len(new_P_geo)<=2 or len(new_R_geo)<=2:
        print("Failed to optimize geometry for model reaction.")
        return []
    xyz_write(".tmp_R.xyz", new_R_E, new_R_geo)
    reactant=yp.yarpecule(".tmp_R.xyz", canon=False)
    os.system("rm .tmp_R.xyz")
    xyz_write(".tmp_P.xyz", new_R_E, new_P_geo)
    product=yp.yarpecule(".tmp_P.xyz", canon=False)
    os.system("rm .tmp_P.xyz")
    R=reaction(reactant, product, args=rxn.args, opt=True)
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
        #print(num_add_hydrogen)
        if num_add_hydrogen > 0:
            bond_length=el_radii[new_E[i]]+el_radii["H"]
        for j in range(num_add_hydrogen):
            new_E.append("H")
            #const=2*3.1415926/num_add_hydrogen*float(j)
            #argu=1
            #cycle=0
            connect_ids=[count_k for count_k, k in enumerate(new_bondmat[i]) if count_k!=i and k>=1]
            if len(connect_ids)==1: # don't need to use cross
                vec=[new_geo[i][0]-new_geo[connect_ids[0]][0],new_geo[i][1]-new_geo[connect_ids[0]][1], new_geo[i][2]-new_geo[connect_ids[0]][2]]
                #print(vec)
                vec=vec/np.linalg.norm(vec)
                if vec[2]>1E-6: imag_vec=np.array([1.0, 1.0, -(vec[0]+vec[1])/vec[2]])
                else: imag_vec=np.array([1.0, 1.0, 0.0])
                rotate=rotate_matrix(imag_vec, 1.0/6.0*math.pi+np.random.rand()/math.pi)
                vec=np.dot(rotate, vec)
                vec=vec/np.linalg.norm(vec)
                #print(vec)
                new_coord=new_geo[i]+vec*bond_length
            else:
                vecs=[]
                # print(connect_ids)
                for k in connect_ids:
                    vec=[new_geo[k][0]-new_geo[i][0], new_geo[k][1]-new_geo[i][1], new_geo[k][2]-new_geo[i][2]]
                    vecs.append(vec/np.linalg.norm(vec))
                for k in range(len(connect_ids)-1):
                    if k==0: 
                        vec=np.cross(vecs[k], vecs[k+1])
                        while np.linalg.norm(vec)<1E-6: # two vectors are parallel (sin(theta)=0)
                            vec=np.cross(vecs[k], vecs[k+1]+0.1*np.random.rand(3))
                            vec=vec/np.linalg.norm(vec)
                    else:
                        vec=np.cross(vec, vecs[k+1])
                        while np.linalg.norm(vec)<1E-6:
                            vec=np.cross(vec, vecs[k+1]+0.1*np.random.rand(3))
                            vec=vec/np.linalg.norm(vec)
                #vec=vec/np.linalg.norm(vec)
                goal=1
                cycle=0
                while goal:
                    cycle+=1
                    dot_product=[]
                    for k in vecs:
                        dot_product.append(np.dot(vec, k))
                    if max(dot_product)>0.95: # Too close
                        goal=1
                        abs_dot=[k for k in dot_product]
                        axis=vecs[abs_dot.index(min(abs_dot))]
                        deg=0.5*math.pi
                        rotate=rotate_matrix(axis, deg)
                        #print(cycle)
                        #print(rotate)
                        vec=np.dot(rotate, vec)
                    if max(dot_product)<0.95 or cycle>5:
                        goal=0
                #for k in vecs:
                #    if (np.dot(vec,k))>0.95: print("Close")
                new_coord=new_geo[i]+vec*bond_length+0.01*np.random.rand(3)
                #print(vecs)
                #print(vec)
            new_geo.append(new_coord)
            '''
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
                    #print(dist)
                    if dist<0.2: argu=1
            new_geo.append(new_coord)
            '''
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
    # new_geo=opt_geo(new_E, new_geo, new_bondmat)
    # exit()
    try:
        #print("A")
        #print("Model Reactant")
        #for count_i, i in enumerate(new_geo):
        #    print(f"{new_E[count_i]} {i[0]} {i[1]} {i[2]}")
        
        new_geo=opt_geo(new_E, new_geo, new_bondmat)
        #print("After OPT")
        #for count_i, i in enumerate(new_geo):
        #    print(f"{new_E[count_i]} {i[0]} {i[1]} {i[2]}")
    except:
        return [], [], []
    try:
        #print("Model Product")
        new_BE_change=np.asarray(new_BE_change)
        new_bondmat=new_bondmat+new_BE_change
        
        #for count_i, i in enumerate(new_geo):
        #    print(f"{new_E[count_i]} {i[0]} {i[1]} {i[2]}")
        #print("B")
        
        new_change_geo=opt_geo(new_E, new_geo, new_bondmat) 
    except:
        return [], [], []
    return new_E, new_geo, new_change_geo

def rotate_matrix(axis, deg):
    rotate_matrix=scipy.linalg.expm(np.cross(np.eye(3), axis/scipy.linalg.norm(axis)*deg))
    return rotate_matrix

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

def break_H_bonds(mol, react=[]):
    hashes=set([])
    bonds=[(count_r, count_c) for count_r, row in enumerate(mol.adj_mat) for count_c, col in enumerate(row) if (count_r in react and col>0)]
    for b in bonds:
        adj_mat=copy.copy(mol.adj_mat)
        adj_mat[b[0], b[1]]=0
        adj_mat[b[1], b[0]]=0
        bmat=copy.copy(mol.bond_mats[0])
        bmat[b[0], b[1]]-=1
        bmat[b[1], b[0]]-=1
        bmat[b[0], b[0]]+=1
        bmat[b[1], b[1]]+=1
        tmp=yp.yarpecule((adj_mat, mol.geo, mol.elements, mol.q), canon=False)
        tmp.bond_mats[0]=bmat
        if tmp.hash not in hashes:
            yield tmp
            hashes.add(tmp.hash)

def form_radical_bonds(mol):
    yarpecules=prepare_list(mol)
    hashes=set([_.hash for _ in yarpecules])
    for y in yarpecules:
        react=[]
        for count, _ in enumerate(y.bond_mats[0]):
            if _[count]%2!=0: react.append(count)
        if len(react)!=3: continue
        form_bonds=[(react[0], react[1]), (react[0], react[2]), (react[1], react[2])]
        for i in form_bonds:
            adj_mat=copy.copy(y.adj_mat)
            bmat=copy.copy(y.bond_mats[0])
            adj_mat[i[0], i[1]]=1
            adj_mat[i[1], i[0]]=1
            bmat[i[0], i[1]]+=1
            bmat[i[1], i[0]]+=1
            bmat[i[0], i[0]]-=1
            bmat[i[1], i[1]]-=1
            tmp=yp.yarpecule((adj_mat, y.geo, y.elements, y.q), canon=False)
            tmp.bond_mats[0]=bmat
            if tmp.hash not in hashes:
                yield tmp
                hashes.add(tmp.hash)

if __name__=="__main__":
    parameters = sys.argv[1]
    parameters = yaml.load(open(parameters, "r"), Loader=yaml.FullLoader)
    main(parameters)
