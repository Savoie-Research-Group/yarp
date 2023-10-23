import yarp as yp
import numpy as np
from yarp.find_lewis import all_zeros
from yarp.find_lewis import bmat_unique
import os, sys, yaml, fnmatch
import logging
from openbabel import pybel
from utils import *
from reaction import *
# YARP methodology by Hsuan-Hao Hsu, Qiyuan Zhao, and Brett M. Savoie
# First part: Enumeration part
def main(args:dict):
    input_path=args['input']
    output_path=args['output']
    break_bond=int(args['n_break'])
    strategy=int(args['strategy'])
    conf_path=args['conf_path']
    n_conf=int(args['n_conf'])
    method=args['method']
    form_all=int(args["form_all"])
    lewis_criteria=float(args["lewis_criteria"])
    if os.path.exists(output_path) is False: os.makedirs('{}'.format(output_path))
    print(f"""Welcome to
                __   __ _    ____  ____  
                \ \ / // \  |  _ \|  _ \ 
                 \ V // _ \ | |_) | |_) |
                  | |/ ___ \|  _ <|  __/ 
                  |_/_/   \_\_| \_\_|
                          // Yet Another Reaction Program
        """)
    if os.path.isfile(input_path): # Read smiles in
        mol=[i.split('\n')[0] for i in open(input_path, 'r+').readlines()]
    else:
        mol=[input_path+"/"+i for i in os.listdir(input_path) if fnmatch.fnmatch(i, '*.xyz') or fnmatch.fnmatch(i, '*.mol')]
    print("-----------------------")
    print("------First Step-------")
    print("------Enumeration------")
    print("-----------------------")
    for i in mol: rxns=enumeration(i, output_path, break_bond, form_all, criteria=lewis_criteria)
    print("-----------------------")
    print("---------Done----------")
    print("-----------------------")
    print("-----------------------")
    print("------Second Step------")
    print("Conformational Sampling")
    print("-----------------------")
    rxns=conf_sampling(rxns, method=method, strategy=strategy)

    return

def enumeration(input_mol, output_path, nb, form_all, criteria=0.0):
    reactant=yp.yarpecule(input_mol)
    mol=yp.yarpecule(input_mol)
    print("Do the reaction enumeration on molecule: {} ({})".format(return_inchikey(reactant),input_mol))
    name=input_mol.split('/')[-1].split('.')[0]
    # break bonds
    break_mol=list(yp.break_bonds(mol, n=nb))
    # form bonds
    if form_all: products=yp.form_bonds_all(break_mol)
    else: products=yp.form_bonds(break_mol)
    # Finish generate products
    products=[_ for _ in products if _.bond_mat_scores[0]<=criteria and sum(np.abs(_.fc))<=2.0]
    print(f"{len(products)} cleaned products after find_lewis() filtering")
    rxn=[]
    for count_i, i in enumerate(products):
        R=reaction(reactant, products[0])
        rxn.append(R)
    return rxn

def conf_sampling(rxns, method='rdkit', strategy=2):
    for count_i, i in enumerate(rxns):
        rxns[count_i]=i.conf_gen(method=method, strategy=strategy)
    return rxns

def write_reaction(R, P, filename="reaction.xyz"):
    out=open(filename, 'w+')
    out.write('{}\n'.format(len(R.elements)))
    out.write('q {}\n'.format(R.q))
    for count_i, i in enumerate(R.elements):
        if len(i)>1:
            i=i.capitalize()
            out.write("{:<20s} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(i, R.geo[count_i][0], R.geo[count_i][1], R.geo[count_i][2]))
        else: out.write("{:<20s} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(i.upper(), R.geo[count_i][0], R.geo[count_i][1], R.geo[count_i][2]))
    out.write('{}\n'.format(len(P.elements)))
    out.write('q {}\n'.format(P.q))
    for count_i, i in enumerate(P.elements):
        if len(i)>1:
            i=i.capitalize()
            out.write("{:<20s} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(i, P.geo[count_i][0], P.geo[count_i][1], P.geo[count_i][2]))
        else: out.write("{:<20s} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(i.upper(), P.geo[count_i][0], P.geo[count_i][1], P.geo[count_i][2]))
    out.close()
    return

if __name__=="__main__":
    parameters = sys.argv[1]
    parameters = yaml.load(open(parameters, "r"), Loader=yaml.FullLoader)
    main(parameters)     
