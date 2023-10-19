import yarp as yp
import numpy as np
from yarp.find_lewis import mol_write
from yarp.find_lewis import all_zeros
from yarp.find_lewis import bmat_unique
import os, sys, yaml, fnmatch
import logging
from openbabel import pybel
# YARP methodology by Hsuan-Hao Hsu, Qiyuan Zhao, and Brett M. Savoie
# First part: Enumeration part
def main(args:dict):
    input_path=args['input']
    output_path=args['output']
    break_bond=int(args['n_break'])
    if args["n_form"]=="all": form_bond=-1
    else: form_bond=int(args['n_form'])
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
    for i in mol: enumeration(i, output_path, break_bond, form_bond)
    return

def enumeration(input_mol, output_path, nb, nf):
    reactant=yp.yarpecule(input_mol)
    mol=yp.yarpecule(input_mol)
    print("Do the reaction enumeration on molecule: {}".format(input_mol))
    name=input_mol.split('/')[-1].split('.')[0]
    # break bonds
    break_mol=list(yp.break_bonds(mol, n=nb))
    # form bonds
    if nf==-1: products=yp.form_bonds_all(break_mol)
    else: products=yp.form_n_bonds(break_mol, nf=nf)
    # Finish generate products
    products=[_ for _ in products if _.bond_mat_scores[0]<=2.5 and sum(np.abs(_.fc))<=2.0]
    print(f"{len(products)} cleaned products after find_lewis() filtering")
    # Geometry optimization by mmff94/openbabel for products
    for count_i, i in enumerate(products):
        adj_mat=i.adj_mat
        geo=i.geo
        elements=i.elements
        q=i.q
        i.geo=opt_geo_pybel(geo, adj_mat, elements, q=q)
        filename='{}/{}-{}.xyz'.format(output_path, name, count_i)
        write_reaction(reactant, i, filename=filename)
    return

def opt_geo_pybel(geo, adj_mat,elements,q=0, filename='tmp'):
    tmp_filename = '.{}.mol'.format(filename)
    count = 0
    while os.path.isfile(tmp_filename):
        count += 1
        if count == 10:
            print("ERROR in opt_geo_rdkit: could not find a suitable filename for the tmp geometry. Exiting...")
            return geo
        else:
            tmp_filename = ".{}".format(filename) + tmp_filename
    mol_write(tmp_filename,elements,geo,adj_mat,q=q,append_opt=False)
    mol=next(pybel.readfile("mol", tmp_filename))
    mol.localopt(forcefield='uff')
    coords=geo
    for count_i, i in enumerate(geo):
        coords[count_i]=mol.atoms[count_i].coords
    os.system("rm {}".format(tmp_filename))
    return coords

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
