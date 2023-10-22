import sys, itertools, timeit, os, copy                                                                                                                                                                   
import numpy as np
from yarp.taffi_functions import graph_seps,table_generator,return_rings,adjmat_to_adjlist,canon_order
from yarp.properties import el_to_an,an_to_el,el_mass
from yarp.find_lewis import mol_write, find_lewis,return_formals,return_n_e_accept,return_n_e_donate,return_formals,return_connections,return_bo_dict
from yarp.hashes import atom_hash,yarpecule_hash
from yarp.input_parsers import xyz_parse,xyz_q_parse,xyz_from_smiles, mol_parse
from yarp.misc import merge_arrays, prepare_list
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import EnumerateStereoisomers, AllChem, TorsionFingerprints, rdmolops, rdDistGeom
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.ML.Cluster import Butina
def geometry_opt(molecule):
    """
    Geometry optimization on product side by pybel
    """
    mol_file='.tmp.mol'
    mol_write(mol_file, molecule, append_opt=False)
    mol=next(pybel.readfile("mol", mol_file))
    mol.localopt(forcefield='uff')
    for count_i, i in enumerate(molecule.geo):
        molecule.geo[count_i]=mol.atoms[count_i].coords
    os.system("rm {}".format(mol_file))
    return molecule

def return_inchikey(molecule):
    E=molecule.elements
    G=molecule.geo
    bond_mat=molecule.bond_mats[0]
    q=molecule.q
    gs=graph_seps(molecule.adj_mat)
    adj_mat=molecule.adj_mat
    groups=[]
    loop_ind=[]
    for i in range(len(gs)):
        if i not in loop_ind:
            new_group=[count_j for count_j, j in enumerate(gs[i, :]) if j>=0]
            loop_ind += new_group
            groups+=[new_group]
    inchikey=[]
    mol=copy.deepcopy(molecule)
    for group in groups:
        N_atom=len(group)
        mol=copy.deepcopy(molecule)
        mol.elements=[E[ind] for ind in group]
        mol.bond_mats=[bond_mat[group][:, group]]
        mol.geo=np.zeros([N_atom, 3])
        mol.adj_mat=adj_mat[group][:, group]
        for count_i, i in enumerate(group): mol.geo[count_i, :]=G[i, :]
        mol_write(".tmp.mol", mol)
        mol=next(pybel.readfile("mol", ".tmp.mol"))
        inchi=mol.write(format='inchikey').strip().split()[0]
        inchikey+=[inchi]
        os.system("rm .tmp.mol")
    if len(groups) == 1:
        return inchikey[0]
    else:
        return '-'.join(sorted([i[:14] for i in inchikey]))                   
