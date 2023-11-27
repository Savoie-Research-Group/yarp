import sys, itertools, timeit, os
import numpy as np
from yarp.taffi_functions import table_generator,return_rings,adjmat_to_adjlist,canon_order
from yarp.properties import el_to_an,an_to_el,el_mass
from yarp.find_lewis import find_lewis,return_formals,return_n_e_accept,return_n_e_donate,return_formals,return_connections,return_bo_dict
from yarp.hashes import atom_hash,yarpecule_hash
from yarp.input_parsers import xyz_parse,xyz_q_parse,xyz_from_smiles, mol_parse
from yarp.misc import merge_arrays, prepare_list
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import EnumerateStereoisomers, AllChem, TorsionFingerprints, rdmolops, rdDistGeom
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.ML.Cluster import Butina
from utils import *

class reaction:
    """
    Base class for storing information of a reaction and performing conformational sampling
    
    Attributes
    ----------

    reactant: yarpecule class for reactant

    product: yarpecule class for product

    
    conf_path: the folder to store conformers    

    opt: perform initial geometry optimization on product side. (default: False)
    
    """
    def __init__(self, reactant, product, args:dict):
        
        self.reactant=reactant
        self.product=product

        # safe check
        for count_i, i in enumerate(reactant.elements):
            if i != product.elements[count_i]:
                print("Fatal error: reactant and product are not same. Please check the input.....")
                exit()
        
        if opt: geometry_opt()
        self.reactant_conf={}
        self.product_conf={}
        self.reactant_inchi=return_inchikey(self.reactant)
        self.product_inchi=return_inchikey(self.product)
        self.conf_path=args['conf_path']
        self.n_conf=int(args["n_conf"])
        self.strategy=int(args['strategy'])
        self.method=args['method']
        
        if os.path.isdir(self.conf_path) is False: os.system('mkdir {}'.format(self.conf_path))

    def conf_gen(self):
        """
        Perform conformational sampling
        
        Attributes
        ----------
        
        method: the conformational sampling methods. (option: crest, rdkit) (default: rdkit)
        
        """
        if self.method=='rdkit': self.conf_rdkit()
        if self.method=='crest': self.conf_crest()

    def conf_rdkit(self):
        if self.strategy==0 or self.strategy==2:
            if os.path.isdir('{}/{}'.format(self.conf_path, self.reactant_inchi)) is False: os.system('mkdir {}/{}'.format(self.conf_path, self.reactant_inchi))
            if os.path.isfile('{}/{}/rdkit_conf.xyz'.format(self.conf_path, self.reactant_inchi)) is False:
                # sampling on reactant side
                mol_file='.reactant.tmp.mol'
                mol_write(mol_file, self.reactant, append_opt=False)
                mol=Chem.rdmolfiles.MolFromMolFile(mol_file, removeHs=False)
                ids=AllChem.EmbedMultipleConfs(mol, useRandomCoords=True, numConfs=50, maxAttempts=1000000, pruneRmsThresh=0.1,\
                                               useExpTorsionAnglePrefs=False, useBasicKnowledge=True, enforceChirality=False)
                ids=list(ids)
                out=open('{}/{}/rdkit_conf.xyz'.format(self.conf_path, self.reactant_inchi), 'w+')
                os.system('rm .reactant.tmp.mol')
                for count_i, i in enumerate(ids):
                    geo=mol.GetConformer(i).GetPositions()
                    self.reactant_conf[count_i]=geo
                    out.write('{}\n\n'.format(len(self.reactant.elements)))
                    for count, e in enumerate(self.reactant.elements):
                        out.write('{} {} {} {}\n'.format(e.capitalize(), geo[count][0], geo[count][1], geo[count][2]))
            else:
                _, geo=xyz_parse('{}/{}/rdkit_conf.xyz'.format(self.conf_path, self.reactant_inchi), multiple=True)
                for count_i, i in enumerate(geo):
                    self.reactant_conf[count_i]=i
        if self.strategy==1 or self.strategy==2:
            if os.path.isdir('{}/{}'.format(self.conf_path, self.product_inchi)) is False: os.system('mkdir {}/{}'.format(self.conf_path, self.product_inchi))
            if os.path.isfile('{}/{}/rdkit_conf.xyz'.format(self.conf_path, self.product_inchi)) is False:
                # sampling on reactant side
                mol_file='.product.tmp.mol'
                mol_write(mol_file, self.product, append_opt=False)
                mol=Chem.rdmolfiles.MolFromMolFile(mol_file, removeHs=False)
                ids=AllChem.EmbedMultipleConfs(mol, useRandomCoords=True, numConfs=50, maxAttempts=1000000, pruneRmsThresh=0.1,\
                                               useExpTorsionAnglePrefs=False, useBasicKnowledge=True, enforceChirality=False)
                ids=list(ids)
                out=open('{}/{}/rdkit_conf.xyz'.format(self.conf_path, self.product_inchi), 'w+')
                os.system('rm .product.tmp.mol')
                for count_i, i in enumerate(ids):
                    geo=mol.GetConformer(i).GetPositions()
                    self.product_conf[count_i]=geo
                    out.write('{}\n\n'.format(len(self.product.elements)))
                    for count, e in enumerate(self.product.elements):
                        out.write('{} {} {} {}\n'.format(e.capitalize(), geo[count][0], geo[count][1], geo[count][2]))
            else:
                _, geo=xyz_parse('{}/{}/rdkit_conf.xyz'.format(self.conf_path, self.product_inchi), multiple=True)
                for count_i, i in enumerate(geo):
                    self.product_conf[count_i]=i

    def conf_crest(self):
        
