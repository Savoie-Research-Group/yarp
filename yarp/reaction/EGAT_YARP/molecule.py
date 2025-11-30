#load yaml
import omegaconf
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem
try:
    from graphgenhelperfunctions import return_matrix
except ImportError:
    from yarp.reaction.EGAT_YARP.graphgenhelperfunctions import return_matrix
try:
    from utilities.yarp.taffi_functions import graph_seps, adjmat_to_adjlist
except ImportError:
    from yarp.reaction.EGAT_YARP.utilities.yarp.taffi_functions import graph_seps, adjmat_to_adjlist
try:
    from utilities.yarp.taffi_functions import return_rings
except ImportError:
    from yarp.reaction.EGAT_YARP.utilities.yarp.taffi_functions import return_rings
try:
    from graphgenhelperfunctions import find_stereochemistry
except ImportError:
    from yarp.reaction.EGAT_YARP.graphgenhelperfunctions import find_stereochemistry
el_to_an = { "h": 1,  "he": 2,
             "li":3,  "be":4,                                                                                                      "b":5,    "c":6,    "n":7,    "o":8,    "f":9,    "ne":10,
             "na":11, "mg":12,                                                                                                     "al":13,  "si":14,  "p":15,   "s":16,   "cl":17,  "ar":18,
             "k":19, "ca":20,  "sc":21,  "ti":22,  "v":23,  "cr":24,  "mn":25,  "fe":26,  "co":27,  "ni":28,  "cu":29,  "zn":30,  "ga":31,  "ge":32,  "as":33,  "se":34,  "br":35,  "kr":36,
             "rb":37, "sr":38,  "y":39,   "zr":40,  "nb":41, "mo":42,  "tc":43,  "ru":44,  "rh":45,  "pd":46,  "ag":47,  "cd":48,  "in":49,  "sn":50,  "sb":51,  "te":52,  "i":53,   "xe":54,
             "cs":55, "ba":56,            "hf":72,  "ta":73, "w":74,   "re":75,  "os":76,  "ir":77,  "pt":78,  "au":79,  "hg":80,  "tl":81,  "pb":82,  "bi":83,  "po":84,  "at":85,  "rn":86}
# add values for title case
for _ in list(el_to_an.keys()):
    el_to_an[_.title()] = el_to_an[_]

el_mass = {'H':1.00794,'He':4.002602,'Li':6.941,'Be':9.012182,'B':10.811,'C':12.011,'N':14.00674,'O':15.9994,'F':18.9984032,'Ne':20.1797,
                     'Na':22.989768,'Mg':24.3050,'Al':26.981539,'Si':28.0855,'P':30.973762,'S':32.066,'Cl':35.4527,'Ar':39.948,
                     'K':39.0983,'Ca':40.078,'Sc':44.955910,'Ti':47.867,'V':50.9415,'Cr':51.9961,'Mn':54.938049,'Fe':55.845,'Co':58.933200,'Ni':58.6934,'Cu':63.546,'Zn':65.39,
                     'Ga':69.723,'Ge':72.61,'As':74.92159,'Se':78.96,'Br':79.904,'Kr':83.80,
                     'Rb':85.4678,'Sr':87.62,'Y':88.90585,'Zr':91.224,'Nb':92.90638,'Mo':95.94,'Tc':98.0,'Ru':101.07,'Rh':102.90550,'Pd':106.42,'Ag':107.8682,'Cd':112.411,
                     'In':114.818,'Sn':118.710,'Sb':121.760,'Te':127.60,'I':126.90447,'Xe':131.29,
                     'Cs':132.90545,'Ba':137.327,'La':138.9055,'Hf':178.49,'Ta':180.9479,'W':183.84,'Re':186.207,'Os':190.23,'Ir':192.217,'Pt':195.078,'Au':196.96655,'Hg':200.59,
                     'Tl':204.3833,'Pb':207.2,'Bi':208.98038,'Po':209.0,'At':210.0,'Rn':222.0}
# add values for lower case
for _ in list(el_mass.keys()):
    el_mass[_.lower()] = el_mass[_]

element_encode = {key.capitalize(): [value, el_mass[key.capitalize()]] for key, value in el_to_an.items()}
atom_chiral_encode = {'?': [0,0,1], 'R': [0,1,0], 'S': [1,0,0]}
bond_encode = {'T1': [0,0,0,1,0],'T2':[0,0,1,0,0],'T3':[0,1,0,0,0],'T4':[1,0,0,0,0],'T5':[0,0,0,0,1]}
OLD_BOND_ENCODE = {'T1': [0,0,0,1],'T2':[0,0,1,0],'T3':[0,1,0,0],'T4':[1,0,0,0],'T5':[0,0,0,1]}
bond_order_encode = {'B0': [0,0,0,0,1],'B1':[0,0,0,1,0],'B2':[0,0,1,0,0],'B3':[0,1,0,0,0],'BA':[1,0,0,0,0]}
bond_stereo_encode = {'ANY': [0,0,1], 'E': [0,1,0], 'Z': [1,0,0]}
bond_rotat_encode = {'TRUE':[1,0],'FALSE':[0,1]}
FULL_HYB_ENCODE = {Chem.HybridizationType.S: [0,0,0,1,0,0,0,0,0], Chem.HybridizationType.SP: [0,0,1,0,0,0,0,0,0], Chem.HybridizationType.SP2: [0,1,0,0,0,0,0,0,0], 
                                    Chem.HybridizationType.SP3: [1,0,0,0,0,0,0,0,0],Chem.HybridizationType.SP3D: [0,0,0,0,1,0,0,0,0],Chem.HybridizationType.SP3D2: [0,0,0,0,0,1,0,0,0],
                                    Chem.HybridizationType.OTHER: [0,0,0,0,0,0,1,0,0],Chem.HybridizationType.UNSPECIFIED: [0,0,0,0,0,0,0,1,0],Chem.HybridizationType.SP2D: [0,0,0,0,0,0,0,0,1]}
LIMITED_HYB_ENCODE = {Chem.HybridizationType.S: [0,0,0,1], Chem.HybridizationType.SP: [0,0,1,0], Chem.HybridizationType.SP2: [0,1,0,0], Chem.HybridizationType.SP3: [1,0,0,0]}


class Molecule():
    def __init__(self, smiles:str, arguments):
        """
        Class that generates the molecule from the SMILES string.
        Parameters
        ----------
        smiles: str
                SMILES string of the molecule
        arguments: argparse.Namespace
                Arguments passed to the class.
        Returns
        -------
        None
        """
        self.smiles = smiles
        self.args   = arguments
        self.reactive_atoms = []
        self.Btype = []
        self.Radicals = [] 
        self.lp = [] # lone pairs
        self.br = None # bond rotatable
        if self.args.useFullHyb:
            self.atom_hybrid_encode = FULL_HYB_ENCODE
        else:
            self.atom_hybrid_encode = LIMITED_HYB_ENCODE

    def ReadMolecule(self,verbose=False):
        smiles = self.smiles
        args=self.args

        ###### PARSE ELEMENTS AND GET THE RELEVANT MATRICES
        self.elements, self.adj, self.bond_mat, self.fc = return_matrix(smiles)
        ###### GENERATE THE DISTANCE MATRIX
        self.gs = graph_seps(self.adj)
        self.gs[self.gs < 0] = 100
        ###### GET THE LIST OF RING ATOMS 
        self.ring_atoms = return_rings(adjmat_to_adjlist(self.adj),max_size=20,remove_fused=True)
        ###### GET THE LOCATION OF RADICALS
        if args.getradical == 'RDKit':
            #Obtain list of radicals and diradicals via RDKit
            self.Radicals = return_radicals_RDKit(smiles)
            self.lp       = return_lonepairs_RDKit(smiles)
        elif args.getradical == 'YARP':
            self.lp,self.Radicals = getradicalsYARP(smiles)

        ###### GET THE LOCATION OF ROTATABLE BONDS
        if args.getbondrot is True:
            self.br = getRotatableBondCount(smiles)

        if args.geteneg:
            self.eneg = get_electronegativity(smiles)

        if args.getspiro:
            self.spiro = GetSpiroAtoms(smiles)
                            
        if args.getbridgehead:
            self.brid = GetBridgeheadAtoms(smiles)

        if args.getbondpolarity:
            self.bpol = calculate_bond_polarity(smiles)

        ###### GET THE BOND STEREOCHEMISTRY, ATOM HYBRIDIZATION AND AROMATICITY, LOCATION OF CHIRAL CENTERS. 
        self.CT, self.AA, self.HY, self.BS, self.Conj, self.BA = find_stereochemistry(smiles)

    def GenerateAtomFeature(self,verbose=False):
        #print(f"Generating Atom Features\n")
        atom_features = []
        molecule = Chem.MolFromSmiles(self.smiles,sanitize=False)
        for ind in range(len(self.elements)):
            ###### GET ELEMENTS 
            atom_mappings = min([atom.GetAtomMapNum() for atom in molecule.GetAtoms()])
            #print(f"ind: {ind}, Rsmiles: {Rsmiles}, molecule: {molecule}, Generated Atom Mapping: {atom_mappings}\n")
            if atom_mappings == 0:
                ind_in_mol = ind
            else:
                ind_in_mol = ind+1
            Etype = element_encode[self.elements[ind]]
            ###### GET NEIGHBORS AND THEIR TYPES
            # There is an option to only look at Hydrogen Bonding
            neighbors = [self.elements[counti] for counti,i in enumerate(self.adj[ind,:]) if i != 0]
            if self.args.onlyH:
                NE_count = [neighbors.count('H')]
            else:
                NE_count = [neighbors.count('H'), neighbors.count('C'), neighbors.count('N'), neighbors.count('O')]
         
            # distance matrix, if molecular, then reactive_atoms should be an empty list
            if len(self.reactive_atoms) > 0:
                dis = min([self.gs[ind][indr] for indr in self.reactive_atoms])
            else:
                dis = 0 
            ###### CHECK IF ATOM IN A RING 
            if True in [ind in ra_list for ra_list in self.ring_atoms]: inR = 1
            else: inR = 0
            
            ###### GET AROMATICITY
            if self.elements[ind] == 'H': 
                aromaticity = 0
            elif self.AA[ind_in_mol]: 
                aromaticity = 1
            else: 
                aromaticity = 0
       
            ###### GET HYBRIDIZATION
            if self.elements[ind] == 'H':
                if not self.args.useFullHyb: 
                    hybrid = self.atom_hybrid_encode[Chem.HybridizationType.S]
                else:
                    hybrid = [0,0,0,1,0,0,0,0,0]
            else:
                hybrid = self.atom_hybrid_encode[self.HY[ind_in_mol]]

            ###### CHECK IF IT IS IN A CHIRAL CENTER
            if ind_in_mol in self.CT: 
                chiral = atom_chiral_encode[self.CT[ind_in_mol]]
            else: 
                chiral = [0,0,1]

            ###### MERGE ALL FEATURES 
            # Note: If statements are used to check for feature importance (that is a later step of analysis) and for adding additional features 
            atomfeature = []
            node_feats = 0
            # Add Element info if not stated that you want to remove it from training. 
            if not self.args.removeelementinfo:
                #print(f"Etype: {Etype}\n")
                atomfeature += Etype
                node_feats += len(Etype)
                if verbose: print(f"remove_Element_Info node_feats: {node_feats}\n")
            
            # Add Neighbor info if not stated that you want to remove it from training. 
            if not self.args.removeneighborcount:
                atomfeature += NE_count
                #print(f"NE_count: {NE_count}\n")
                node_feats += len(NE_count)
                if verbose: print(f"remove_Neighbor node_feats: {node_feats}\n")
            
            # Add reactive info if not stated that you want to remove it from training. 
            if not self.args.removereactiveinfo:
                if not self.args.molecular:
                    atomfeature += [dis]
                    #print(f"dis: {dis}\n")
                    node_feats += 1
                    if verbose: print(f"remove_reactive_info node_feats: {node_feats}\n")
            
            # Add ring info if not stated that you want to remove it from training. 
            if not self.args.removeringinfo:
                atomfeature += [inR]
                #print(f"RingInfo: {inR}\n")
                node_feats += 1
                if verbose: print(f"remove_Ring_Info node_feats: {node_feats}\n")
            
            # Add charge info if not stated that you want to remove it from training. 
            if not self.args.removeformalchargeinfo:
                atomfeature += [self.fc[ind]]
                #print(f"formal charge: {self.fc[ind]}\n")
                node_feats += 1
                if verbose: print(f"remove_FormalCharge_Info node_feats: {node_feats}\n")
            
            # Add aromaticity info if not stated that you want to remove it from training. 
            if not self.args.removearomaticity:
                atomfeature += [aromaticity]
                #print(f"aromaticity: {aromaticity}\n")
                node_feats += 1
                if verbose: print(f"remove_Aromatic_Info node_feats: {node_feats}\n")
            
            # Add chiral info if not stated that you want to remove it from training. 
            if not self.args.removechiralinfo:
                atomfeature += chiral
                node_feats  += len(chiral)
                if verbose: print(f"remove_Chiral_Info node_feats: {node_feats}\n")
                #print(f"chiral: {chiral}\n")
            
            # Add hybrid info if not stated that you want to remove it from training. 
            if not self.args.removehybridinfo:
                atomfeature += hybrid
                #print(f"hybrid: {hybrid}\n")
                node_feats += len(hybrid)
                if verbose: print(f"remove_hybrid_Info node_feats: {node_feats}\n")
            
            # Add radical info if stated. 
            if self.args.getradical:
                atomfeature += [self.Radicals[ind]]
                #print(f"R_Radicals[ind]: {R_Radicals[ind]}\n")
                node_feats += 1
                if verbose: print(f"remove_getradical_Info node_feats: {node_feats}\n")
                # Add lone pair info if stated.
                atomfeature += [self.lp[ind]]
                node_feats += 1
                if verbose: print(f"remove_lone_pair_Info node_feats: {node_feats}\n")

            atom_features.append(atomfeature)
            if verbose: print(f"NODE FEATURE PREP = {node_feats}\n")
        self.atom_F = atom_features

    def GenerateBondFeature(self):
        
        args = self.args
        edges_u = self.edges_u
        edges_v = self.edges_v
        ###### GET BOND FEATURES
        bond_features = []
        ###### GET THE BOND ORDER
        for ind in range(len(edges_u)):
            edge_feats = 0
            bondfeature = []
            ###### GET THE ATOM PAIRS FOR EACH BOND
            edge_ind = sorted([edges_u[ind],edges_v[ind]])
            edge_ind_mol = sorted([edges_u[ind]+1,edges_v[ind]+1])
            BO = self.bond_mat[edge_ind[0],edge_ind[1]]

            ###### CHECK IF BOND IS IN A RING
            if True in [(edge_ind[0] in ra_list and edge_ind[1] in ra_list) for ra_list in self.ring_atoms]: inR = 1
            else: inR = 0
       
            if not args.removebondtypeinfo and len(self.Btype) > 0:
                bondfeature += self.Btype
                edge_feats += len(self.Btype)
 
            if not args.removeringinfo:
                bondfeature += [inR]
                edge_feats += 1
            if BO == 0:
                if not args.removebondorderinfo:
                    bondfeature += [0,0,0,0,1]
                    edge_feats += 5

                # Add Bond Conjugation info if not stated that you want to remove it from training. 
                if not args.removeconjinfo:
                    bondfeature += [0]
                    edge_feats += 1


                # Add Bond Stereochemistry info if not stated that you want to remove it from training. 
                if not args.removestereoinfo:
                    bondfeature += [0,0,0]
                    edge_feats += 3

                if args.getbondrot:
                    bondfeature += [0,0]
                    edge_feats += 2

            else:
                # parse bond order type
                if tuple(edge_ind_mol) in self.BA and self.BA[tuple(edge_ind_mol)]: bond_type = bond_order_encode['BA']
                else: bond_type = bond_order_encode['B{}'.format(int(BO))]

                ###### CHECK BOND CONJUGATION
                if tuple(edge_ind_mol) in self.Conj and self.Conj[tuple(edge_ind_mol)]: bond_conj = [1]
                else: bond_conj = [0]

                ###### CHECK BOND STEREOCHEM
                if tuple(edge_ind_mol) in self.BS: bond_stereo = bond_stereo_encode[self.BS[tuple(edge_ind_mol)]]
                else: bond_stereo = [0,0,1]

                if not args.removebondorderinfo:
                    bondfeature += bond_type
                    edge_feats += len(bond_type)

                # Add Bond Conjugation info if not stated that you want to remove it from training. 
                if not args.removeconjinfo:
                    bondfeature += bond_conj
                    edge_feats += len(bond_conj)

                # Add Bond Stereochemistry info if not stated that you want to remove it from training. 
                if not args.removestereoinfo:
                    bondfeature += bond_stereo
                    edge_feats += len(bond_stereo)

                if args.getbondrot:
                    bond_rotat_feature = []
                    if tuple(edge_ind_mol) in self.br:
                        bond_rotat_feature += bond_rotat_encode['TRUE']
                    else:
                        bond_rotat_feature += bond_rotat_encode['FALSE']
                    edge_feats += len(bond_rotat_feature)
                    bondfeature += bond_rotat_feature
            #print(f"ind in edges_u: {ind}, BO = {BO}, bondfeature = {bondfeature}\n")
            bond_features.append(bondfeature)
            #print(f"EDGE FEATURE PREP = {edge_feats}, length = {bondfeature}\n") #At most 10 edge features

        self.bond_F = bond_features




if __name__ == "__main__":
    config_file = "/groups/bsavoie2/bpiguave/EGAT-JEPA/EGAT_YARP/REACTION.yaml"
    with open(config_file, 'r') as f:
        config = omegaconf.OmegaConf.load(f)
    omegaconf.OmegaConf.set_struct(config, False)
    arguments = config
    print(arguments)

    molecule = Molecule(smiles="[C:0]([C:1]([C:2](=[O:5])[H:9])([H:10])[H:11])([O:3][O:4][H:6])([H:7])[H:8]", arguments=arguments)
    molecule.ReadMolecule()
    elements = molecule.elements
    # generate 
    edges_u,edges_v  = [],[]
    for i in range(len(elements)):
        for j in range(len(elements)):
            # if reaction, also check if P_adj >
            Record_R = False; Record_P = False
            if molecule.adj[i][j] > 0:
                Record_R = True
            if Record_R or Record_P:
                edges_u.append(i)
                edges_v.append(j)
    molecule.edges_u = edges_u
    molecule.edges_v = edges_v
    molecule.GenerateAtomFeature()
    molecule.GenerateBondFeature()
    print(molecule.atom_F)
    print(molecule.bond_F)