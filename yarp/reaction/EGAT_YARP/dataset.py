from torch.utils.data import Dataset
import pandas as pd 
try:
    from molecule import Molecule
except ImportError:
    from yarp.reaction.EGAT_YARP.molecule import Molecule
try:
    from molecule import OLD_BOND_ENCODE
except ImportError:
    from yarp.reaction.EGAT_YARP.molecule import OLD_BOND_ENCODE

import omegaconf
import os
import time
import traceback
import torch
import dgl
try:
    from RDKit.RDKitHelpers import getInchifromSMILES, RemoveMapping
except ImportError:
    from yarp.reaction.EGAT_YARP.RDKit.RDKitHelpers import getInchifromSMILES, RemoveMapping
from rdkit import Chem
try:
    from graphgenhelperfunctions import return_reactive
except ImportError:
    from yarp.reaction.EGAT_YARP.graphgenhelperfunctions import return_reactive


def getctype(smi:str, molecular:bool=False):
    """
    Determines the type of a chemical reaction or molecule based on a given SMILES string.

    Parameters:
    - smi (str): A SMILES string representing either a molecule/mixture or a chemical reaction.
    - molecular (bool, optional): If True, treats the SMILES string as a molecule/mixture. 
                                  If False, treats it as a chemical reaction. Default is False.

    Returns:
    - str: A string indicating the number of components (in molecular mode) or 
           the number of reactants and products (in reaction mode).

    Examples:
    >>> getctype("CCO.CC", molecular=True)
    'R2'
    >>> getctype("CCO.CC>>CCOC", molecular=False)
    'R2P1'
    """
    if molecular:
        return f"R{len(smi.split('.'))}"
    else:
        smi = smi.split('>>')
        return f"R{len(smi[0].split('.'))}P{len(smi[1].split('.'))}"


def GetCollateFunction(args):
    """
    Select the appropriate collate function based on the dataset configuration.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Configuration arguments containing dataset settings
        
    Returns:
    --------
    function
        The appropriate collate function for the dataset configuration
    """
    # Define the mapping of configurations to collate functions
    collate_functions = {
        # Reaction dataset collate functions
        'reaction': {
            'hasaddons': {
                'additionals': {
                    True: collateall,
                    False: collatewithaddons
                }
            },
            'no_addons': {
                'additionals': {
                    True: collatewitthadditonals,
                    False: collatetargetsonly
                }
            }
        }
    }
    
    # Determine the dataset type
    dataset_type = 'reaction'
    
    # Determine if we have addons
    addon_type = 'hasaddons' if args.hasaddons else 'no_addons'
    
    # Determine if we have additional features
    has_additionals = args.additionals is not None
    
    # Get the appropriate collate function
    collate_fn = collate_functions[dataset_type][addon_type]['additionals'][has_additionals]
    
    return collate_fn
def collatetargetsonly(samples):
    """
    Function that takes the dataset and gets the batched data. 
    Parameters
    ----------
    samples: torch.DataLoader

    Returns
    -------
    names: string
            Names of the .json files being looked at
    types: string
            Names of the reaction type of the .json file
    Rbatched_graph: DGL graph
            Graph of the reactant
    Pbatched_graph: DGL graph
            Graph of the Product
    smiles: string
            Smiles of the Reactant and Product
    targets: float
            Target values
    """
    '''
    with Pool() as p:
        samples = p.map(filter_none, samples)
    '''
    samples = filter_none(samples)
    try:
        names,types,Rgraphs, Pgraphs, smiles,targets = map(list, zip(*samples))
        Rbatched_graph = dgl.batch(Rgraphs)
        Pbatched_graph = dgl.batch(Pgraphs)
        return names,types,Rbatched_graph, Pbatched_graph, smiles,targets
    except ValueError:
        return None

def collatewitthadditonals(samples):
    """
    Function that takes the dataset and gets the batched data. 
    Parameters
    ----------
    samples: torch.DataLoader

    Returns
    -------
    names: string
            Names of the .json files being looked at
    types: string
            Names of the reaction type of the .json file
    Rbatched_graph: DGL graph
            Graph of the reactant
    Pbatched_graph: DGL graph
            Graph of the Product
    smiles: string
            Smiles of the Reactant and Product
    targets: float
            Target values
    additionals: float
            Values being passed in the FFNN
    """
    samples = filter_none(samples)
    try:
        names,types,Rgraphs, Pgraphs, smiles,targets,additionals = map(list, zip(*samples))
        Rbatched_graph = dgl.batch(Rgraphs)
        Pbatched_graph = dgl.batch(Pgraphs)
        return names,types,Rbatched_graph, Pbatched_graph, smiles,targets,additionals
    except ValueError:
        return None

def collatewithaddons(samples):
    """
    Function that takes the dataset and gets the batched data. 
    Parameters
    ----------
    samples: torch.DataLoader

    Returns
    -------
    names: string
            Names of the .json files being looked at
    types: string
            Names of the reaction type of the .json file
    Rbatched_graph: DGL graph
            Graph of the reactant
    Pbatched_graph: DGL graph
            Graph of the Product
    smiles: string
            Smiles of the Reactant and Product
    targets: float
            Target values
    Radd: list
            List of the RDKit Global Features passed from the Reactant
    Padd: list
            List of the RDKit Global Features passed from the Product
    """    
    # The input `samples` is a list of pairs (graph, target)
    samples = filter_none(samples)
    try:
        names,types,Rgraphs, Pgraphs, smiles,targets,Radd,Padd = map(list, zip(*samples))
        Rbatched_graph = dgl.batch(Rgraphs)
        Pbatched_graph = dgl.batch(Pgraphs)
        return names,types,Rbatched_graph, Pbatched_graph, smiles,targets,Radd,Padd
    except ValueError:
        return None

def collateall(samples):
    """
    Function that takes the dataset and gets the batched data. 
    Parameters
    ----------
    samples: torch.DataLoader

    Returns
    -------
    names: string
            Names of the .json files being looked at
    types: string
            Names of the reaction type of the .json file
    Rbatched_graph: DGL graph
            Graph of the reactant
    Pbatched_graph: DGL graph
            Graph of the Product
    smiles: string
            Smiles of the Reactant and Product
    targets: float
            Target values
    Radd: list
            List of the RDKit Global Features passed from the Reactant
    Padd: list
            List of the RDKit Global Features passed from the Product
    additionals: float
            Values being passed in the FFNN
    """    
    # The input `samples` is a list of pairs (graph, target)
    samples = filter_none(samples)
    try:
        names,types,Rgraphs, Pgraphs, smiles,targets,additionals,Radd,Padd = map(list, zip(*samples))
        Rbatched_graph = dgl.batch(Rgraphs)
        Pbatched_graph = dgl.batch(Pgraphs)
        return names,types,Rbatched_graph, Pbatched_graph,smiles,targets,additionals,Radd,Padd
    except ValueError:
        return None

def filter_none(samples):
    return [s for s in samples if s is not None]

class FastDataset(Dataset):
    """
    Class that creates a Torch dataset by generating the graphs from the csv on the fly. 
    Parameters
    ----------
    root: string
            Folder to look at. 
    npoints: int
            Maximum point in each molecule in the database (in RGD1 max_point = 33)
    split: string
            Split to look at. 
    class_choice: string or list
            Types of reactions to look at.
    exclude: list
            Reactions to not look at. 
    randomize:
            Shuffle the data so that the batches are loaded with different reaction classes along with shuffling the batches themselves
    fold: int
            Fold to look at
    foldtype: string
            Fold type to look at
    size: int
            Look at the first N samples
    target: string or list
            Target column or set of columns to predict
    additional: string or list
            Column or set of columns to use for predictions. 
    hasaddons: bool
            Checks if RDKit global features need to be used
    molecular: bool
            Checks if the molecular setting is used. 

    Returns
    -------
    DataLoader: torch.DataLoader
    """
    def __init__(self, args, dataset: str, 
                 class_choice=None,exclude=[], 
                 size=None,  
                 explicit_H=False):
        ###### LOAD VARIABLES
        self.args = args
        if dataset is not None:
            self.root = dataset
        else:
            self.root = args.input
        self.dataset = pd.read_csv(self.root)
        self.cat = []
        self.target = args.target
        self.additional = args.additionals
        self.addons = args.hasaddons
        self.molecular = args.molecular
        self.datapath = []
        self.rtypes = []
        self.smiles = args.smiles
        self.method_mapping = args.method_mapping
        self.getradical = args.getradical
        self.onlyH=args.onlyH
        self.getbondrot=args.getbondrot
        self.atom_map=args.atom_map
        self.removeelementinfo=args.removeelementinfo
        self.removeneighborcount=args.removeneighborcount
        self.removeringinfo=args.removeringinfo
        self.removereactiveinfo=args.removereactiveinfo
        self.removeformalchargeinfo = args.removeformalchargeinfo
        self.removearomaticity = args.removearomaticity
        self.removechiralinfo=args.removechiralinfo
        self.removehybridinfo=args.removehybridinfo
        self.removebondtypeinfo=args.removebondtypeinfo
        self.removebondorderinfo=args.removebondorderinfo
        self.removeconjinfo=args.removeconjinfo
        self.removestereoinfo=args.removestereoinfo
        self.getRDKitFeatures=args.hasaddons
        self.getRDKITNormfatures=args.hasnormedaddons
        self.isOldEdition=args.useOld
        self.useFullHyb=args.useFullHyb
        self.geteneg=args.geteneg
        self.getbpol=args.getbondpolarity
        self.getspiroinfo=args.getspiro
        self.getbridgehead=args.getbridgehead
        self.num_workers=args.num_workers
        self.data_path = args.data_path
        
        self.foldtype = args.foldtype
        self.randomize= args.randomize
        
        self.explicit_H=explicit_H
        if not os.path.isdir(self.data_path): 
            os.makedirs(self.data_path, exist_ok=True)

        if not class_choice is None:
            self.cat = [k for k in self.cat if k in class_choice]

        if self.root.split('.')[-1] == 'csv':
            start = time.time()
            data = pd.read_csv(self.root)
            ends = time.time()-start
            if len(exclude) != 0: 
                data = data[~data.index.isin(exclude)]

            if 'rxntype' not in data.columns:
                data['rxntype'] = data[self.smiles].apply(getctype,args=(self.molecular, ))
            ends = time.time()-start
            
        else:
            raise ImportError('FastDataset can only work for .csv or .xlsx or .tsv only')

    def convertSMILEStoEGATGraph(self, Rind: int, method_mapping='RxnMapper', num_workers=1):
        molecular = self.molecular
        ###### READ CSV FILE 
        try:
            ###### PARSE THE ELEMENT IN THE DATAFRAME AND SET UP THE LIST WE SAVE THE DATA TO
            rxn = self.dataset.iloc[Rind,:]
            info = {}
            info['Indices'] = Rind
            ###### GET THE SMILES STRINGS AND SAVE THE inchi keys TO THE DICTIONARY
            if not molecular:
                RPsmiles = rxn[self.smiles].split('>>')
                Rsmiles = RPsmiles[0]
                Psmiles = RPsmiles[1]
                info['Rinchi'] = getInchifromSMILES(Rsmiles)
                info['Pinchi'] = getInchifromSMILES(Psmiles)
            else:
                Rsmiles = rxn[self.smiles]
                # Add hydrogen add indices of atoms if smiles does not have hydrogen#
                if not hashydrogen(Rsmiles):
                    Rsmiles = Smiles_AddH_AAM(Rsmiles)
                info['Rinchi'] = getInchifromSMILES(Rsmiles)
            ###### IF THEY ARE NOT ATOM-MAPPED, ATOM-MAP THEM USING THE HELPER FUNCTION WE WROTE
            # Note: Right now, we can use RxnMapper as a stop-gap to get some kind of working solution.
            # One way to manually do it is by aligning the R and P geometries we generate from openbabel and then atom-map w.r.t to RMSD.
            # for now, remove atom mapping #
            if not molecular:
                NRsmiles = RemoveMapping(Rsmiles)
                NPsmiles = RemoveMapping(Psmiles)
                info["Rsmiles"] = Chem.MolToSmiles(NRsmiles)
                info["Psmiles"] = Chem.MolToSmiles(NPsmiles)
            else:
                NRsmiles = RemoveMapping(Rsmiles)
                info["Rsmiles"] = Rsmiles
            ###### PARSE ELEMENTS AND GET THE RELEVANT MATRICES 
            ###### CHECK THE CONSISTENCY OF ELEMENTS, IF THEY ARE NOT CONSISTENT, EXCLUDE THEM FROM THE GRAPH GENERATION.
            # Note for the future: We might need to revisit this and try to see if we can 'balance' them in some way.
            Reactant = Molecule(Rsmiles, self.args)
            Reactant.ReadMolecule()
            if not molecular:
                Product = Molecule(Psmiles, self.args)
                Product.ReadMolecule()
                # print('Reactant elements: ', Reactant.elements)
                # print('Product elements: ', Product.elements)

                #TODO: CHECK IF THIS IS CORRECT
                # if Reactant.elements != Product.elements:
                #     with open(os.path.join(self.data_path,'exclude.txt'),'a') as f:
                #         f.write(f'{Rind}\n')
                #         f.write(f'The reaction between {Rsmiles} and {Psmiles} failed because the element matrix is imbalanced. \n')
                #         print(f'The reaction between {Rsmiles} and {Psmiles} failed because the element matrix is imbalnaced. \n')
                #     return None

            elements = Reactant.elements
            
            ###### GENERATE THE RP-ADJACENCY MATRIX SO THAT AT LEAST ONE SIDE IS CONNECTED
            edges_u,edges_v  = [],[]
            for i in range(len(elements)):
                for j in range(len(elements)):
                    # if reaction, also check if P_adj >
                    Record_R = False; Record_P = False
                    if Reactant.adj[i][j] > 0:
                        Record_R = True
                    if not molecular:
                        if Product.adj[i][j] > 0:
                            Record_P = True
                    if Record_R or Record_P:
                        edges_u.append(i)
                        edges_v.append(j)
            Reactant.edges_u = edges_u
            Reactant.edges_v = edges_v
            if not molecular:
                Product.edges_u = edges_u
                Product.edges_v = edges_v
            
            ###### GET THE REACTIVE ATOMS AND BONDS
            if not molecular:
                bond_changes,reactive_atoms,bond_formed,bond_broken,bond_ochangeup,bond_ochangedown = return_reactive(elements,Reactant.bond_mat,Product.bond_mat)
                Reactant.reactive_atoms = reactive_atoms
                Product.reactive_atoms  = reactive_atoms
            try:
                Reactant.GenerateAtomFeature()
                Reactant.GenerateBondFeature()
                if not molecular:
                    Product.GenerateAtomFeature()
                    Product.GenerateBondFeature()
               
                # determine bond type #
                for ind in range(len(edges_u)):
                    ###### GET THE ATOM PAIRS FOR EACH BOND
                    edge_ind = sorted([edges_u[ind],edges_v[ind]])
                    edge_ind_mol = sorted([edges_u[ind]+1,edges_v[ind]+1])
                    BO_R = Reactant.bond_mat[edge_ind[0],edge_ind[1]]
                    if not molecular:
                        BO_P = Product.bond_mat[edge_ind[0],edge_ind[1]]
                    
                    ###### GET THE BOND TYPE
                    # the branch of isOldEdition
                    if not molecular:
                        if self.isOldEdition:
                            if BO_R == BO_P:
                                RBtype_str = 'T1'
                                PBtype_str = 'T1'
                            elif BO_R == 0.0:
                                RBtype_str = 'T4'
                                PBtype_str = 'T3'
                            elif BO_P == 0.0:
                                RBtype_str = 'T3'
                                PBtype_str = 'T4'
                            elif BO_R != BO_P and BO_R > 0:
                                RBtype_str = 'T2'
                                PBtype_str = 'T2'
                            RBtype = OLD_BOND_ENCODE[RBtype_str]
                            PBtype = OLD_BOND_ENCODE[PBtype_str]
                        else:
                            if BO_R == BO_P:
                                RBtype_str = 'T1'
                                PBtype_str = 'T1'
                            elif BO_R == 0.0:
                                RBtype_str = 'T4'
                                PBtype_str = 'T3'
                            elif BO_P == 0.0:
                                RBtype_str = 'T3'
                                PBtype_str = 'T4'
                            elif BO_R < BO_P and BO_R > 0:
                                RBtype_str = 'T2'
                                PBtype_str = 'T2'
                            elif BO_R > BO_P and BO_R > 0:
                                RBtype_str = 'T5'
                                PBtype_str = 'T5'
                            RBtype = bond_encode[RBtype_str]
                            PBtype = bond_encode[PBtype_str]

                        
                        #print(f"ind: {ind}, BO_R = {BO_R}, BO_P = {BO_P}, RBtype_str: {RBtype_str}, RBtype: {RBtype}\n")
                        #print(f"ind: {ind}, BO_R = {BO_R}, BO_P = {BO_P}, PBtype_str: {PBtype_str}, PBtype: {PBtype}\n")

                        Reactant.bond_F[ind] = RBtype + Reactant.bond_F[ind]
                        Product.bond_F[ind]  = PBtype + Product.bond_F[ind]
                        #print(f"Product.bond_F[ind]: {Product.bond_F[ind]}\n") 
                    # check if a bond is in reactant bond feature dict or not
                    ###### CHECK IF A BOND IS IN REACTANT BOND FEATURE DICT OR NOT 

                
                ###### ADD THE GLOBAL FEATURES USING DESCRIPTASTORUS
                if self.getRDKitFeatures or self.getRDKITNormfatures:
                    if self.getRDKitFeatures:
                        generator = rdDescriptors.RDKit2D()
                    else:
                        generator = rdNormalizedDescriptors.RDKit2DNormalized()
                    if not molecular:
                        smilist = [Rsmiles,Psmiles]
                    else:
                        smilist = [Rsmiles]
                    results = generator.process(smilist)
                    if results[0] is None:
                        with open(os.path.join(self.data_path,'exclude.txt'),'a') as f:
                            f.write('{}\n'.format(Rind))
                            f.write(f'The reaction between {Rsmiles} and {Psmiles} failed because the Additional Descriptor failed. \n')
                        return None 
                    elif results[0] is False:
                        with open(os.path.join(self.data_path,'exclude.txt'),'a') as f:
                            f.write('{}\n'.format(Rind))
                            f.write(f'The reaction between {Rsmiles} and {Psmiles} failed because the Additional Descriptor failed. \n')
                        return None 
                    else:
                        info['R_Addon'] = list(results[1])
                        if not molecular:
                            info['P_Addon'] = list(results[2])

                # pack info into one list
                info['u'] = edges_u
                #print(f"edges_u: {edges_u}\n")
                info['v'] = edges_v
                #print(f"Reactant.bond_F: {Reactant.bond_F}\n") 
                #print(f"Product.bond_F: {Product.bond_F}\n")
                #print(f"edges_v: {edges_v}\n")
                info['atom_F_R'] = Reactant.atom_F
                info['bond_F_R'] = Reactant.bond_F
                if not molecular:
                    info['atom_F_P'] = Product.atom_F
                    info['bond_F_P'] = Product.bond_F

                return info    
            except Exception as e:
                with open(os.path.join(self.data_path,'fail.txt'),'a') as ff:
                    ff.write('{}\n'.format(Rind))
                    if not molecular:
                        ff.write(f'INNER TRY: The reaction between {Rsmiles} and {Psmiles} failed because of {e} \n')
                        print(traceback.print_exc())
                        print(info)
                    else:
                        ff.write(f'INNER TRY: The reaction for {Rsmiles} failed because of {e} \n')
                        print(traceback.print_exc())
                        print(info)
                return {}
        except Exception as e:
            with open(os.path.join(self.data_path,'fail.txt'),'a') as ff:
                ff.write('{}\n'.format(Rind))
                if not molecular:
                    ff.write(f'OUTER TRY: The reaction between {Rsmiles} and {Psmiles} failed because of {e} \n')
                    print(traceback.print_exc())
                    print(info)
                else:
                    ff.write(f'OUTER TRY: The reaction for {Rsmiles} failed because of {e} \n')
                    print(f'OUTER TRY: The reaction for {Rsmiles} failed because of {e} \n')
                    print(traceback.print_exc())
                    print(info)
            return {}
            
    def __getitem__(self, index):
        

        ###### LOAD DATAPATH 
        info = self.convertSMILEStoEGATGraph(index, self.method_mapping, self.num_workers)
        try:            
            ###### CREATE GRAPH
            u, v = torch.Tensor(info['u']).int(),torch.Tensor(info['v']).int()
            gR   = dgl.graph((u,v))
            if not self.molecular: 
                gP   = dgl.graph((u,v))
            #print(f"graphR: #node: {gR.num_nodes()}, #edge: {gR.num_edges()}\n")
            ###### APPEND FEATURES
            #print(f"info['atom_F_R']: {info['atom_F_R']}\n")
            #print(f"info['bond_F_R']: {info['bond_F_R']}\n")
            if not self.molecular :
                gR.ndata['x'] = torch.Tensor(info['atom_F_R'])
                gR.edata['x'] = torch.Tensor(info['bond_F_R'])
                gP.ndata['x'] = torch.Tensor(info['atom_F_P'])
                gP.edata['x'] = torch.Tensor(info['bond_F_P'])

                #if gR.ndata['x'].max() > 50 or gP.ndata['x'].max() > 50:
                #    print(f"max in node data: {gR.ndata['x'].max()}\n")
                    #with open('check.txt','a') as ff:
                    #    ff.write('{}\n'.format(self.root + '--'+str(index)))
            else:
                #print(torch.Tensor(info['atom_F_R']).shape)
                
                try:
                    gR.ndata['x'] = torch.Tensor(info['atom_F_R'])
                    gR.edata['x'] = torch.Tensor(info['bond_F_R'])
                except:
                    gR.ndata['x'] = torch.Tensor(info['atom_F_R'])
                    gR.edata['x'] = torch.Tensor(info['bond_F_R'])
                    
                #if gR.ndata['x'].max() > 50: 
                #    with open('check.txt','a') as ff:
                #        ff.write('{}\n'.format(self.root + '--'+index))
            #print(f"graphR after appending features: #node: {gR.num_nodes()}, #edge: {gR.num_edges()}\n")
            #print(f"gR.ndata['x'], len_node_f: {len(gR.ndata['x'])}, len_edge_f: {len(gR.edata['x'])}\n")
            #print(f"gR.ndata['x'], shape node feature: {gR.ndata['x'].shape}\n")
            #print(f"gP.ndata['x'], shape node feature: {gP.ndata['x'].shape}\n")
            
            #print(f"node feature for gR = {gR.ndata['x']}\n")
            ###### MAKE THE OUTPUT TENSORS
            # Add name of the file and the reaction class it inherits
            samples = [info['Indices']]
            if self.args.AtomPropPrediction:
                retrieved_node,chosen_index = select_node_features(gR)
                # Corrupt the node in original graph
                gR = mask_node_features(gR,chosen_index)
                # Show corrupted node
                samples += [gR,retrieved_node]
                if not self.molecular: 
                    retrieved_nodeP,chosen_indexP = select_node_features(gP)
                    gP = mask_node_features(gP,chosen_indexP)
                    samples += [gP,retrieved_nodeP]
                    # Corrupt the node in original graph
                    gP = mask_node_features(gP,chosen_indexP)
            elif self.args.use_corrupt_embedding:
                original_graph = gR
                retrieved_node,chosen_index = select_node_features(gR)
                # Corrupt the node in original graph
                corrupted_graph = mask_node_features(gR,chosen_index)
                samples += [original_graph,corrupted_graph]
            else:# Add the graph
                if not self.molecular:
                    samples += [gR,gP]
                else:
                    samples += [gR]
            #print(list(info.keys()))
            # Add what the SMILES and Inchi keys are
            if not self.molecular :
                samples += [[info['Rsmiles'],info['Psmiles'],info['Rinchi'],info['Pinchi']]]
            else:
                samples += [[info['Rsmiles'],info['Rinchi']]]

            
            # Add what the Additional Tensor is if needed
            if self.additional is not None:
                additionaltensor = []
                if isinstance(self.additional,list) or isinstance(self.additional,omegaconf.listconfig.ListConfig):
                    additionaltensor += [float(info[output]) for output in self.additional]
                else:
                    additionaltensor += [float(info[self.additional])]
                additionaltensor = torch.Tensor(additionaltensor)
                samples += [additionaltensor]

            # Add what the Add-Ons are
            if self.addons:
                if not self.molecular :
                    samples += [torch.Tensor(info['R_Addon']),torch.Tensor(info['P_Addon'])]
                else:
                    samples += [torch.Tensor(info['R_Addon'])]
            return samples
        except Exception as e:
            print(self.root + '--'+ str(index), f'failed because of {e}')
            print(traceback.print_exc())
            print(info)
            return None
    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    config_file = "/groups/bsavoie2/bpiguave/EGAT-JEPA/EGAT_YARP/REACTION.yaml"
    with open(config_file, 'r') as f:
        config = omegaconf.OmegaConf.load(f)
    omegaconf.OmegaConf.set_struct(config, False)
    dataset_path = '/groups/bsavoie2/bpiguave/EGAT-JEPA/EGAT_YARP/formatted_smiles.csv'
    df = pd.read_csv(dataset_path)
    print(df.iloc[1,:])
    dataset = FastDataset(config, dataset=dataset_path)
    print(dataset.__getitem__(0))