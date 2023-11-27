import os,sys
import numpy as np
import logging
import pickle
import time
from copy import deepcopy
from collections import Counter
import multiprocessing as mp
from multiprocessing import Queue
from logging.handlers import QueueHandler
from concurrent.futures import ProcessPoolExecutor, TimeoutError

from ase import io
from ase.build import minimize_rotation_and_translation
from scipy.spatial.distance import cdist
from xgboost import XGBClassifier

# eventually this will be replaced by yarp.wrappers
import yarp as yp
from yarp.input_parsers import xyz_parse
from yarp.find_lewis import find_lewis
from yarp.taffi_functions import xyz_write, table_generator,graph_seps
from wrappers.xtb import XTB
from utils import *
from constants import Constants

def check_dup_ts_pysis(tsopt_job_list, logger):
    ''' 
    For a given list of pysis ts optimization jobs, check and exclude duplicated TSs
    '''
    keep_list = []
    reactions = dict()
    for tsopt_job in tsopt_job_list:
        # check if tsopt job is finished
        if tsopt_job.calculation_terminated_normally() is False:
            logger.info(f"TSopt job {tsopt_job.jobname} fails, skip this reaction...")
            print(f"TSopt job {tsopt_job.jobname} fails, skip this reaction...")
            continue

        # check if this is a true ts
        if tsopt_job.is_true_ts() is False:
            logger.info(f"TSopt job {tsopt_job.jobname} fails to locate a true transition state, skip this reaction...")
            print(f"TSopt job {tsopt_job.jobname} fails to locate a true transition state, skip this reaction...")
            continue

        # parse output files
        rxn_ind = tsopt_job.jobname
        rxn_name= '_'.join(rxn_ind.split('_')[:-1])
        TSenergy= tsopt_job.get_energy()
        E,G     = tsopt_job.get_final_ts()
        
        # check duplicated TSs
        if rxn_name not in reactions:
            reactions[rxn_name] = [TSenergy]
            keep_list.append(tsopt_job)
        else:
            min_ene_diff = min([Constants.ha2kcalmol*abs(TSenergy-ene) for ene in reactions[rxn_name]])
            if min_ene_diff < 0.05:
                logger.info(f"TSopt job {tsopt_job.jobname} locates a duplicated transition state, skip this reaction...")
                print(f"TSopt job {tsopt_job.jobname} locates a duplicated transition state, skip this reaction...")
            else:
                reactions[rxn_name].append(TSenergy)
                keep_list.append(tsopt_job)
            
    return keep_list

def analyze_outputs(working_folder,irc_job_list,logger,charge=0,dg_thresh=None,nc_thresd=5,uncertainty=5,select='tight',return_D2_dict=False):
    '''
    Analyze the first stage YARP calculation, find intended&unintended reactions, etc
    Input: dg_thresh -- threshold for activation energy, above which no further DFT analysis will be needed (default: None)
           nc_thresd -- maximum number of conformations selected for DFT analysis (default: 5)
           uncertainty -- trust region of low-level (e.g., GFN2-xTB) calculations
           working_folder -- output files, including IRC_record.txt, selected_tss.txt, will be stored in this folder
           irc_job_list -- a list of irc jobs
           return_D2_dict -- if this is True, an additional dictionary for Delta2 model will be generated
    '''
    # initialize output dictionary
    reactions  = dict()
    conf_index = dict()
    select_rxns= dict()

    # create D2 dict if is applied
    if return_D2_dict: D2_rxns = dict()

    # create record.txt to write the IRC result
    with open(f'{working_folder}/IRC-record.txt','w') as g:
        g.write(f'{"reaction":40s} {"R":<60s} {"P":<60s} {"type":<15s} {"barrier":<10s}\n')

    # loop over IRC output files
    for irc_job in irc_job_list:
        # check job status
        if irc_job.calculation_terminated_normally() is False:
            logger.info(f"IRC job {irc_job.jobname} fails, skip this reaction")
            print(f"IRC job {irc_job.jobname} fails, skip this reaction")
            continue

        # obtain irc output
        job_success = False
        rxn_ind = irc_job.jobname
        rxn_name= '_'.join(rxn_ind.split('_')[:-1])
        try:
            E, G1, G2, TSG, barrier1, barrier2 = irc_job.analyze_IRC()
            _,TSE,_ = irc_job.get_energies_from_IRC()
            barriers = [barrier2, barrier1]
            job_success = True
        except:
            pass

        if job_success is False: continue

        # obtain the original input reaction
        input_xyz = f"{working_folder}/rxn_conf/{rxn_ind}.xyz"
        [[_, RG], [_, PG]] = xyz_parse(input_xyz, multiple=True)

        # compute adjacency matrix and bond matrix
        adjmat_1, adjmat_2, R_adjmat, P_adjmat = table_generator(E, G1), table_generator(E, G2), table_generator(E, RG), table_generator(E, PG)
        bond_mats_1, _ = find_lewis(E, adjmat_1, charge)
        bond_mats_2, _ = find_lewis(E, adjmat_2, charge)

        # obtain smiles
        mol1=yp.yarpecule(E, G1)
        mol2=yp.yarpecule(E, G2)
        smiles = [return_smi(mol1, namespace=f'{rxn_ind}-1'), return_smi(mol2, namespace=f'{rxn_ind}-2')]

        # compare adj_mats
        adj_diff_1r = np.abs(adjmat_1 - R_adjmat)
        adj_diff_1p = np.abs(adjmat_1 - P_adjmat)
        adj_diff_2r = np.abs(adjmat_2 - R_adjmat)
        adj_diff_2p = np.abs(adjmat_2 - P_adjmat)

        # match two IRC end nodes
        rtype = 'Unintended'
        node_map = {'R': None, 'P': None}
        if adj_diff_1r.sum() == 0:
            if adj_diff_2p.sum() == 0:
                rtype = 'Intended'
                node_map = {'R': 1, 'P': 2}
            else:
                rtype = 'P_Unintended'
                node_map = {'R': 1, 'P': 2}
        elif adj_diff_1p.sum() == 0:
            if adj_diff_2r.sum() == 0:
                rtype = 'Intended'
                node_map = {'R': 2, 'P': 1}
            else:
                rtype = 'R_Unintended'
                node_map = {'R': 2, 'P': 1}
        elif adj_diff_2r.sum() == 0:
            rtype = 'P_Unintended'
            node_map = {'R': 2, 'P': 1}
        elif adj_diff_2p.sum() == 0:
            rtype = 'R_Unintended'
            node_map = {'R': 1, 'P': 2}

        # analyze outputs
        if rtype == 'Intended':
            rsmiles = smiles[node_map['R']-1]
            psmiles = smiles[node_map['P']-1]
            barrier = barriers[node_map['R']-1]
        elif rtype == 'P_Unintended':
            rsmiles = smiles[node_map['R']-1]
            psmiles = smiles[node_map['P']-1]
            barrier = barriers[node_map['R']-1]
        elif rtype == 'R_Unintended':
            rsmiles = smiles[node_map['R']-1]
            psmiles = smiles[node_map['P']-1]
            barrier = barriers[node_map['P']-1]
        else:
            rsmiles = smiles[0]
            psmiles = smiles[1]
            barrier = None

        ### select based on input settings
        rxn_smi_r = f'{rsmiles}>>{psmiles}'
        rxn_smi_p = f'{psmiles}>>{rsmiles}'
        reactions[rxn_ind] = {'rxn':rxn_smi_r, 'barrier':barrier, 'TS_SPE': TSE, 'rtype':rtype, 'reactant':rsmiles, 'product':psmiles, 'select':1, \
                              'E':E, 'RG':RG, 'PG':PG, 'TSG':TSG, 'RPset':1}

        # select based on the using scenario
        if select == 'network':
            if rtype not in ['Intended', 'P_Unintended']:
                reactions[rxn_ind]['select'] = 0
                continue
        elif select == 'tight':
            if rtype != 'Intended':
                reactions[rxn_ind]['select'] = 0
                continue
        else:
            if rtype not in ['Intended', 'R_Unintended', 'P_Unintended']:
                reactions[rxn_ind]['select'] = 0
                continue

        if rtype != 'Unintended' and return_D2_dict and barrier > 0:
            D2_rxns[rxn_ind] = {'TS_SPE': TSE, 'reactant':smiles[node_map['R']-1], 'product':smiles[node_map['P']-1], 'E': E, 'RG': [G1,G2][node_map['R']-1], 'PG': [G1,G2][node_map['P']-1], 'TSG': TSG}

        # for the selected reactions, add additional set of reference reaction if this rxn is not intended
        if reactions[rxn_ind]['select'] == 1 and rtype != 'Intended':
            reactions[rxn_ind]['RPset'] = 2
            reactions[rxn_ind]['RG_irc'] = [G1,G2][node_map['R']-1]
            reactions[rxn_ind]['PG_irc'] = [G1,G2][node_map['P']-1]

        # in case IRC calculation is wrong, print out warning information and ignore this rxn
        if reactions[rxn_ind]['select'] and barrier < 0:
            print(f"Reaction {irc_job.jobname} has a barrier less than 0, which indicates the end node optimization of IRC has some trouble, please manually check this reaction")
            logger.info(f"Reaction {irc_job.jobname} has a barrier less than 0, which indicates the end node optimization of IRC has some trouble, please manually check this reaction")
            reactions[rxn_ind]['select'] = 0
            continue

        # if there is a dg_thresh, the barrier needs to be smaller than the threshold plus the soft margin to be selected
        if dg_thresh is not None and barrier > dg_thresh + uncertainty:
            print(f"Reaction {irc_job.jobname} has a barrier of {barrier} kcal/mol which is higher than the barrier threshold")
            logger.info(f"Reaction {irc_job.jobname} has a barrier of {barrier} kcal/mol which is higher than the barrier threshold")
            reactions[rxn_ind]['select'] = 0
            continue

        # append rxn_index into conf_index
        if rxn_smi_r not in conf_index.keys() and rxn_smi_p not in conf_index.keys():
            conf_index[rxn_smi_r] = [rxn_ind]
        elif rxn_smi_r in conf_index.keys():
            conf_index[rxn_smi_r].append(rxn_ind)
        else:
            conf_index[rxn_smi_p].append(rxn_ind)
            reactions[rxn_ind]['rxn'] = rxn_smi_p

    # keep at most nc_thresd rxns for further calculations
    for rxn, rxn_inds in conf_index.items():
        if len(rxn_inds) > nc_thresd:
            TSE_thresh = sorted([reactions[rxn_ind]['TS_SPE'] for rxn_ind in rxn_inds])[nc_thresd]
            select_ind = sorted([rxn_ind for rxn_ind in rxn_inds if reactions[rxn_ind]['TS_SPE'] < TSE_thresh])
        else:
            select_ind = sorted(rxn_inds)
        # select rxns in the finial dictionary
        for ind in select_ind: select_rxns[ind] = reactions[ind]

    # write down reaction informations into IRC output file
    for rxn_ind in sorted(reactions.keys()):
        rxn = reactions[rxn_ind]
        with open(f'{working_folder}/IRC-record.txt','a') as g:
            g.write(f'{rxn_ind:40s} {rxn["reactant"]:60s} {rxn["product"]:60s} {rxn["rtype"]:15s} {str(rxn["barrier"]):10s}\n')

    if return_D2_dict:
        return select_rxns, D2_rxns
    else:
        return select_rxns

def apply_D2(data,ene_dict,LOT='DFT',logger=False):
    ''' 
    Perform Delta^2 ML model on the xTB optimized reactant/product and transition state to obtain more accurate energy predictions
    two levels of theory (LOT) are available: DFT (B3LYP-D3/TZVP) or Gaussian-4
    Based on the traning data of this Delta^2 model, the applicable range are
    1. within 10 heavy atoms (no more than 12)
    2. must be neutral, closed shell system
    3. containing C,H,O,N-only elements
    '''
    ##### Step 1: build up D2 model
    from ML import EnsembledModel
    import torch
    # search for device
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu' # for safer usage, we force the model to be applied on cpu
    start = time.time()

    # load in models
    model_list = []
    if LOT == 'G4':
        print("Gaussian-4 (G4) models are loaded")
        if logger: logger.info("Gaussian-4 (G4) models are loaded")
        sae = {1: -0.09364398454577838, 6: -35.95667813236401, 7: -51.81919960387285, 8: -71.14043670055156}
        prefix = '/'.join(os.path.abspath(__file__).split("/")[:-1])+'/bin/G4_models'
    elif LOT == 'DFT':
        print("DFT (B3LYP-D3/TZVP) models are loaded")
        if logger: logger.info("DFT (B3LYP-D3/TZVP) models are loaded")
        sae = {1: -0.09565701919376791, 6: -35.97628109983475, 7: -51.84891086964016, 8: -71.18551920716808}
        prefix = '/'.join(os.path.abspath(__file__).split("/")[:-1])+'/bin/DFT_models'
    else:
        print("DFT (B3LYP-D3/TZVP) Gibbs Free Energy models are loaded")
        if logger: logger.info("DFT (B3LYP-D3/TZVP) Gibbs Free Energy models are loaded")
        sae = {1: -0.08499769224879117, 6: -35.97471868936887, 7: -51.84838797019447, 8: -71.18618243177728}
        prefix = '/'.join(os.path.abspath(__file__).split("/")[:-1])+'/bin/DG_models'
        
    # search for jpt files
    jpts = sorted([os.path.join(prefix, filename) for filename in os.listdir(prefix) if filename[-4:]==".jpt"])

    # construct the ensemble model
    for i in jpts:
        model = torch.jit.load(i, map_location=device)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        model_list.append(model)

    ens_model = EnsembledModel(model_list, x=['coord', 'numbers', 'charge'], out=['energy'], detach=False)
    ens_model = torch.jit.script(ens_model).to(device)
    print(f"Successfully loaded the Delta^2 ML model, used time: {time.time()-start} (s)")
    if logger: logger.info(f"Successfully loaded the Delta^2 ML model, used time: {time.time()-start} (s)")
    start = time.time()

    ##### Step 2: run Delta^2 model on geometries
    pred_enes = dict()
    for k,v in data.items():
    
        # prepare input format
        coord = torch.tensor(v['coord'],requires_grad=False,dtype=torch.float32,device=device)
        numbers = torch.tensor(v['numbers'],requires_grad=False,dtype=torch.int64,device=device)
        charge = torch.zeros(len(v['charge']),device=device) # SHAPE IS BATCH
        _in  = dict(coord=coord, numbers=numbers,charge=charge)
        
        # run ensemble prediction
        _out = ens_model(_in)
        
        # search for xTB energies
        xtb_enes = torch.tensor(np.array([ene_dict[name.decode('utf8')] for name in v['_id']]),dtype=torch.float64,device=device)

        # compute correction term
        corr = torch.tensor(np.array([sum([sae[j] for j in number]) for number in v['numbers']]),dtype=torch.float64,device=device)

        # obtain mean and std of ensemble model
        E_mean = corr + xtb_enes + _out['energy']
        E_std  = _out['energy_std']

        # save predictions
        for i in range(len(v['_id'])):
            pred_enes[v['_id'][i].decode('utf8')] = {'E_pred': E_mean[i].detach().cpu().numpy(), 'E_std': E_std[i].detach().cpu().numpy()}

    print(f"Successfully running the Delta^2 model, used time: {time.time()-start} (s)")
    if logger: logger.info(f"Successfully running the Delta^2 ML model, used time: {time.time()-start} (s)")

    return pred_enes

