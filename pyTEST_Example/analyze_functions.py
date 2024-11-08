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

def analyze_outputs(rxns):
    scratch=rxns[0].args["scratch"]
    nc_thresd=rxns[0].args["nconf_dft"]
    charge=rxns[0].args["charge"]
    select=rxns[0].args["select"]
    g=open(f'{scratch}/IRC-record.txt', 'w')
    g.write(f'{"reaction":40s} {"R":<60s} {"P":<60s} {"type":<15s} {"barrier":10s}\n')
    for count, rxn in enumerate(rxns):
        key=[i for i in rxn.IRC_xtb.keys()]
        for conf_i in key:
            rxn_ind=f'{rxn.reactant_inchi}_{rxn.id}_{conf_i}'
            print(rxn_ind)
            barrier=rxn.IRC_xtb[conf_i]["barriers"][0]
            RG=rxn.IRC_xtb[conf_i]["node"][0]
            PG=rxn.IRC_xtb[conf_i]["node"][1]
            Type=rxn.IRC_xtb[conf_i]["type"]
            R_adj=table_generator(rxn.reactant.elements, RG)
            P_adj=table_generator(rxn.reactant.elements, PG)
            R_bondmat, _=find_lewis(rxn.reactant.elements, R_adj, charge)
            P_bondmat, _=find_lewis(rxn.reactant.elements, P_adj, charge)
            R_smi=return_smi(rxn.reactant.elements, RG, R_bondmat[0])
            P_smi=return_smi(rxn.reactant.elements, PG, P_bondmat[0])
            rxns[count].IRC_xtb[conf_i]['smiles']=[R_smi, P_smi]
            rxns[count].IRC_xtb[conf_i]['select']=1
            if select=='network':
                if Type not in ["intended", "P_unitended"]: rxns[count].IRC_xtb[conf_i]['select']=0
            elif select=='tight':
                if Type != 'intended': rxns[count].IRC_xtb[conf_i]['select']=0
            else:
                if Type not in ["intended", "R_unintended", "P_unintended"]:
                    rxns[count].IRC_xtb[conf_i]['select']=0
            g.write(f'{rxn_ind:40s} {R_smi:60s} {P_smi:60s} {Type:15s} {barrier:10f}\n')
    
    return rxns

def apply_IRC_model(rxns):
    args=rxns[0].args
    IRC_model=XGBClassifier()
    IRC_model.load_model(os.path.join(args["model_path"], "IRC_model.json"))
    for count, rxn in enumerate(rxns):
        key=[i for i in rxn.TS_dft.keys()]
        for conf_i in key:
            rxns[count]=predict_TS(rxn, conf_i, IRC_model)
    return rxns

def predict_TS(rxn, conf_i, IRC_model):
    E=rxn.reactant.elements
    RG=rxn.reactant.geo
    PG=rxn.product.geo
    Radj=rxn.reactant.adj_mat
    Padj=rxn.product.adj_mat
    intend_prob=TS_prediction(rxn, conf_i, IRC_model)
    thermal=rxn.TS_dft[conf_i]["thermal"]
    SPE=rxn.TS_dft[conf_i]["SPE"]
    barrier=rxn.IRC_xtb[conf_i]["barriers"][0]
    if max(intend_prob)<0.5: rxn.IRC_xtb[conf_i]["RP"]=[False, 0]
    else:
        if intend_prob[0]>intend_prob[1]:
            rxn.IRC_xtb[conf_i]["RP"]=[True, 0]
        else:
            rxn.IRC_xtb[conf_i]["RP"]=[True, 1]

    return rxn
def TS_prediction(rxn, conf_i, IRC_model):
    Radii=yp.el_radii
    E=rxn.reactant.elements
    Radj=rxn.reactant.adj_mat
    Padj=rxn.product.adj_mat
    TS_G=rxn.TS_dft[conf_i]["geo"]
    imag_mode=rxn.TS_dft[conf_i]["imag_mode"]
    bond_break, bond_form=[], []
    adj_change=Padj-Radj
    for i in range(len(E)):
        for j in range(i+1, len(E)):
            if adj_change[i][j]==-1: bond_break+=[(i, j)]
            if adj_change[i][j]==1: bond_form+=[(i, j)]
    center_atoms=list(set(sum(bond_break, ())+sum(bond_from, ())))
    if len(center_atoms)==0: return [0, 0]

    TS_f=TS_G+imag_mode*0.5
    TS_b=TS_G-imag_mode*0.5

    bonds=bond_break+bond_form
    dist=0
    for bond in bonds: dist=max(dist, np.linalg.norm(TS_G[bond[0]]-TS_G[bond[1]]))

    P_bonds=[]
    Dist_Mat=np.triu(cdist(TS_G, TS_G))
    x_ind, y_ind=np.where((Dist_Mat>0.0) & (Dist_Mat<max([Radii[i]**2.0 for i in Radii.keys()])))
    for count, i in enumerate(x_ind):
        if Dist_Mat[i, y_ind[count]] < (Radii[E[i]]+Radii[E[y_ind[count]]])*1.8:
            P_bonds+=[(i, y_ind[count])]

    movement=[(i, np.linalg.norm(imag_mode[i])) for i in range(len(E))]
    dis_change=[]
    for bond in P_bonds:
        dis=abs(np.linalg.norm(TS_f[bond[0]]-TS_f[bond[1]])-np.linalg.norm(TS_b[bond[0]]-TS_b[bond[1]]))
        dis_change+=[((bond[0], bond[1]), dis)]
    dis_change.sort(key=lambda x: -x[-1])
    dis_ind=[ind[0] for ind in dis_change]
    index=[]
    target=[]
    for pair in bond_break+bond_form:
        if tuple(sorted(pair)) in dis_ind:
            index+=[dis_ind.index(tuple(sorted(pair)))]
            target+=[dis_change[index[-1]][-1]]
        else:
            index+=[len(dis_ind)]
            target==[0]
    try:
        freq_unexpect=sum([ind[-1] for count_i,ind in enumerate(dis_change[:max(index)]) if count_i not in index])
    except:
        freq_unexpect = [0]
    prob1 = IRCmodel.predict_proba(np.array([[np.mean(target),min(target),freq_unexpect,dist]]))[0][1]

    if rxn.IRC_xtb[conf_i]["select"]==1 and rxn.IRC_xtb[conf_i]["type"]!="intended": return [prob1, 0]
    
    Radj=table_generator(E, rxn.IRC_xtb[conf_i]["node"][0])
    Padj=table_generator(E, rxn.IRC_xtb[conf_i]["node"][-1])
    bond_break, bond_form=[], []
    adj_change=Padj-Radj
    
    for i in range(len(E)):
        for j in range(i+1, len(E)):
            if adj_change[i][j]==-1: bond_break+=[(i, j)]
            if adj_change[i][j]==1: bond_form+=[(i, j)]
    center_atoms=list(set(sum(bond_break, ())+sum(bond_from, ())))
    if len(center_atoms)==0: return [0, 0]
    
    bonds=bond_break+bond_form
    dist=0
    for bond in bonds: dist=max(dist, np.linalg.norm(TS_G[bond[0]]-TS_G[bond[1]]))

    P_bonds=[]
    Dist_Mat=np.triu(cdist(TS_G, TS_G))
    x_ind, y_ind=np.where((Dist_Mat>0.0) & (Dist_Mat<max([Radii[i]**2.0 for i in Radii.keys()])))
    for count, i in enumerate(x_ind):
        if Dist_Mat[i, y_ind[count]] < (Radii[E[i]]+Radii[E[y_ind[count]]])*1.8:
            P_bonds+=[(i, y_ind[count])]

    movement=[(i, np.linalg.norm(imag_mode[i])) for i in range(len(E))]
    dis_change=[]
    for bond in P_bonds:
        dis=abs(np.linalg.norm(TS_f[bond[0]]-TS_f[bond[1]])-np.linalg.norm(TS_b[bond[0]]-TS_b[bond[1]]))
        dis_change+=[((bond[0], bond[1]), dis)]
    dis_change.sort(key=lambda x: -x[-1])
    dis_ind=[ind[0] for ind in dis_change]
    index=[]
    target=[]
    for pair in bond_break+bond_form:
        if tuple(sorted(pair)) in dis_ind:
            index+=[dis_ind.index(tuple(sorted(pair)))]
            target+=[dis_change[index[-1]][-1]]
        else:
            index+=[len(dis_ind)]
            target==[0]
    try:
        freq_unexpect=sum([ind[-1] for count_i,ind in enumerate(dis_change[:max(index)]) if count_i not in index])
    except:
        freq_unexpect = [0]
    prob2 = IRCmodel.predict_proba(np.array([[np.mean(target),min(target),freq_unexpect,dist]]))[0][1]
    
    return [prob1, prob2]

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

