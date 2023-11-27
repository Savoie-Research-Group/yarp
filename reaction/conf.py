import os,sys
import numpy as np
import logging
import pickle
import time
import pandas as pd
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
from wrappers.xtb import XTB
from utils import *

from yarp.taffi_functions import table_generator, xyz_write
from yarp.find_lewis import find_lewis

def generate_rxn_conf(input_data_list):
    for input_data in input_data_list:
        rxn, logging_queue = input_data
        rxn.rxn_conf_generation(logging_queue)
    return rxn
"""
def generate_rxn_conf(input_data_list):
    for input_data in input_data_list:
        rxn, job_id, args, logging_queue = input_data
        RG=rxn.reactant.geo
        RE=rxn.reactant.elements
        PG=rxn.product.geo
        PE=rxn.product.elements
        R_adj=rxn.reactant.adj_mat
        P_adj=rxn.product.adj_mat
        R_bond_mats=rxn.reactant.bond_mats
        P_bond_mats=rxn.product.bond_mats
        conf_rxn=dict()
        count=0
        # Create a dictionary to store the conformers and product/reactant bond mat.
        if args['strategy']!=0:
            for i in rxn.product_conf.keys():
                conf_rxn[count]={"E": RE, 'bond_mat_r': R_bond_mats[0], "G": deepcopy(rxn.product_conf[i]), 'direct': "B"}
                count=count+1
        if args['strategy']!=1:
            for i in rxn.reactant_conf.keys():
                conf_rxn[count]={"E": RE, "bond_mat_r": P_bond_mats[0], "G": deepcopy(rxn.reactant_conf[i]), 'direct': "F"}
        # load ML model to find conformers 
        if len(conf_rxn)>3*args['n_conf']: model=pickle.load(open(os.path.join(args['model_path'],'rich_model.sav'), 'rb'))
        else: model=pickle.load(open(os.path.join(args['model_path'],'poor_model.sav'), 'rb'))

        ind_list, pass_obj_values=[], []
        for conf_ind, conf_entry in conf_rxn.items():
            # apply force-field optimization
            Gr = opt_geo(conf_entry['E'],conf_entry['G'],conf_entry['bond_mat_r'],ff=args['ff'],step=100,filename=f'tmp_{job_id}')
            if len(Gr)==0: continue
            tmp_xyz_p = f"{args['scratch_xtb']}/{job_id}_p.xyz"
            xyz_write(tmp_xyz_p,conf_entry['E'],Gr)
            tmp_xyz_r = f"{args['scratch_xtb']}/{job_id}_r.xyz"
            xyz_write(tmp_xyz_r,conf_entry['E'],conf_entry['G'])
            # calculate indicator
            indicators = return_indicator(conf_entry['E'],conf_entry['G'],Gr,namespace=f'tmp_{job_id}')
            reactant=io.read(tmp_xyz_r)
            product=io.read(tmp_xyz_p)
            minimize_rotation_and_translation(reactant,product)
            io.write(tmp_xyz_p,product)
            _,Gr_opt = xyz_parse(tmp_xyz_p)
            indicators_opt = return_indicator(conf_entry['E'],conf_entry['G'],Gr_opt,namespace=f'tmp_{job_id}')

            # if applying ase minimize_rotation_and_translation will increase the intended probability, use the rotated geometry
            if model.predict_proba(indicators)[0][1] < model.predict_proba(indicators_opt)[0][1]: indicators, Gr = indicators_opt, Gr_opt
            # check whether the channel is classified as intended and check uniqueness
            if model.predict_proba(indicators)[0][1] > 0.4 and check_duplicate(indicators,ind_list,thresh=0.025):
                ind_list.append(indicators)
                pass_obj_values.append((model.predict_proba(indicators)[0][0],deepcopy(conf_entry['G']),Gr,deepcopy(conf_entry['direct'])))

            # remove tmp file
            if os.path.isfile(tmp_xyz_r): os.remove(tmp_xyz_r)
            if os.path.isfile(tmp_xyz_p): os.remove(tmp_xyz_p)

        # generate conformers based on prob
        pass_obj_values = sorted(pass_obj_values, key=lambda x: x[0])
        N_conf = 0
        for count, item in enumerate(pass_obj_values):
            input_type = item[3]
            tmp_xyz_r = f"{args['scratch_xtb']}/{job_id}_r.xyz"
            tmp_xyz_p = f"{args['scratch_xtb']}/{job_id}_p.xyz"
            xyz_write(tmp_xyz_r, RE, item[1])
            xyz_write(tmp_xyz_p, RE, item[2])
            if args['opt']:
                if args['low_solvation']:
                    solvation_model, solvent = args['low_solvation'].split('/')
                    optjob = XTB(input_geo=tmp_xyz_p,work_folder=args['scratch_xtb'],jobtype=['opt'],jobname=f'opt_{job_id}_p',solvent=solvent,\
                                 solvation_model=solvation_model,charge=args['charge'],multiplicity=args['multiplicity'])
                else:
                    optjob = XTB(input_geo=tmp_xyz_p,work_folder=args['scratch_xtb'],jobtype=['opt'],jobname=f'opt_{job_id}_p',charge=args['charge'],multiplicity=args['multiplicity'])

                optjob.execute()

                if optjob.optimization_success():
                    _, Gr = optjob.get_final_structure()
                else:
                    #logger.info(f"xtb geometry optimization fails for the other end of {job_id} (conf: {conf_ind}), will use force-field optimized geometry for instead")
                    #Gr = item[2]
                    continue

                xyz_write(tmp_xyz_p,conf_entry['E'],Gr)

            if input_type=='F':
                _, rg=xyz_parse(tmp_xyz_r)
                _, pg=xyz_parse(tmp_xyz_p)
                rxn.rxn_conf[N_conf]={"R": rg, "P": pg}
                os.system(f"cp {tmp_xyz_r} {args['conf_output']}/{job_id}_{N_conf}.xyz; cat {tmp_xyz_p} >> {args['conf_output']}/{job_id}_{N_conf}.xyz;rm {tmp_xyz_r} {tmp_xyz_p}")
            else:
                _, rg=xyz_parse(tmp_xyz_p)
                _, pg=xyz_parse(tmp_xyz_r)
                rxn.rxn_conf[N_conf]={"R": rg, "P": pg}
                os.system(f"cp {tmp_xyz_p} {args['conf_output']}/{job_id}_{N_conf}.xyz; cat {tmp_xyz_r} >> {args['conf_output']}/{job_id}_{N_conf}.xyz;rm {tmp_xyz_r} {tmp_xyz_p}")
            N_conf=N_conf+1
            if N_conf>=args["n_conf"]: break

        if len(pass_obj_values) == 0:
            print(f"WARNING: None of the reaction conformation can be generated for the input reaction {job_id}. please check this reaction to make sure it is a vaild one")

        # add a joint-opt alignment if too few alignments pass the criteria
        # will add soon
    return
"""
def return_indicator(E,RG,PG,namespace='node'):
    '''
    Function to find indicators for reactant-product alignments
    Input:
          E:   elements
          RG:  reactant geometry
          RG:  reactant geometry
    Output:
          RMSD: mass-weighted RMSD between reactant and product, threshold < 1.6
          max_dis:  maximum bond length change between non-H atoms, threshold < 4.0
          min_cross_dis: shorted distance between atoms' path (non-H atoms) to original bonds, threshold > 0.6
          path_cross: if all atoms involved in bond changes are non-H, path_cross refers to the distance between two bond changes, threshold > 0.6
          max_Hdis: maximum bond length change if contains H, threshold < 4.5 (* optional)
          min_Hcross_dis: shorted distance between atoms' path (H atoms involves) to original bonds, threshold > 0.4 (* optional)
          h = RMSD/1.6 + max_dis/4.0 + 0.6/min_cross_dis + 0.6/path_cross + 0.5 * max_Hdis/4.5 + 0.1/min_cross_dis
    '''

    # calculate adj_mat
    Radj = table_generator(E, RG)
    Padj = table_generator(E, PG)

    # determine bond changes
    bond_break, bond_form=[], []
    del_adj = Padj - Radj
    for i in range(len(E)):
        for j in range(i+1, len(E)):
            if del_adj[i][j]==-1: bond_break+=[(i, j)]
            if del_adj[i][j]==1: bond_form+=[(i, j)]
    # identify hydrogen atoms, atoms involved in the reactions
    H_index=[i for i, e in enumerate(E) if e=='H']
    involve=list(set(list(sum(bond_break+bond_form, ()))))

    # create observed segments
    bond_seg={i:[] for i in bond_break+bond_form}
    for bond in bond_break:
        bond_seg[bond]=(PG[bond[1]]-PG[bond[0]], np.linalg.norm(PG[bond[1]]-PG[bond[0]]))
    for bond in bond_form:
        bond_seg[bond]=(RG[bond[1]]-RG[bond[0]], np.linalg.norm(RG[bond[1]]-RG[bond[0]]))

    # create bond list to check cross
    bond_dict={i: [] for i in bond_break+bond_form}
    for i in range(len(E)):
        for j in range(i+1, len(E)):
            for bond in bond_break:
                if Padj[i][j]>0 and i not in bond and j not in bond: bond_dict[bond]+=[(i, j)]
            for bond in bond_form:
                if Radj[i][j]>0 and i not in bond and j not in bond: bond_dict[bond]+=[(i, j)]

    # Compute indicator
    rmsd = return_RMSD(E,RG,PG,rotate=False,mass_weighted=True,namespace=namespace)
    Hbond_dis = np.array([i[1] for bond,i in bond_seg.items() if (bond[0] in H_index or bond[1] in H_index)])
    bond_dis  = np.array([i[1] for bond,i in bond_seg.items() if (bond[0] not in H_index and bond[1] not in H_index)])
    if len(Hbond_dis)>0: 
        max_Hdis=max(Hbond_dis)
    else:                           
        max_Hdis=2.0

    if len(bond_dis)>0: 
        max_dis=max(bond_dis)
    else: 
        max_dis=2.0

    # Compute "cross" behaviour
    min_cross, min_Hcross=[], []
    for bond in bond_break:
        cross_dis=[]
        for ibond in bond_dict[bond]:
            _,_,dis=closestDistanceBetweenLines(PG[bond[0]], PG[bond[1]], PG[ibond[0]], PG[ibond[1]])
            cross_dis+=[dis]
        if len(cross_dis)>0: 
            min_dis=min(cross_dis)
        else: 
            min_dis=2.0

        if bond[0] in H_index or bond[1] in H_index: 
            min_Hcross+=[min_dis]
        else: 
            min_cross+=[min_dis]

    for bond in bond_form:
        cross_dis=[]
        for ibond in bond_dict[bond]:
            _,_,dis=closestDistanceBetweenLines(RG[bond[0]], RG[bond[1]], RG[ibond[0]], RG[ibond[1]])
            cross_dis+=[dis]
        if len(cross_dis) > 0: 
            min_dis=min(cross_dis)
        else: 
            min_dis=2.0
        if bond[0] in H_index or bond[1] in H_index: 
            min_Hcross+=[min_dis]
        else: 
            min_cross+=[min_dis]

    # Find the smallest bonds distance for each bond, if None, return 2.0
    if len(min_cross) > 0:
        min_cross_dis = min(min_cross)
    else:
        min_cross_dis = 2.0

    if len(min_Hcross) > 0:
        min_Hcross_dis = min(min_Hcross)
    else:
        min_Hcross_dis = 2.0
    # Find the cross distanc ebetween two bond changes
    if len([ind for ind in involve if ind in H_index]) ==0:

        if len(bond_break) == 2:
            _,_,dis = closestDistanceBetweenLines(PG[bond_break[0][0]],PG[bond_break[0][1]],PG[bond_break[1][0]],PG[bond_break[1][1]],clampAll=True)
        else:
            dis = 2.0
        path_cross = dis

        if len(bond_form) == 2:
            _,_,dis = closestDistanceBetweenLines(RG[bond_form[0][0]],RG[bond_form[0][1]],RG[bond_form[1][0]],RG[bond_form[1][1]],clampAll=True)
        else:
            dis = 2.0
        path_cross = min(dis,path_cross)

    else:
        path_cross = 2.0

    # return in dataframe format
    indicators = [rmsd, max_dis, max_Hdis, min_cross_dis, min_Hcross_dis, path_cross]
    feature_names = ['RMSD','max_dis','max_Hdis','min_cross_dis','min_Hcross_dis','path_cross']

    return pd.DataFrame([indicators],columns=feature_names)          

def closestDistanceBetweenLines(a0,a1,b0,b1,clampAll=True,clampA0=False,clampA1=False,clampB0=False,clampB1=False):
    ''' 
    Calculate spatial distance between two segments
    Input:  two lines defined by numpy.array pairs (a0,a1,b0,b1)
    Output: the closest points on each segment and their distance
    '''
    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0=True
        clampA1=True
        clampB0=True
        clampB1=True

    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)
   
    _A = A / magA
    _B = B / magB
   
    cross = np.cross(_A, _B);
    denom = np.linalg.norm(cross)**2
   
    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A,(b0-a0))

        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(_A,(b1-a0))

            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0,b0,np.linalg.norm(a0-b0)
                    return a0,b1,np.linalg.norm(a0-b1)


            # Is segment B after A?
        elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1,b0,np.linalg.norm(a1-b0)
                    return a1,b1,np.linalg.norm(a1-b1)
        # Segments overlap, return distance between parallel segments
        return None,None,np.linalg.norm(((d0*_A)+a0)-b0)

    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0);
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA/denom;
    t1 = detB/denom;

    pA = a0 + (_A * t0) # Projected closest point on segment A
    pB = b0 + (_B * t1) # Projected closest point on segment B

    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1

        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1

        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = np.dot(_B,(pA-b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)

        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = np.dot(_A,(pB-a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    return pA,pB,np.linalg.norm(pA-pB)

def seperate_mols(E,G,adj_mat=None,namespace='sep'):
    ''' Function to seperate molecules and return a dictionary of each segment '''
    # generate adj mat
    if adj_mat is None: adj_mat = table_generator(E, G)
    # Seperate reactant(s)
    gs      = graph_seps(adj_mat)
    groups  = []
    loop_ind= []
    for i in range(len(gs)):
        if i not in loop_ind:
            new_group =[count_j for count_j,j in enumerate(gs[i,:]) if j >= 0]
            loop_ind += new_group
            groups   +=[new_group]

    # Determine the inchikey of all components in the reactant
    mols = {}
    for group in groups:
        # parse element and geometry of each fragment
        N_atom = len(group)
        frag_E = [E[ind] for ind in group]
        frag_G = np.zeros([N_atom,3])
        for count_i,i in enumerate(group):
            frag_G[count_i,:] = G[i,:]
        # generate inchikey
        xyz_write(f"{namespace}_input.xyz",frag_E,frag_G)
        molecule = next(pybel.readfile("xyz", f"{namespace}_input.xyz"))
        inchikey = molecule.write(format="inchikey").strip().split()[0]
        os.system(f"rm {namespace}_input.xyz")
        # store this fragment
        if inchikey not in mols.keys():
            mols[inchikey] = [[frag_E,frag_G]]
        else:
            mols[inchikey].append([frag_E,frag_G])

    return mols

def check_multi_molecule(adj,geo,factor='auto'):
    ''' Function to identify whether two or more reactants far away from each other in multi-molecular cases'''
    # Seperate molecules(s)
    gs      = graph_seps(adj)
    groups  = []
    loop_ind= []
    for i in range(len(gs)):
        if i not in loop_ind:
            new_group = [count_j for count_j,j in enumerate(gs[i,:]) if j >= 0]
            loop_ind += new_group
            groups   += [new_group]

    # if only one fragment, return True
    if len(groups) == 1: return True

    # compute center of mass
    centers = []
    radius  = []
    for group in groups:
        center = np.array([0.0, 0.0, 0.0])
        for i in group:
            center += geo[i,:]/float(len(group))

        centers += [center]
        radius  += [max([ np.linalg.norm(geo[i,:]-center) for i in group])]

    # iterate over all paris of centers
    combs = combinations(range(len(centers)), 2)
    max_dis = 0
    satisfy = []

    if factor == 'auto':
        if len(adj) > 12: factor = 1.5
        else: factor = 2.0
        if min([len(j) for j in groups]) < 5: factor = 2.5

    for comb in combs:
        dis = np.linalg.norm(centers[comb[0]]-centers[comb[1]])
        if dis > factor * (radius[comb[0]]+radius[comb[1]]):
            satisfy += [False]
        else:
            satisfy += [True]

    return (False not in satisfy)

def return_RMSD(E,G1,G2,rotate=True,mass_weighted=False,namespace='node'):
    ''' Calcualte RMSD (Root-mean-square-displacement)'''
    # Initialize mass_dict (used for identifying the dihedral among a coincident set that will be explicitly scanned)
    mass_dict = {'H':1.00794,'He':4.002602,'Li':6.941,'Be':9.012182,'B':10.811,'C':12.011,'N':14.00674,'O':15.9994,'F':18.9984032,'Ne':20.1797,\
                 'Na':22.989768,'Mg':24.3050,'Al':26.981539,'Si':28.0855,'P':30.973762,'S':32.066,'Cl':35.4527,'Ar':39.948,\
                 'K':39.0983,'Ca':40.078,'Sc':44.955910,'Ti':47.867,'V':50.9415,'Cr':51.9961,'Mn':54.938049,'Fe':55.845,'Co':58.933200,'Ni':58.6934,'Cu':63.546,'Zn':65.39,\
                 'Ga':69.723,'Ge':72.61,'As':74.92159,'Se':78.96,'Br':79.904,'Kr':83.80,\
                 'Rb':85.4678,'Sr':87.62,'Y':88.90585,'Zr':91.224,'Nb':92.90638,'Mo':95.94,'Tc':98.0,'Ru':101.07,'Rh':102.90550,'Pd':106.42,'Ag':107.8682,'Cd':112.411,\
                 'In':114.818,'Sn':118.710,'Sb':121.760,'Te':127.60,'I':126.90447,'Xe':131.29,\
                 'Cs':132.90545,'Ba':137.327,'La':138.9055,'Hf':178.49,'Ta':180.9479,'W':183.84,'Re':186.207,'Os':190.23,'Ir':192.217,'Pt':195.078,'Au':196.96655,'Hg':200.59,\
                 'Tl':204.3833,'Pb':207.2,'Bi':208.98038,'Po':209.0,'At':210.0,'Rn':222.0}

    if rotate:

        # write two xyz file
        xyz_write('{}1.xyz'.format(namespace),E,G1)
        xyz_write('{}2.xyz'.format(namespace),E,G2)
        node1 = io.read('{}1.xyz'.format(namespace))
        node2 = io.read('{}2.xyz'.format(namespace))
        minimize_rotation_and_translation(node1,node2)
        io.write('{}2.xyz'.format(namespace),node2)

        # reload node 2 geometry and compute RMSD
        _,G2  = xyz_parse('{}2.xyz'.format(namespace))

        try:
            os.remove('{}1.xyz'.format(namespace))
            os.remove('{}2.xyz'.format(namespace))
        except:
            pass
    # compute RMSD
    DG = G1 - G2
    RMSD = 0
    if mass_weighted:
        for i in range(len(E)):
            RMSD += sum(DG[i]**2)*mass_dict[E[i]]

        return np.sqrt(RMSD / sum([mass_dict[Ei] for Ei in E]))

    else:
        for i in range(len(E)):
            RMSD += sum(DG[i]**2)

        return np.sqrt(RMSD / len(E))

def check_duplicate(i,total_i,thresh=0.05):
    ''' check duplicate indicators, return True if unique'''
    if len(total_i) == 0: return True
    min_dis = min([np.linalg.norm(np.array(i)-np.array(j)) for j in total_i])
    # if rmsd > 0.1, this will be a unique conformation
    if min_dis > thresh: return True
    else: return False
