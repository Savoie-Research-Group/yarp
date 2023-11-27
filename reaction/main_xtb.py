import yarp as yp
import numpy as np
import threading
import multiprocessing as mp
from multiprocessing import Queue
from logging.handlers import QueueHandler
from joblib import Parallel, delayed
from yarp.find_lewis import all_zeros
from yarp.find_lewis import bmat_unique
import os, sys, yaml, fnmatch
import logging
from openbabel import pybel
from utils import *
from wrappers.reaction import *
from job_mapping import *
from wrappers.crest import CREST
from qc_jobs import *
from conf import *
from analyze_functions import *
from wrappers.pysis import PYSIS
from wrappers.gsm import GSM

# YARP methodology by Hsuan-Hao Hsu, Qiyuan Zhao, and Brett M. Savoie
def main(args:dict):
    input_path=args['input']
    scratch=args['scratch']
    break_bond=int(args['n_break'])
    strategy=int(args['strategy'])
    n_conf=int(args['n_conf'])
    nprocs   = int(args['nprocs'])
    c_nprocs = int(args['c_nprocs'])
    mem      = int(args['mem'])*1000 
    if args['low_solvation']: args['solvation_model'], args['solvent'] = args['low_solvation'].split('/')
    else: args['solvation_model'], args['solvent'] = 'alpb', False
    method=args['method']
    form_all=int(args["form_all"])
    lewis_criteria=float(args["lewis_criteria"])
    crest=args['crest']
    xtb=args['xtb']
    charge=args['charge']
    multiplicity=args['multiplicity']
    scratch_xtb    = f'{scratch}/xtb_run'
    scratch_crest  = f'{scratch}/conformer'
    conf_output    = f'{scratch}/rxn_conf'
    args['scratch_xtb']  = scratch_xtb
    args['scratch_crest']= scratch_crest
    args['conf_output']  = conf_output
    if os.path.exists(scratch) is False: os.makedirs('{}'.format(scratch))
    if os.path.isdir(scratch_xtb) is False: os.mkdir(scratch_xtb)
    if os.path.isdir(scratch_crest) is False: os.mkdir(scratch_crest)
    if os.path.isdir(conf_output) is False: os.mkdir(conf_output)

    logging_path = os.path.join(scratch, "YARPrun.log")
    logging_queue = mp.Manager().Queue(999)                                                                                                    
    logger_p = mp.Process(target=logger_process, args=(logging_queue, logging_path), daemon=True)
    logger_p.start()
    start = time.time()
    Tstart= time.time()
    logger = logging.getLogger("main")
    logger.addHandler(QueueHandler(logging_queue))
    logger.setLevel(logging.INFO)

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
    for i in mol: rxns=enumeration(i, args=args)
    
    print("-----------------------")
    print("------Second Step------")
    print("Conformational Sampling")
    print("-----------------------")
    if method=='rdkit':
        for count_i, i in enumerate(rxns): rxn[count_i]=i.conf_rdkit(strategy=strategy)
    elif method=='crest':
        chunks=[]
        input_rxns=[]
        for count_i, i in enumerate(rxns): input_rxns.append((count_i, i, args))
        thread = min(nprocs, len(input_rxns))
        chunk_size= len(input_rxns) // thread
        remainder = len(input_rxns) % thread
        startidx=0
        for i in range(thread):
            endidx = startidx + chunk_size + (1 if i < remainder else 0)
            chunks.append(input_rxns[startidx:endidx])
            startidx = endidx
        all_job_mappings = Parallel(n_jobs=thread)(delayed(process_input_rxn)(chunk) for chunk in chunks)
        job_mappings = merge_job_mappings(all_job_mappings)
        crest_thread=nprocs//c_nprocs
        track_crest={}
        crest_job_list=[]
        for inchi, jobi in job_mappings.items():
            wf=f"{scratch_crest}/{inchi}"
            if os.path.isdir(wf) is False: os.mkdir(wf)
            inp_xyz=f"{wf}/{inchi}.xyz"
            xyz_write(inp_xyz, jobi["E"], jobi['G'])
            crest_job=CREST(input_geo=inp_xyz, work_folder=wf, nproc=c_nprocs, mem=mem, quick_mode=args['crest_quick'], opt_level=args['opt_level'],\
                            solvent=args['solvent'], solvation_model=args['solvation_model'], charge=args['charge'], multiplicity=args['multiplicity'])
            if args['crest_quick']: crest_job.add_command(additional='-rthr 0.1 -ewin 8 ')
            crest_job_list.append(crest_job)
            for jobid in jobi['jobs']: track_crest[jobid]=crest_job
        input_job_list=[(crest_job, logging_queue) for crest_job in crest_job_list]
        Parallel(n_jobs=crest_thread)(delayed(run_crest)(*task) for task in input_job_list)
        rxns=read_crest_in_class(rxns, scratch_crest)
    # Generate the reaction id for different inchi
    inchi_array=[]
    for i in rxns:
        if i.reactant_inchi not in inchi_array: inchi_array.append(i.reactant_inchi)
    inchi_dict=dict()
    for i in inchi_array: inchi_dict[i]=0
    for i in rxns:
        inchi=i.reactant_inchi
        idx=inchi_dict[inchi]
        i.id=idx
        inchi_dict[inchi]=idx+1
    
    print("-----------------------")
    print("-------Third Step------")
    print("Conformation Generation")
    print("-----------------------")
    if os.path.isdir(conf_output) is True and len(os.listdir(conf_output))>0:
        print("Reaction conformation sampling has already been done in the target folder, skip this step...")
    else:
        thread=min(nprocs, len(rxns))
        chunk_size=len(rxns)//thread
        remainder=len(rxns)%thread
        input_data_list=[(rxn, logging_queue) for rxn in rxns]
        chunks=[]
        startidx=0
        for i in range(thread):
            endidx=startidx+chunk_size+(1 if i < remainder else 0)
            chunks.append(input_data_list[startidx:endidx])
            startidx=endidx
        modified_rxns=Parallel(n_jobs=thread)(delayed(generate_rxn_conf)(chunk) for chunk in chunks)
        rxns=modified_rxns
        print(f"Finish generating reaction conformations, the output conformations are stored in {conf_output}\n")
    
    print("-----------------------")
    print("-------Forth Step------")
    print("-Growing String Method-")
    print("----------and----------")
    print("------Berny TS Opt-----")
    print("-----------------------")
    # write the reaction xyz to conf_output for following GSM calculation
    for i in rxns:
        key=[j for j in i.rxn_conf.keys()]
        for j in key:
            name=f"{conf_output}/{i.reactant_inchi}_{i.id}_{j}.xyz"
            write_reaction(i.reactant.elements, i.rxn_conf[j]["R"], i.rxn_conf[j]["P"], filename=name)
    rxns_confs = [rxn for rxn in os.listdir(conf_output) if rxn[-4:]=='.xyz']
    gsm_thread = min(nprocs, len(rxns_confs))
    gsm_jobs   = {}

    # preparing and running GSM-xTB
    for count,rxn in enumerate(rxns_confs):
        # prepare GSM job
        rxn_ind = rxn.split('.xyz')[0]
        wf = f"{scratch}/{rxn_ind}"
        if os.path.isdir(wf) is False: os.mkdir(wf)
        inp_xyz = f"{conf_output}/{rxn}"
        gsm_job = GSM(input_geo=inp_xyz,input_file=args['gsm_inp'],work_folder=wf,method='xtb', jobname=rxn_ind, jobid=count, charge=args['charge'],\
                      multiplicity=args['multiplicity'], solvent=args['solvent'], solvation_model=args['solvation_model'])
        gsm_job.prepare_job()
        gsm_jobs[rxn_ind] = gsm_job

    # Create a process pool with gsm_thread processes
    gsm_job_list = [gsm_jobs[ind] for ind in sorted(gsm_jobs.keys())]

    # Run the tasks in parallel
    input_job_list = [(gsm_job, logging_queue) for gsm_job in gsm_job_list]
    Parallel(n_jobs=gsm_thread)(delayed(run_gsm)(*task) for task in input_job_list)
    tsopt_jobs = {}
    for gsm_job in gsm_job_list:
        if gsm_job.calculation_terminated_normally() is False:
            print(f'GSM job {gsm_job.jobname} fails to converge, please check this reaction...')
        elif gsm_job.find_correct_TS() is False:
            print(f'GSM job {gsm_job.jobname} fails to locate a TS, skip this rxn...')
        else:
            TSE, TSG = gsm_job.get_TS()
            xyz_write(f"{gsm_job.work_folder}/{gsm_job.jobname}-TSguess.xyz",TSE,TSG)
            if not args['solvent']:
                pysis_job = PYSIS(input_geo=f"{gsm_job.work_folder}/{gsm_job.jobname}-TSguess.xyz",work_folder=gsm_job.work_folder,jobname=gsm_job.jobname,jobtype='tsopt',charge=args['charge'],multiplicity=args['multiplicity'])
            else:
                if args['solvation_model'].lower() == 'alpb':
                    pysis_job = PYSIS(input_geo=f"{gsm_job.work_folder}/{gsm_job.jobname}-TSguess.xyz",work_folder=gsm_job.work_folder,jobname=gsm_job.jobname,jobtype='tsopt',\
                                      charge=args['charge'],multiplicity=args['multiplicity'],alpb=args['solvent'])
                else:
                    pysis_job = PYSIS(input_geo=f"{gsm_job.work_folder}/{gsm_job.jobname}-TSguess.xyz",work_folder=gsm_job.work_folder,jobname=gsm_job.jobname,jobtype='tsopt',\
                                      charge=args['charge'],multiplicity=args['multiplicity'],gbsa=args['solvent'])
            pysis_job.generate_input(calctype='xtb', hess=True, hess_step=1)
            tsopt_jobs[gsm_job.jobname]= pysis_job

    # Create a process pool with gsm_thread processes
    tsopt_job_list= [tsopt_jobs[ind] for ind in sorted(tsopt_jobs.keys())]
    tsopt_thread  = min(nprocs, len(tsopt_job_list))

    # Run the tasks in parallel
    input_job_list = [(tsopt_job, logging_queue, args['pysis_wt']) for tsopt_job in tsopt_job_list]
    Parallel(n_jobs=tsopt_thread)(delayed(run_pysis)(*task) for task in input_job_list)

    # check tsopt jobs
    tsopt_job_list = check_dup_ts_pysis(tsopt_job_list, logger)
    
    print("-----------------------")
    print("-------Fifth Step------")
    print("----IRC calculation----")
    print("-----------------------")
    irc_jobs = {}
    for tsopt_job in tsopt_job_list:
        TSE, TSG = tsopt_job.get_final_ts()
        xyz_write(f"{tsopt_job.work_folder}/{tsopt_job.jobname}-TS.xyz",TSE,TSG)

        if not args['solvent']:
            pysis_job = PYSIS(input_geo=f"{tsopt_job.work_folder}/{tsopt_job.jobname}-TS.xyz",work_folder=tsopt_job.work_folder,jobname=tsopt_job.jobname,jobtype='irc',charge=args['charge'],multiplicity=args['multiplicity'])
        else:
            if args['solvation_model'].lower() == 'alpb':
                pysis_job = PYSIS(input_geo=f"{tsopt_job.work_folder}/{tsopt_job.jobname}-TS.xyz",work_folder=tsopt_job.work_folder,jobname=tsopt_job.jobname,jobtype='irc',charge=args['charge'],\
                                  multiplicity=args['multiplicity'],alpb=args['solvent'])
            else:
                pysis_job = PYSIS(input_geo=f"{tsopt_job.work_folder}/{tsopt_job.jobname}-TS.xyz",work_folder=tsopt_job.work_folder,jobname=tsopt_job.jobname,jobtype='irc',charge=args['charge'],\
                                  multiplicity=args['multiplicity'],gbsa=args['solvent'])

        if os.path.isfile(f'{tsopt_job.work_folder}/ts_final_hessian.h5'): pysis_job.generate_input(calctype='xtb', hess_init=f'{tsopt_job.work_folder}/ts_final_hessian.h5')
        else: pysis_job.generate_input(calctype='xtb')
        irc_jobs[tsopt_job.jobname] = pysis_job

    # Create a process pool with gsm_thread processes
    irc_job_list= [irc_jobs[ind] for ind in sorted(irc_jobs.keys())]
    irc_thread  = min(nprocs, len(irc_job_list))

    # Run the tasks in parallel
    input_job_list = [(irc_job, logging_queue, args['pysis_wt']) for irc_job in irc_job_list]
    Parallel(n_jobs=irc_thread)(delayed(run_pysis)(*task) for task in input_job_list)
    
    # Report and apply D2 ML model
    if args['apply_d2'] is False:
        reactions = analyze_outputs(scratch,irc_job_list,logger,charge=args['charge'],nc_thresd=args['nconf_dft'],select=args['select'])
    else:
        reactions,D2_rxns = analyze_outputs(scratch,irc_job_list,logger,charge=args['charge'],nc_thresd=args['nconf_dft'],select=args['select'],return_D2_dict=True)
    return

def read_crest_in_class(rxns, scratch_crest):
    conf_inchi=[inchi for inchi in os.listdir(scratch_crest) if os.path.isdir(scratch_crest+'/'+inchi)]
    for i in conf_inchi:
        elements, geos = xyz_parse(f"{scratch_crest}/{i}/crest_conformers.xyz", multiple=True)
        for count_j, j in enumerate(rxns):
            if j.product_inchi in i:
                for count_k, k in enumerate(geos):
                    rxns[count_j].product_conf[count_k]=k
            if j.reactant_inchi in i:
                for count_k, k in enumerate(geos):
                    rxns[count_j].reactant_conf[count_k]=k
    return rxns

def enumeration(input_mol, args={}):
    nb=args["n_break"]
    form_all=args["form_all"]
    criteria=args["lewis_criteria"]
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
    products=yp.sieve_bmat_scores(products)
    #products=yp.sieve_fused_rings(mols,keep=False)
    #products=yp.sieve_rings(mols,{3,4},keep=False)
    #products=yp.sieve_fc(mols,[-1,1],keep=False)
    products=[_ for _ in products if _.bond_mat_scores[0]<=criteria and sum(np.abs(_.fc))<=2.0]
    print(f"{len(products)} cleaned products after find_lewis() filtering")
    rxn=[]
    for count_i, i in enumerate(products):
        R=reaction(reactant, products[0], args=args, opt=True)
        rxn.append(R)
    return rxn

def write_reaction(elements, RG, PG, filename="reaction.xyz"):
    out=open(filename, 'w+')
    out.write("{}\n\n".format(len(elements)))
    for count_i, i in enumerate(elements):
        i.capitalize()
        out.write("{:<20s} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(i, RG[count_i][0], RG[count_i][1], RG[count_i][2]))
    out.write("{}\n\n".format(len(elements)))
    for count_i, i in enumerate(elements):
        i.capitalize()
        out.write("{:<20s} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(i, PG[count_i][0], PG[count_i][1], PG[count_i][2]))
    out.close()
    return

def write_reaction_yp(R, P, filename="reaction.xyz"):
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
