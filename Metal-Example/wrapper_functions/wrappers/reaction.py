import sys, itertools, timeit, os
import numpy as np
import pickle
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
from copy import deepcopy
from xtb import *
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))
from utils import *
from conf import *

class reaction:
    """
    Base class for storing information of a reaction and performing conformational sampling
    
    Attributes
    ----------

    reactant: yarpecule class for reactant

    product: yarpecule class for product

    opt_P: perform initial geometry optimization on product  side. (default: False)
    opt_R: perform initial geometry optimization on reactant side. (default: False)
    
    """
    def __init__(self, reactant, product, args=dict(), opt_R=False, opt_P=True):
        
        self.reactant=reactant
        self.product=product
        self.args=args
        self.conf_path=self.args["scratch_crest"]
        n_conf=self.args["n_conf"]
        self.n_conf=self.args["n_conf"]
        # safe check
        for count_i, i in enumerate(reactant.elements): reactant.elements[count_i]=i.capitalize()
        for count_i, i in enumerate(product.elements): product.elements[count_i]=i.capitalize()
        for count_i, i in enumerate(reactant.elements):
            if i != product.elements[count_i]:
                print(f"count_i: {count_i}, reactant elements: {i}\n")
                print(f"count_i: {count_i}, product  elements: {product.elements[count_i]}\n")
                print("Fatal error: reactant and product are not same. Please check the input.....")
                exit()
        if opt_R: self.reactant=geometry_opt(self.reactant)
        if opt_P: self.product=geometry_opt(self.product)
        self.reactant_dft_opt=dict()
        self.product_dft_opt=dict()
        self.reactant_conf=dict()
        self.product_conf=dict()
        self.reactant_energy=dict()
        self.product_energy=dict()
        self.reactant_inchi=return_inchikey(self.reactant)
        self.product_inchi=return_inchikey(self.product)
        self.reactant_smiles=return_smi_yp(self.reactant)
        self.reactant_smiles=return_smi_yp(self.product)
        self.rxn_conf=dict()
        self.id=0
        self.TS_guess=dict()
        self.TS_xtb=dict()
        self.TS_dft=dict()
        self.IRC_xtb=dict()
        self.IRC_dft=dict()
        self.constrained_TS=dict()
        if os.path.isdir(self.conf_path) is False: os.system('mkdir {}'.format(self.conf_path))

    def conf_rdkit(self):
        if self.args["strategy"]==0 or self.args["strategy"]==2:
            if os.path.isdir('{}/{}'.format(self.conf_path, self.reactant_inchi)) is False: os.system('mkdir {}/{}'.format(self.conf_path, self.reactant_inchi))
            if os.path.isfile('{}/{}/rdkit_conf.xyz'.format(self.conf_path, self.reactant_inchi)) is False:
                # sampling on reactant side
                mol_file='.reactant.tmp.mol'
                mol_write_yp(mol_file, self.reactant, append_opt=False)
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
        if self.args["strategy"]==1 or self.args["strategy"]==2:
            if os.path.isdir('{}/{}'.format(self.conf_path, self.product_inchi)) is False: os.system('mkdir {}/{}'.format(self.conf_path, self.product_inchi))
            if os.path.isfile('{}/{}/rdkit_conf.xyz'.format(self.conf_path, self.product_inchi)) is False:
                # sampling on reactant side
                mol_file='.product.tmp.mol'
                mol_write_yp(mol_file, self.product, append_opt=False)
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

    def rxn_conf_generation(self, logging_queue):

        job_id=f"{self.reactant_inchi}_{self.id}"
        
        RG=self.reactant.geo
        RE=self.reactant.elements
        R_adj=self.reactant.adj_mat
        R_bond_mats=self.reactant.bond_mats

        PG=self.product.geo
        PE=self.product.elements
        P_adj=self.product.adj_mat
        P_bond_mats=self.product.bond_mats

        tmp_rxn_dict=dict()
        count=0
        # Create a dictionary to store the conformers and product/reactant bond mat.
        if self.args["strategy"]!=0:
            for i in self.product_conf.keys():
                print(f"self.product_conf[{i}]: {self.product_conf[i]}\n")
                tmp_rxn_dict[count]={"E": RE, "bond_mat_r": R_bond_mats[0], "G": deepcopy(self.product_conf[i]), 'direct':'B'}
                count=count+1
        print(f"Product sample, there are {count} tmp rxns\n", flush = True)

        if self.args["strategy"]!=1:
            for i in self.reactant_conf.keys():
                print(f"self.reactant_conf[{i}]: {self.reactant_conf[i]}\n")
                tmp_rxn_dict[count]={"E": RE, "bond_mat_r": P_bond_mats[0], "G": deepcopy(self.reactant_conf[i]), 'direct': "F"}
                count=count+1
        print(f"Total: there are {count} tmp rxns\n", flush = True)

        #exit()

        # load ML model to find conformers 
        if len(tmp_rxn_dict)>3*self.n_conf: model=pickle.load(open(os.path.join(self.args['model_path'],'rich_model.sav'), 'rb'))
        else: model=pickle.load(open(os.path.join(self.args['model_path'],'poor_model.sav'), 'rb'))

        ind_list, pass_obj_values=[], []

        # Load additional constraints proposed by the user
        reactant_total_constraints = []
        product_total_constraints = []

        if self.args['constraint']:
            if not self.args['reactant_dist_constraint'] is None:
                constraint_argument = self.args['reactant_dist_constraint']
                inp_list = constraint_argument.split(',')
                for a in range(0, int(len(inp_list) / 3)):
                    arg_list = [int(inp_list[a * 3]), int(inp_list[a * 3 + 1]), float(inp_list[a * 3 + 2])]
                    reactant_total_constraints.append(arg_list)
            elif not self.args['product_dist_constraint'] is None:
                constraint_argument = self.args['product_dist_constraint']
                inp_list =  constraint_argument.split(',')
                for a in range(0, int(len(inp_list) / 3)):
                    arg_list = [int(inp_list[a * 3]), int(inp_list[a * 3 + 1]), float(inp_list[a * 3 + 2])]
                    product_total_constraints.append(arg_list)

        for conf_ind, conf_entry in tmp_rxn_dict.items():
            print(f"ind: {conf_ind}, conf_entry: {conf_entry['G'][0]}, direction: {tmp_rxn_dict[conf_ind]['direct']}\n")
            # apply force-field optimization
            # apply xTB-restrained optimization soon!
            # Zhao's note: Change here to adj_mat as well...
            # JUST DEBUG, Try skip this line?
            ###Gr = opt_geo(conf_entry['E'],conf_entry['G'],conf_entry['bond_mat_r'],ff=self.args['ff'],step=100,filename=f'tmp_{job_id}')
            #Gr = opt_geo(conf_entry['E'],conf_entry['G'],conf_entry['adj_mat_r'],ff=self.args['ff'],step=100,filename=f'tmp_{job_id}')
            #Gr = deepcopy(PG)
            # skip the UFF opt ???

            #Zhao's note: added the xtb manually#
            #xtb opt now uses the all the bonds as constraints#
            #additional constraints are also applied by the user in args['reactant_dist_constraint'] or args['product_dist_constraint']
            tmp_xyz_r = f"{self.args['scratch_xtb']}/{job_id}_r.xyz"
            xyz_write(tmp_xyz_r,conf_entry['E'],conf_entry['G'])

            if(tmp_rxn_dict[conf_ind]['direct'] == 'F'):
                ADMatrix = P_adj
                All_constraint = return_all_constraint(self.product)
                All_constraint = All_constraint + product_total_constraints
            else:
                ADMatrix = R_adj
                All_constraint = return_all_constraint(self.reactant)
                All_constraint = All_constraint + reactant_total_constraints
                print(f"All_constraint for reactant: {All_constraint}\n")

            if self.args["low_solvation"]:
                solvation_model, solvent = self.args["low_solvation"].split("/")
                optjob=XTB(input_geo=tmp_xyz_r,
                        work_folder=self.args["scratch_xtb"],lot=self.args["lot"], jobtype=["opt"],nproc=self.args['xtb_nprocs'],\
                        solvent=solvent, solvation_model=solvation_model,
                        jobname=f"joint_xtb",
                        charge=self.args["charge"], multiplicity=self.args["multiplicity"])
                optjob.add_command(distance_constraints=All_constraint)
            else:
                optjob=XTB(input_geo=tmp_xyz_r,
                        work_folder=self.args["scratch_xtb"],lot=self.args["lot"], jobtype=["opt"],nproc=self.args['xtb_nprocs'],\
                        jobname=f"joint_xtb",
                        charge=self.args["charge"], multiplicity=self.args["multiplicity"])
                optjob.add_command(distance_constraints=All_constraint)
            optjob.execute()
           
            #exit()

            if(conf_ind % 20 == 0): print(f"Processed/Optimized {conf_ind} rxn confs\n", flush = True)

            if optjob.optimization_success():
                _, Gr = optjob.get_final_structure()
            else:
                #logger.info(f"xtb geometry optimization fails for the other end of {job_id} (conf: {conf_ind}), will use force-field optimized geometry for instead")
                Gr = []

            if len(Gr)==0: 
                #print(f"NO Gr Generated!!!!\n", flush = True)
                continue
            tmp_xyz_p = f"{self.args['scratch_xtb']}/{job_id}_p.xyz"
            xyz_write(tmp_xyz_p,conf_entry['E'],Gr)
            tmp_xyz_r = f"{self.args['scratch_xtb']}/{job_id}_r.xyz"
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
            #Zhao's note: may need a threshold argument for this 
            model_threshold = 0.4
            if('model_threshold' in self.args):
                model_threshold = float(self.args['model_threshold'])
            if model.predict_proba(indicators)[0][1] > model_threshold and check_duplicate(indicators,ind_list,thresh=0.025):
                ind_list.append(indicators)
                pass_obj_values.append((model.predict_proba(indicators)[0][0],deepcopy(conf_entry['G']),Gr,deepcopy(conf_entry['direct'])))

            # remove tmp file
            if os.path.isfile(tmp_xyz_r): os.remove(tmp_xyz_r)
            if os.path.isfile(tmp_xyz_p): os.remove(tmp_xyz_p)

        #exit()

        pass_obj_values=sorted(pass_obj_values, key=lambda x: x[0])
        
        #Zhao's note: consider making 2 lists of top R/P conformers
        #do cross terms, for example, the top 4 pass_obj_values are from cross terms
        #for the cross terms, it will use the sorted best conformers from both sides
        #modify "top_cross_terms" to use it, if not using, just let it = 0
        top_cross_terms = 0
        top_R = [a for a in pass_obj_values if a[3] == 'F']
        top_P = [a for a in pass_obj_values if a[3] == 'B']
        top_R_index, top_P_index = 0, 0
        for index_val in range(0, top_cross_terms):
            direction = pass_obj_values[index_val][3]
            if(direction == 'F'):
                pass_obj_values[index_val] = (pass_obj_values[index_val][0], pass_obj_values[index_val][1], deepcopy(top_P[top_P_index][1]), pass_obj_values[index_val][3])
                top_P_index += 1
            if(direction == 'B'):
                pass_obj_values[index_val] = (pass_obj_values[index_val][0], pass_obj_values[index_val][1], deepcopy(top_R[top_R_index][1]), pass_obj_values[index_val][3])
                top_R_index += 1

        #exit()

        N_conf=0
        for item in pass_obj_values:
            # item[0] is the predicted score
            input_type=item[3]

            print(f"items in pass_obj_values: {item}\n")

            tmp_xyz_r = f"{self.args['scratch_xtb']}/{job_id}_r.xyz"
            tmp_xyz_p = f"{self.args['scratch_xtb']}/{job_id}_p.xyz"
            print(f"tmp_xyz_r file: {tmp_xyz_r}\n")
            print(f"tmp_xyz_p file: {tmp_xyz_p}\n")
            xyz_write(tmp_xyz_r, RE, item[1])
            xyz_write(tmp_xyz_p, RE, item[2])
            if self.args['opt']:
                if self.args['low_solvation']:
                    solvation_model, solvent = self.args['low_solvation'].split('/')
                    optjob = XTB(input_geo=tmp_xyz_p,work_folder=self.args['scratch_xtb'],jobtype=['opt'],nproc=self.args['xtb_nprocs'], jobname=f'opt_{job_id}_p',solvent=solvent,\
                                 solvation_model=solvation_model,charge=self.args['charge'],multiplicity=self.args['multiplicity'])
                else:
                    os.system(f"cp {tmp_xyz_p} self.args['scratch_xtb']/asdasdasdasd_inp_P.xyz")
                    optjob = XTB(input_geo=tmp_xyz_p,work_folder=self.args['scratch_xtb'],jobtype=['opt'],nproc=self.args['xtb_nprocs'], jobname=f'opt_{job_id}_p',charge=self.args['charge'],multiplicity=self.args['multiplicity'])

                optjob.execute()

                if optjob.optimization_success():
                    _, Gr = optjob.get_final_structure()
                else:
                    #Gr = item[2]
                    continue

                xyz_write(tmp_xyz_p,conf_entry['E'],Gr)

            if input_type=='F':
                _, rg=xyz_parse(tmp_xyz_r)
                _, pg=xyz_parse(tmp_xyz_p)
                self.rxn_conf[N_conf]={"R": rg, "P": pg}
                # Zhao's note: Maybe keep a copy of the xyz files?
                os.system(f"cp {tmp_xyz_r} {self.args['conf_output']}/{job_id}_{N_conf}.xyz; cat {tmp_xyz_p} >> {self.args['conf_output']}/{job_id}_{N_conf}.xyz;rm {tmp_xyz_r} {tmp_xyz_p}")
                #os.system(f"rm {tmp_xyz_r}")
                #os.system(f"rm {tmp_xyz_p}")
            else:
                _, rg=xyz_parse(tmp_xyz_p)
                _, pg=xyz_parse(tmp_xyz_r)
                self.rxn_conf[N_conf]={"R": rg, "P": pg}
                # Zhao's note: Maybe keep a copy of the xyz files?
                os.system(f"cp {tmp_xyz_p} {self.args['conf_output']}/{job_id}_{N_conf}.xyz; cat {tmp_xyz_r} >> {self.args['conf_output']}/{job_id}_{N_conf}.xyz;rm {tmp_xyz_r} {tmp_xyz_p}")
                #os.system(f"rm {tmp_xyz_r}")
                #os.system(f"rm {tmp_xyz_p}")
            N_conf=N_conf+1
            if N_conf>=self.args["n_conf"]: break

        if len(pass_obj_values) == 0:
            print(f"WARNING: None of the reaction conformation can be generated for the input reaction {job_id}. please check this reaction to make sure it is a vaild one")

        # add a joint-opt alignment if too few alignments pass the criteria
        # will add soon
        #exit()
        return

