import sys, itertools, timeit, os, copy  
from openbabel import pybel
from openbabel import openbabel as ob
from collections import Counter    
import numpy as np
from yarp.taffi_functions import graph_seps,table_generator,return_rings,adjmat_to_adjlist,canon_order
from yarp.properties import el_to_an,an_to_el,el_mass, el_radii
from yarp.find_lewis import find_lewis,return_formals,return_n_e_accept,return_n_e_donate,return_formals,return_connections,return_bo_dict
from yarp.hashes import atom_hash,yarpecule_hash
from yarp.input_parsers import xyz_parse,xyz_q_parse,xyz_from_smiles, mol_parse
from yarp.misc import merge_arrays, prepare_list
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import EnumerateStereoisomers, AllChem, TorsionFingerprints, rdmolops, rdDistGeom
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.ML.Cluster import Butina
import fnmatch

from wrappers.xtb      import *

from scipy.spatial.distance import cdist

'''
from wrappers.pysis    import *
from wrappers.orca     import *
from wrappers.gaussian import *
from wrappers.crest    import *
########################################
# Calculator for xTB/DFT Calculations  #
# Supported software:                  #
# xTB, Pysis, ORCA, Gaussian           #
########################################
class Calculator:
    def __init__(self, args):
        """
        Initialize an DFT input class
        a wrapper for DFT input keywords 
        including charge, multiplicity, solvation, level of theory, etc.
        different calculator uses different syntax for jobtype, this class takes in a unified name set
        the class will take: opt, tsopt, copt, irc, gsm, ... (all small cases)
        each one will be translated to the corresponding string accepted by the chosen calculator
        """
        self.input_geo   = ""
        self.work_folder = os.getcwd()
        self.xtb_lot     = args["lot"]
        self.lot         = args["dft_lot"]
        self.jobtype     = 'OPT'
        self.nproc       = int(args["dft_nprocs"])
        self.mem         = int(args["mem"]*1000)
        self.mix_basis   = args['dft_mix_basis']
        self.mix_lot     = args['dft_mix_lot']
        self.jobname     = 'job'
        self.charge      = args["charge"]
        self.multiplicity= args["multiplicity"]
        self.solvent     = args["solvent"]
        self.dielectric  = args["dielectric"]
        self.dispersion  = args["dispersion"]
        self.solvation_model = args["solvation_model"]
        self.grid        = 2
        self.writedown_xyz   = True
        print(f"self.lot: {self.lot}\n")
        print(f"self.mix_basis: {self.mix_basis}\n")
        print(f"self.mix_lot:   {self.mix_lot}\n")

    def Setup(self, package, args, constraints=[]):
        # PYSIS: used for geo optimization, ts-optimization (xtb), gsm (jobtype=string, coord_type="cart"), irc (jobtype=irc)#
        if(package == "PYSIS"):
            if(self.jobtype == 'gsm'):
                self.jobtype = 'string'
            alpb=args["solvent"]
            gbsa=args["solvent"]
            if(args["low_solvation_model"].lower()=='alpb'):
                gbsa=False
            else:
                alpb=False
            JOB = PYSIS(input_geo=self.input_geo, 
                        work_folder=self.work_folder, 
                        pysis_dir=args["pysis_path"], 
                        jobname=self.jobname, 
                        jobtype=self.jobtype, 
                        nproc=self.nproc, 
                        charge=self.charge, 
                        multiplicity=self.multiplicity, 
                        alpb=alpb,
                        gbsa=gbsa)
            if('opt' in self.jobtype): #gsm or irc don't need hessian keywords, opt and tsopt need them
                JOB.generate_input(calctype='xtb', hess=True, hess_step=1)
            elif('string' in self.jobtype):
                JOB.generate_input(calctype='xtb')
            elif('irc' in self.jobtype):
                if os.path.isfile(f"{self.work_folder}/ts_final_hessian.h5"): 
                    JOB.generate_input(calctype="xtb", hess_init=f"{self.work_folder}/ts_final_hessian.h5")
                else: JOB.generate_input(calctype='xtb')
            JOB.generate_constraints(distance_constraints = constraints)
            
        elif(package=="XTB"):
            if(self.jobtype == 'opt'):
                self.jobtype = ['opt']
            else:
                print(f"XTB wrapper can only do geometry optimization ('opt')!\n")
                exit()
            JOB =  XTB(input_geo=self.input_geo, 
                        work_folder=self.work_folder,
                        lot=self.xtb_lot, 
                        jobtype=["opt"],
                        solvent=args["solvent"], 
                        solvation_model=args["low_solvation_model"],
                        jobname=self.jobname, 
                        charge=args["charge"], 
                        multiplicity=args["multiplicity"])
            JOB.add_command(distance_constraints=constraints)
'''

#################################################################
# Add Mix Basis Set (available in ORCA and Gaussian)            #
# For ORCA: add specific basis set info to the end of the atom  #
# e.g.: C 0.1 0.1 0.1 newgto "def2-TZVP" end                    #
# For Gaussian: add to the end of the file                      #
# e.g.:                                                         #
#        ****                                                   #
#        44 48 0                                                #
#        def2TZVP                                               #
#        ****                                                   #
#################################################################

def compare_lists(list1, list2):
    """
    Compare two lists of lists and return the indices of differing elements.

    Parameters:
    - list1: The first list of lists to compare.
    - list2: The second list of lists to compare.

    Returns:
    A list of tuples, where each tuple contains the indices of differing elements.
    """
    differing_indices = []

    # Iterate through the lists to the length of the shorter list
    for i in range(min(len(list1), len(list2))):
        # Ensure both elements are lists before comparison and have the same length

        try:
            for j in range(min(len(list1[i]), len(list2[i]))):
                if list1[i][j] != list2[i][j]:
                    if not i in differing_indices:
                        differing_indices.append(i)
        except:
            print(f"Have issue when comparing lists of list !!!\n")

    return differing_indices

#Zhao's note: the function read finds the metal, and assign basis set to those within the 1st/2nd layer of the metal#
#each atom in these layers will be assigned a list, [element_name+index, basis-set_name]
#for example, [Zn1, def2-TZVP]
#the index is needed to have precise control 

def treat_mix_lot_metal_firstLayer(args, elements, geometry):
    if args['dft_mix_firstlayer']:
        first_layer_index = []
        # get adj_mat for TS
        TS_adj_mat = table_generator(elements, geometry)
        # get the metals
        #metal_element = [e for e in elements if e in el_metals]
        metal_ind = [ind for ind, e in enumerate(elements) if e in el_metals]
        # add reactive atoms to the indices #
        if 'Reactive_Atoms' in args:
            if not (args['Reactive_Atoms'] is None):
                metal_ind.extend(args['Reactive_Atoms'])
                #metal_ind.unique()
                metal_ind = list(dict.fromkeys(metal_ind))
                print(f"TZ indices: {metal_ind}\n")
        if len(metal_ind) == 0: return
        # get 1st layer
        counter = 0
        for metal in metal_ind:
            metal_row = TS_adj_mat[metal]
            link_ind  = [ind for ind, val in enumerate(metal_row) if val > 0]
            link_element = [elements[a] for a in link_ind]
            counter += 1

        if(len(link_ind) > 0):
            atom_list = []
            for atom_index in range(0, len(link_ind)):
                atom_list = [link_element[atom_index]+str(link_ind[atom_index]), args['dft_mix_firstlayer_lot']]

                if atom_list not in args['dft_mix_lot']:
                    first_layer_index.append(atom_list)


        # add second layer if needed
        second_layer = False
        counter = 0
        if second_layer:
            for atom_ind in link_ind:
                atom_row = TS_adj_mat[atom_ind]
                # pop the metal in the neighbor list
                neighbor_ind     = [ind for ind, val in enumerate(atom_row) if(val > 0 and not elements[ind] in el_metals)]
                neighbor_element = [elements[a] for a in neighbor_ind]
                counter += 1

                if(len(neighbor_ind) > 0):
                    atom_list = []
                    for atom_index in range(0, len(neighbor_ind)):
                        atom_list = [neighbor_element[atom_index]+str(neighbor_ind[atom_index]), args['dft_mix_firstlayer_lot']]
                        if atom_list not in args['dft_mix_lot']:
                            first_layer_index.append(atom_list)
        args['dft_mix_lot'].extend(first_layer_index)

        # sort the list so that element+index appears at the beginning of the list
        # alnum = alpha-numeric
        alnum_element = [a for a in args['dft_mix_lot'] if (any(x.isalpha() for x in a[0]) and (any(x.isnumeric() for x in a[0])))]
        not_alnum_element = [a for a in args['dft_mix_lot'] if not (any(x.isalpha() for x in a[0]) and (any(x.isnumeric() for x in a[0])))]
        #print(f"alnum_element: {alnum_element}\n", flush = True)
        #print(f"not_alnum_element: {not_alnum_element}\n", flush = True)
        alnum_element.extend(not_alnum_element)
        args['dft_mix_lot'] = alnum_element

        print(f"args[dft_mix_lot]: {args['dft_mix_lot']}\n", flush = True)

def process_mix_basis_input(args):
    args['dft_mix_basis'] = bool(args['dft_mix_basis'])
    if args['dft_mix_basis']:
        dft_mix_lot = []
        inp_list = args['dft_mix_lot'].split(',')
        for a in range(0, int(len(inp_list) / 2)):
            arg_list = [inp_list[a * 2].strip(), inp_list[a * 2 + 1].strip()] # get rid of the space for each input keyword
            dft_mix_lot.append(arg_list)

    #print("dft_mix_lot: ", flush = True)
    #print(dft_mix_lot, flush = True)

    args['dft_mix_lot'] = dft_mix_lot

    # Zhao's note: Print flag about full TZ-level single point energy/free energy correction
    args['dft_fulltz_level_correction'] = bool(args['dft_fulltz_level_correction'])
    if(args['dft_mix_basis'] and args['dft_fulltz_level_correction']):
        print(f"Using Mix (TZ/DZ/SZ) Basis Sets and a later TZ Single-Point Corrections for Energy and Free Energy\n")
    if(args['dft_mix_basis'] and not args['dft_fulltz_level_correction']):
        print(f"Using Mix (TZ/DZ/SZ) Basis Sets, but no TZ Correction used, the Energy/Free Energy might be off. We recommend you use **dft_fulltz_level_correction: True**\n")
    if(not args['dft_mix_basis'] and args['dft_fulltz_level_correction']):
        print(f"Not Using Mix (TZ/DZ/SZ) Basis Sets, but TZ Correction used, What are you using it for???\n")
        raise RuntimeError("Please change your input file!!!")

'''
#Zhao's note: a function for yarp-dft to wait for the dft-jobs that are launched by the previous yarp-dft run
#for example, the previous yarp-dft launched 2 TSOPT job and died, the 2 TSOPT jobs are still running 
#now, if you start a new yarp-dft run, it will wait until those 2 TSOPT jobs are dead.
#jobs will be written to a text file "last_jobs.txt", the text file will tell what jobs are currently running 
def read_wait_for_last_jobs():
    print("checking for unfinished jobs from the previous run\n")
    file_path = 'last_jobs.txt'
    if not os.path.exists(file_path): return
    with open(file_path, 'r') as file:
        # Use list comprehension to convert each line to an integer
        job_ids = [int(line.strip()) for line in file]

    # Now 'numbers' contains the integers as a list
    print(f"unfinished job_ids are: {job_ids}\n")
    print(f"Checking for jobs that are still undone...\n")
    print(f"Need to wait\n")

    slurm_jobs = []
    for job_id in job_ids:
        slurm_job = SLURM_Job()
        slurm_job.job_id = job_id
        if slurm_job.status() == 'FINISHED':
            continue
        print(f"Unfinished job: {job_id}\n")
        slurm_jobs.append(slurm_job)

    #Monitor these jobs#
    monitor_jobs(slurm_jobs)
    print("All previous jobs are finished\n")
'''

def add_mix_basis_for_atom(element, index, mix_lot, package):

    found = False
    count = 0
    for a in range(0, len(mix_lot)):
        first_element = mix_lot[a][0]

        # Zhao's note: there may be cases where you impose 2 basis sets on the same atom via different ways
        # e.g. H, STO-3G, H31, def2-TZVP
        # in this case, the one with element + index should have higher hierarchy than just element alone.
        # so sort mix_lot and get those with element + index as the first ones to be checked
        if (any(x.isalpha() for x in first_element) and (any(x.isnumeric() for x in first_element))):
            # if it is alphanumeric, check if element+number matches
            if element+str(index) == first_element:
                found = True
                count = a
                break
        elif first_element.isalpha():
            # check if element match
            if element == first_element:
                found = True
                count = a
                break
        elif first_element.isnumeric():
            # check if number match
            if index == int(first_element):
                found = True
                count = a
                break
    if found:
        mix_info = ''
        # check if quote mark is in the string, add if not, just for ORCA
        if package == "ORCA" and not (mix_lot[count][1].startswith("\"") and mix_lot[count][1].endswith("\"")):
            mix_lot[count][1] = "\"" + mix_lot[count][1] + "\""

        if package == "ORCA": mix_info = f"newgto {mix_lot[count][1]} end"
        elif package == "Gaussian": mix_info = [mix_lot[count][1], index]
        print(f"mix_information: {mix_info}\n", flush = True)
        return mix_info
    else:
        return ''

# add atom numbers to molecule
def addAtomIndices(mol):
    for i, a in enumerate(mol.GetAtoms()):
        a.SetAtomMapNum(i)

def MatchAtomIndices(mol_from, mol_to):
    assert(mol_to.GetNumAtoms() == mol_from.GetNumAtoms())
    match = mol_to.GetSubstructMatch(mol_from)
    assert(match)
    assert(len(match) == mol_from.GetNumAtoms())
    mol_to = Chem.RenumberAtoms(mol_to, match)
    return mol_to
def ReadMoleculeFromMolFile(mol_file_name):
    molecule = Chem.MolFromMolFile(mol_file_name, removeHs = False)
    return molecule

def Get_Atoms_Chirality(mol, SelectedChiralCenter):
    tags = []
    for center in SelectedChiralCenter:
        center_atom = mol.GetAtoms()[center]
        tag = str(center_atom.GetChiralTag())
        tags.append(tag)
    return tags

###################################################################
# the purpose of this function is to enable isomer enumeration   ##
# By using smiles, enumeration can be based on the smiles string ##
###################################################################
#def mol_to_xyz_to_smiles_to_mol(molecule, name):
# taking in 2 inputs: reactant.mol or product.mol
def Prepare_mol_file_to_xyz_smiles_for_chiralEnum(molfile_name):
    name = molfile_name.split(".mol")[0]
    os.system(f"obabel {name}.mol -O {name}.xyz > NUL 2>&1")
    os.system(f"obabel {name}.xyz -O {name}.mol > NUL 2>&1")
    P_molecule = ReadMoleculeFromMolFile(f"{name}.mol")
    os.system(f"obabel {name}.xyz -O {name}.mol > NUL 2>&1")
    molecule = ReadMoleculeFromMolFile(f"{name}.mol")

    ps = Chem.SmilesParserParams()
    ps.removeHs = False

    os.system(f"obabel {name}.xyz -O {name}.smi > NUL 2>&1")
    with open(f"{name}.smi", 'r') as f:
        first_line = f.readline().strip()   # Read the first line and strip any leading/trailing whitespace
        obabel_smiles = first_line.split()[0]  # Split the line based on whitespace and get the first element
    print(f"obabel_smiles: {obabel_smiles}\n")
    molecule_from_smiles = Chem.MolFromSmiles(obabel_smiles, ps)
    molecule_from_smiles = Chem.rdmolops.AddHs(molecule_from_smiles, addCoords = True)
    print(f"{name}_smiles: {obabel_smiles}, numberofAtoms: {molecule_from_smiles.GetNumAtoms()}\n")
    molecule_from_smiles = MatchAtomIndices(molecule, molecule_from_smiles)
    addAtomIndices(molecule_from_smiles)

    return molecule_from_smiles

def Generate_Isomers(mol, SelectedChiralCenter):
    for center in SelectedChiralCenter:
        center_atom = mol.GetAtoms()[center]
        if(str(center_atom.GetChiralTag()) == 'CHI_UNSPECIFIED'):
            print(f"You specified chiral center is unspecified! CHECK!\n")
            exit()
        mol.GetAtomWithIdx(center).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    opts = StereoEnumerationOptions(onlyUnassigned=True,
                                    rand=0xf00d,
                                    tryEmbedding=True, unique=True)
    isomers = tuple(EnumerateStereoisomers(mol, options = opts))
    for i, iso in enumerate(isomers):
        smi = Chem.MolToSmiles(iso, isomericSmiles=True)
        print(f"isomer {i}, isomer smiles: {smi}\n")
    print(f"# of stereoisomers: {len(isomers)}\n")
    return isomers
def Write_Isomers(isomers, name):
    for i, x in enumerate(isomers):
        smi = Chem.MolToSmiles(x, isomericSmiles=True)
        print(f"smile for x: {smi}\n")
        # ff = AllChem.UFFGetMoleculeForceField(x)
        # ff.Initialize()
        # ff.Minimize(energyTol=1e-7,maxIts=100000)

        Chem.rdmolfiles.MolToXYZFile(x, f"{name}-{i}.xyz")

def geometry_opt(molecule):
    '''
    Perform UFF level geometry optimization on YARP molecule

    Parameters:
    ----------
    molecule : yarpecule object
            molecule to be optimized 

    Returns
    -------
    molecule : yarpecule object
            optimized molecule
    '''

    # Write yarpecule object to a temporary mol file
    mol_file='.tmp.mol'
    mol_write_yp(mol_file, molecule, append_opt=False)
    
    # Use openbabel to perform geometry optimization
    mol=next(pybel.readfile("mol", mol_file))
    mol.localopt(forcefield='uff')
    
    # Convert mol file to yarpecule object, delete mol file and return yarpecule
    for count_i, i in enumerate(molecule.geo):
        molecule.geo[count_i]=mol.atoms[count_i].coords
    
    os.system("rm {}".format(mol_file))
    
    return molecule

def opt_geo_xtb(elements, geo, bond_mat, q=0, filename='tmp'):
    '''
    Apply xTB to find product/reactant geometry from reactant/product geometry.
    elements: the elements for geo (a list)
    geo: the geometry of product or reactant
    bond_mat: the bond electron matrix for reactant or product
    q: the charge state
    '''
    tmp_xyz_file=f".{filename}.xyz"
    tmp_inp_file=f".{filename}.inp"
    bond=return_bond_info(bond_mat)
    length=[]
    constraints=[]
    xyz_write(tmp_xyz_file, elements, geo)
    for i in bond: length.append(el_radii[elements[i[0]]]+el_radii[elements[i[1]]])
    for count_i, i in enumerate(bond): constraints+=[(i[0]+1, i[1]+1, length[count_i])]
    #print("A")
    optjob = XTB(input_geo=tmp_xyz_file,work_folder='.',jobtype=['opt'],jobname='opt',charge=q) 
    optjob.add_command(distance_constraints=constraints, force_constant=1.0)
    optjob.execute()
    # print(optjob.optimization_success())  
    if optjob.optimization_success():
        _, Gr = optjob.get_final_structure()
        print(Gr)
    else:
        print("XTB fails to locate reactant/product pair for this conformer.")
        return []
    adj_mat_o = bondmat_to_adjmat(bond_mat)
    adj_mat_n = table_generator(elements, Gr)
    
    try:
        files=[i for i in os.listdir(".") if fnmatch.fnmatch(i, f".{filename}*")]
        for i in files: os.remove(i)
    except:
        pass

    if np.abs(adj_mat_o-adj_mat_n).sum() == 0:
        return G            
    else:
       print("XTB fails to locate reactant/product pair for this conformer.")
       return []   
# def generate_xtb_constraint(bond, length, filename=".tmp.inp")
def return_bond_info(mat):
    info=[]
    for i in range(len(mat)-1):
        for j in range(i+1, len(mat)):
            if mat[i][j]>0:
                info+=[(i, j)]
    return info
def opt_geo(elements,geo,bond_mat,q=0,ff='mmff94',step=1000,filename='tmp',constraints=[]):
    ''' 
    Apply openbabel to perform force field geometry optimization 
    Will support constraints option in the near future 
    '''
    # Write a temporary molfile for obminimize to use
    tmp_filename = '.{}.mol'.format(filename)
    tmp_xyz_file = '.{}.xyz'.format(filename)
    count = 0
    while os.path.isfile(tmp_filename):
        count += 1
        if count == 10:
            print("ERROR in opt_geo: could not find a suitable filename for the tmp geometry. Exiting...")
            return geo
        else:
            tmp_filename = ".{}".format(filename) + tmp_filename            

    counti = 0
    while os.path.isfile(tmp_xyz_file):
        counti += 1
        if counti == 10:
            print("ERROR in opt_geo: could not find a suitable filename for the tmp geometry. Exiting...")
            return geo
        else:
            tmp_xyz_file = ".{}".format(filename) + tmp_xyz_file

    # write down mol file
    mol_write(tmp_filename,elements,geo,bond_mat,q=q)
    
    # set up openbabel
    conv = ob.OBConversion()
    conv.SetInAndOutFormats('mol','xyz')
    mol = ob.OBMol()
    conv.ReadFile(mol,tmp_filename)

    # Setup the force field with the constraints
    forcefield = ob.OBForceField.FindForceField(ff)
    success = forcefield.Setup(mol)
    if not success:
        forcefield = ob.OBForceField.FindForceField("uff")
        forcefield.Setup(mol)
    #forcefield.Setup(mol, constraints)
    #forcefield.SetConstraints(constraints)

    # Do a given number of steps conjugate gradient minimiazation and save the coordinates to mol.
    forcefield.ConjugateGradients(step)
    forcefield.GetCoordinates(mol)
    # Write the mol to a file
    conv.WriteFile(mol,tmp_xyz_file)

    _,G = xyz_parse(tmp_xyz_file)
    # Remove the tmp file that was read by obminimize
    try:
        files=[i for i in os.listdir(".") if fnmatch.fnmatch(i, f".{filename}*")]
        for i in files: os.remove(i)
    except:
        pass

    # check if geo opt returns desired geometry
    adj_mat_o = bondmat_to_adjmat(bond_mat)
    adj_mat_n = table_generator(elements, G)
    if np.abs(adj_mat_o-adj_mat_n).sum() == 0:
        return G
    else:
        # print(adj_mat_o-adj_mat_n)
        print("Error: geometry optimization by uff is failed.")
        return []

def bondmat_to_adjmat(bond_mat):
    adj_mat=copy.deepcopy(bond_mat)
    for count_i, i in enumerate(bond_mat):
        for count_j, j in enumerate(i):
            if j and count_i!=count_j: adj_mat[count_i][count_j]=1.0
            if count_i==count_j: adj_mat[count_i][count_i]=0.0
    return adj_mat

def mol_write(name, elements, geo, bond_mat, q=0, append_opt=False):
    adj_mat=bondmat_to_adjmat(bond_mat)
    if len(elements) >= 1000:
        print( "ERROR in mol_write: the V2000 format can only accomodate up to 1000 atoms per molecule.")
        return
    mol_dict={3:1, 2:2, 1:3, -1:5, -2:6, -3:7, 0:0}
    # Check for append vs overwrite condition
    if append_opt == True:
        open_cond = 'a'
    else:
        open_cond = 'w'

    # Parse the basename for the mol header
    base_name = name.split(".")
    if len(base_name) > 1:
        base_name = ".".join(base_name[:-1])
    else:
        base_name = base_name[0]

    keep_lone=[count_i for count_i, i in enumerate(bond_mat) if i[count_i]%2==1]
    # deal with radicals
    fc = list(return_formals(bond_mat, elements))
    # deal with charges 
    chrg = len([i for i in fc if i != 0])
    valence=[] # count the number of bonds for mol file
    for count_i, i in enumerate(bond_mat):
        bond=0
        for count_j, j in enumerate(i):
            if count_i!=count_j: bond=bond+int(j)
        valence.append(bond)
    # Write the file
    with open(name,open_cond) as f:
        # Write the header
        f.write('{}\nGenerated by mol_write.py\n\n'.format(base_name))

        # Write the number of atoms and bonds
        f.write("{:>3d}{:>3d}  0  0  0  0  0  0  0  0  1 V2000\n".format(len(elements),int(np.sum(adj_mat/2.0))))

        # Write the geometry
        for count_i,i in enumerate(elements):
            f.write(" {:> 9.4f} {:> 9.4f} {:> 9.4f} {:<3s} 0 {:>2d}  0  0  0  {:>2d}  0  0  0  0  0  0\n".format(geo[count_i][0],geo[count_i][1],geo[count_i][2], i.capitalize(), mol_dict[fc[count_i]], valence[count_i]))
        # Write the bonds
        bonds = [ (count_i,count_j) for count_i,i in enumerate(adj_mat) for count_j,j in enumerate(i) if j == 1 and count_j > count_i ]
        for i in bonds:

            # Calculate bond order from the bond_mat
            bond_order = int(bond_mat[i[0],i[1]])
            
            # add fix of bond order for dative bonds around the transition metal
            bond_elements = [elements[i[0]],elements[i[1]]]
            '''
            #print(f"bond_elements: {bond_elements}", flush = True)
            if (_ in el_metals for _ in bond_elements):# and (bond_order == 0):
                #print(f"FOUND METAL! bond_elements: {bond_elements}", flush = True)
                bond_order = 1
            '''
            if bond_order==0: bond_order=1
            f.write("{:>3d}{:>3d}{:>3d}  0  0  0  0\n".format(i[0]+1,i[1]+1,bond_order))

        # write radical info if exist
        if len(keep_lone) > 0:
            if len(keep_lone) == 1:
                f.write("M  RAD{:>3d}{:>4d}{:>4d}\n".format(1,keep_lone[0]+1,2))
            elif len(keep_lone) == 2:
                f.write("M  RAD{:>3d}{:>4d}{:>4d}{:>4d}{:>4d}\n".format(2,keep_lone[0]+1,2,keep_lone[1]+1,2))
            else:
                print("Only support one/two radical containing compounds, radical info will be skip in the output mol file...")

        if chrg > 0:
            if chrg == 1:
                charge = [i for i in fc if i != 0][0]
                f.write("M  CHG{:>3d}{:>4d}{:>4d}\n".format(1,fc.index(charge)+1,int(charge)))
            else:
                info = ""
                fc_counter = 0
                for count_c,charge in enumerate(fc):
                    if charge != 0:
                        if(fc_counter % 8 == 0): #Only 8 items a line#
                            info += "\nM  CHG{:>3d}".format(chrg - fc_counter if chrg - fc_counter <= 8 else 8)
                        info += '{:>4d}{:>4d}'.format(count_c+1,int(charge))
                        fc_counter += 1
                info += '\n'
                f.write(info)

        f.write("M  END\n$$$$\n")

    #with open(name, 'r') as file:
        # Read the content of the file
        #file_content = file.read()
        # Print the content
        #print(f"{file_content}")
    return

def xyz_write(name, element, geo, append_opt=False):
    if append_opt==False: out=open(name, 'w+')
    else: out=open(name, 'a+')
    out.write('{}\n\n'.format(len(element)))
    for count_i, i in enumerate(element):
        out.write('{} {} {} {}\n'.format(i, geo[count_i][0], geo[count_i][1], geo[count_i][2]))
    out.close()
    return

def mol_write_yp(name,molecule,append_opt=False):
    elements=molecule.elements
    geo=molecule.geo
    bond_mat=molecule.bond_mats[0]
    q=molecule.q
    adj_mat=molecule.adj_mat
    # Consistency check
    if len(elements) >= 1000:
        print( "ERROR in mol_write: the V2000 format can only accomodate up to 1000 atoms per molecule.")
        return
    mol_dict={3:1, 2:2, 1:3, -1:5, -2:6, -3:7, 0:0}
    # Check for append vs overwrite condition
    if append_opt == True:
        open_cond = 'a'
    else:
        open_cond = 'w'

    # Parse the basename for the mol header
    base_name = name.split(".")
    if len(base_name) > 1:
        base_name = ".".join(base_name[:-1])
    else:
        base_name = base_name[0]

    keep_lone=[count_i for count_i, i in enumerate(bond_mat) if i[count_i]%2==1]
    # deal with radicals
    fc = list(return_formals(bond_mat, elements))
    # deal with charges 
    chrg = len([i for i in fc if i != 0])
    valence=[] # count the number of bonds for mol file
    for count_i, i in enumerate(bond_mat):
        bond=0
        for count_j, j in enumerate(i):
            if count_i!=count_j: bond=bond+int(j)
        valence.append(bond)
    # Write the file
    with open(name,open_cond) as f:
        # Write the header
        f.write('{}\nGenerated by mol_write.py\n\n'.format(base_name))

        # Write the number of atoms and bonds
        f.write("{:>3d}{:>3d}  0  0  0  0  0  0  0  0  1 V2000\n".format(len(elements),int(np.sum(adj_mat/2.0))))

        # Write the geometry
        for count_i,i in enumerate(elements):
            f.write(" {:> 9.4f} {:> 9.4f} {:> 9.4f} {:<3s} 0 {:>2d}  0  0  0  {:>2d}  0  0  0  0  0  0\n".format(geo[count_i][0],geo[count_i][1],geo[count_i][2], i.capitalize(), mol_dict[fc[count_i]], valence[count_i]))

        # Write the bonds
        bonds = [ (count_i,count_j) for count_i,i in enumerate(adj_mat) for count_j,j in enumerate(i) if j == 1 and count_j > count_i ] 
        for i in bonds:

            # Calculate bond order from the bond_mat
            bond_order = int(bond_mat[i[0],i[1]])

            # add fix of bond order for dative bonds around the transition metal
            bond_elements = [elements[i[0]],elements[i[1]]]

            #print(f"bond_elements: {bond_elements}", flush = True)
            #if (_ in el_metals for _ in bond_elements):# and (bond_order == 0):
            #    #print(f"FOUND METAL! bond_elements: {bond_elements}", flush = True)
            #    bond_order = 1

            if bond_order==0: bond_order=1
            f.write("{:>3d}{:>3d}{:>3d}  0  0  0  0\n".format(i[0]+1,i[1]+1,bond_order))

        # write radical info if exist
        if len(keep_lone) > 0:
            if len(keep_lone) == 1:
                f.write("M  RAD{:>3d}{:>4d}{:>4d}\n".format(1,keep_lone[0]+1,2))
            elif len(keep_lone) == 2:
                f.write("M  RAD{:>3d}{:>4d}{:>4d}{:>4d}{:>4d}\n".format(2,keep_lone[0]+1,2,keep_lone[1]+1,2))
            else:
                print("Only support one/two radical containing compounds, radical info will be skip in the output mol file...")

        if chrg > 0:
            if chrg == 1:
                charge = [i for i in fc if i != 0][0]
                f.write("M  CHG{:>3d}{:>4d}{:>4d}\n".format(1,fc.index(charge)+1,int(charge)))
            else:
                info = ""
                fc_counter = 0
                for count_c,charge in enumerate(fc):
                    if charge != 0:
                        if(fc_counter % 8 == 0): #Only 8 items a line#
                            info += "M  CHG{:>3d}".format(chrg - fc_counter if chrg - fc_counter <= 8 else 8)
                        info += '{:>4d}{:>4d}'.format(count_c+1,int(charge))
                        fc_counter += 1
                info += '\n'
                f.write(info)

        f.write("M  END\n$$$$\n")

    return

def return_smi(E,G,bond_mat=None,namespace='obabel'):
    ''' Function to Return smiles string using openbabel (pybel) '''
    if bond_mat is None:
        xyz_write(f"{namespace}_input.xyz",E,G)
        # Read the XYZ file using Open Babel
        molecule = next(pybel.readfile("xyz", f"{namespace}_input.xyz"))
        # Generate the canonical SMILES string directly
        smile = molecule.write(format="can").strip().split()[0]
        # Clean up the temporary file
        os.remove(f"{namespace}_input.xyz")
        return smile

    else:
        mol_write(f"{namespace}_input.mol",E,G,bond_mat)
        # Read the mol file using Open Babel
        molecule = next(pybel.readfile("mol", f"{namespace}_input.mol"))
        # Generate the canonical SMILES string directly
        smile = molecule.write(format="can").strip().split()[0]
        # Clean up the temporary file
        os.remove(f"{namespace}_input.mol")

    return smile

def return_smi_mol(E,G,bond_mat,molecule_name):
    ''' Function to Return smiles string using openbabel (pybel) '''
    mol_file_name = f"{molecule_name}.mol"
    mol_write(mol_file_name,E,G,bond_mat)
    # Read the mol file using Open Babel
    molecule = next(pybel.readfile("mol", mol_file_name))
    # Generate the canonical SMILES string directly
    smile = molecule.write(format="can").strip().split()[0]
    # Clean up the temporary file
    #os.remove(mol_file_name)

    return smile

def return_smi_yp(molecule, namespace="obabel"):
    mol_write_yp(f"{namespace}_input.mol",molecule)
    mol=next(pybel.readfile("mol", f"{namespace}_input.mol"))
    smile=mol.write(format="can").strip().split()[0]
    os.remove(f"{namespace}_input.mol")
    return smile

def return_rxn_constraint(mol1, mol2):
    adj1=mol1.adj_mat
    adj2=mol2.adj_mat
    bond_change=[]
    d_adj=np.abs(adj2-adj1)
    for i in range(len(mol1.elements)):
        for j in range(i+1, len(mol1.elements)):
            if d_adj[i][j]!=0: bond_change+=[(i, j)]
    reactive_atoms=list(set([atom for bond in bond_change for atom in bond]))
    # if there are other atoms next to at least two reactive atom in either Reactant or Product, identify them also as reactive atoms
    gs1=graph_seps(adj1)
    gs2=graph_seps(adj2)
    n1=Counter([indj for indj in range(len(mol1.elements)) for indi in reactive_atoms if gs1[indi][indj]==1])
    n2=Counter([indj for indj in range(len(mol1.elements)) for indi in reactive_atoms if gs2[indi][indj]==1])
    reactive_atoms+=list(set([ind for ind, count in n1.items() if count>1]+[ind for ind, count in n2.items() if count>1]))
    
    return bond_change, reactive_atoms

def return_all_constraint(molecule):
    adj_mat=molecule.adj_mat
    elements=molecule.elements
    dis_constraint=[]
    dist_mat = np.triu(cdist(molecule.geo, molecule.geo))
    for count_e, e in enumerate(elements):
        for count_i, i in enumerate(adj_mat[count_e]):
            if i and count_e < count_i:
                dis_constraint.append([count_e+1, count_i+1, dist_mat[count_e, count_i]])
    return dis_constraint

def return_metal_constraint(molecule):
    # this function will return the bond constraint for metallic bonds
    adj_mat=molecule.adj_mat
    elements=molecule.elements
    dis_constraint=[]
    metal_list=['li', 'be',\
                'na', 'mg', 'al',\
                'k', 'ca', 'sc', 'ti', 'v', 'cr', 'mn', 'fe', 'co', 'ni', 'cu', 'zn', 'ga', 'ge',\
                'rb', 'sr', 'y', 'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag', 'cd', 'in', 'sn', 'sb',\
                'cs', 'ba', 'lu', 'hf', 'ta', 'w', 're', 'os', 'ir', 'pt', 'au', 'hg', 'tl', 'pb', 'bi', 'po',\
                'fr', 'ra', 'lr', 'rf', 'db', 'sg', 'bh', 'hs', 'mt', 'ds', 'rg', 'cn',\
                'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy', 'ho', 'er', 'tm', 'yb',\
                'ac', 'th', 'pa', 'u', 'np', 'pu', 'am', 'cm', 'bk', 'cf', 'es', 'fm', 'md', 'no']
    for count_e, e in enumerate(elements):
        if e.lower() in metal_list:
            for count_i, i in enumerate(adj_mat[count_e]):
                if i:
                    dis_constraint.append([count_e+1, count_i+1, el_radii[e.capitalize()]+el_radii[elements[count_i].capitalize()]])
    return dis_constraint

def return_model_rxn(reaction, depth=1):
    # This function is written by Hsuan-Hao Hsu (hsu205@purdue.edu)
    # Read in a true reaction and return a reaction class of model reaction
    elements=reaction.reactant.elements
    R_adj=reaction.reactant.adj_mat
    P_adj=reaction.product.adj_mat
    R_bond=reaction.reactant.bond_mats[0]
    P_bond=reaction.product.bond_mats[0]
    for ind in range(len(bond_mat_2)):
        BE_change=P_bond-R_bond
    print(BE_change)
    bond_break=[]
    bond_form=[]
    return

def return_inchikey(molecule, verbose = False):
    """
    Generate the InChIKey for a given molecule using OpenBabel.
    
    Parameters
    ----------
    molecule : yarpecule object

    Returns
    -------
    inchikey : str
    """
    E=molecule.elements
    G=molecule.geo
    bond_mat=molecule.bond_mats[0]
    q=molecule.q # ERM: not used

    gs=graph_seps(molecule.adj_mat)
    #print(f"molecule.adj_mat: {molecule.adj_mat}\n")
    #print(f"gs: {gs}\n")

    adj_mat=molecule.adj_mat
    groups=[]
    loop_ind=[]
    for i in range(len(gs)):
        if i not in loop_ind:
            new_group=[count_j for count_j, j in enumerate(gs[i, :]) if j>=0]
            loop_ind += new_group
            groups+=[new_group]
    inchikey=[]
    
    mol=copy.deepcopy(molecule) # <-- ERM: is this repetitive with the below for loop?
    #Zhao's note: this seems to generate quite different inchikey if you write to xyz file, need to see why#
    #They do result in different inchikeys, must be the bonding info, consider changing it in sep_mols
    for group in groups:
        N_atom=len(group)
        mol=copy.deepcopy(molecule)
        mol.elements=[E[ind] for ind in group]
        mol.bond_mats=[bond_mat[group][:, group]]
        mol.geo=np.zeros([N_atom, 3])
        mol.adj_mat=adj_mat[group][:, group]
        
        for count_i, i in enumerate(group): mol.geo[count_i, :]=G[i, :]
        
        mol_write_yp(".tmp.mol", mol)

        if verbose: print(os.popen('cat .tmp.mol').read())

        mol=next(pybel.readfile("mol", ".tmp.mol"))
        try:
            inchi=mol.write(format='inchikey').strip().split()[0]
            #print(f"inchi: {inchi}\n")
        except:
            print(f"{mol.write(format='inchikey')}")
            continue
        inchikey+=[inchi]
        os.system("rm .tmp.mol")
    
    
    if len(inchikey)==0:
        return "ERROR"
    elif len(groups) == 1:
        return inchikey[0][:14]
    else:
        return '-'.join(sorted([i[:14] for i in inchikey]))                   
