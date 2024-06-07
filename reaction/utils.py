import sys, itertools, timeit, os, copy  
from openbabel import pybel
from openbabel import openbabel as ob
from collections import Counter    
import numpy as np
from scipy.spatial.distance import cdist
from yarp.taffi_functions import graph_seps,table_generator,return_rings,adjmat_to_adjlist,canon_order
from yarp.properties import el_to_an,an_to_el,el_mass, el_radii, el_metals
from yarp.find_lewis import find_lewis,return_formals,return_n_e_accept,return_n_e_donate,return_formals,return_connections,return_bo_dict
from yarp.hashes import atom_hash,yarpecule_hash
from yarp.input_parsers import xyz_parse,xyz_q_parse,xyz_from_smiles, mol_parse, xyz_parse_simple
from yarp.misc import merge_arrays, prepare_list
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import EnumerateStereoisomers, AllChem, TorsionFingerprints, rdmolops, rdDistGeom
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.ML.Cluster import Butina

class DFT_Input:
    def __init__(self, args):
        """
        Initialize an DFT input class
        a wrapper for DFT input keywords 
        including charge, multiplicity, solvation, level of theory, etc.
        """
        self.input_geo   = ""
        self.work_folder = os.getcwd()
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
    geometry optimization on yarp class
    '''
    mol_file='.tmp.mol'
    mol_write_yp(mol_file, molecule, append_opt=False)
    mol=next(pybel.readfile("mol", mol_file))
    mol.localopt(forcefield='uff')
    for count_i, i in enumerate(molecule.geo):
        molecule.geo[count_i]=mol.atoms[count_i].coords
    os.system("rm {}".format(mol_file))
    return molecule

def opt_geo(elements,geo,bond_mat,q=0,ff='mmff94',step=100,filename='tmp',constraints=[]):
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
        os.remove(tmp_filename)
        os.remove(tmp_xyz_file)
    except:
        pass

    # check if geo opt returns desired geometry
    adj_mat_o = bondmat_to_adjmat(bond_mat)
    adj_mat_n = table_generator(elements, G)
    if np.abs(adj_mat_o-adj_mat_n).sum() == 0:
        return G
    else:
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
            #print(f"xyz: {geo[count_i][0]}, {geo[count_i][1]}, {geo[count_i][2]}  ")
            #print(f"element: {i} ({i.capitalize()}, formal_charge: {mol_dict[fc[count_i]]}), valence: {valence[count_i]}\n")
            f.write(" {:> 9.4f} {:> 9.4f} {:> 9.4f} {:<3s} 0 {:>2d}  0  0  0  {:>2d}  0  0  0  0  0  0\n".format(geo[count_i][0],geo[count_i][1],geo[count_i][2], i.capitalize(), mol_dict[fc[count_i]], valence[count_i]))
        # Write the bonds
        bonds = [ (count_i,count_j) for count_i,i in enumerate(adj_mat) for count_j,j in enumerate(i) if j == 1 and count_j > count_i ]
        for i in bonds:

            # Calculate bond order from the bond_mat
            bond_order = int(bond_mat[i[0],i[1]])

            # add fix of bond order for dative bonds around the transition metal
            bond_elements = [elements[i[0]],elements[i[1]]]

            #print(f"bond_elements: {bond_elements}", flush = True)
            if (_ in el_metals for _ in bond_elements):# and (bond_order == 0):
                bond_order = 1
            
            if bond_order==0: bond_order=1

            #Zhao's note on 051424: strangely, this may not be able to handle metal-metal bond, for example, for the transmetalation case with Pd-Zn, 
            #it cannot handle Pd-Zn bond, and the mol file has a bond order of "-1", which cannot be processed into inchikey. So we can assign "1" here.
            if bond_order == -1: bond_order = 1

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

    return

# Zhao's note: need to fix #
def xyz_write(name, element, geo, append_opt=False):
    if append_opt==False: out=open(name, 'w+')
    else: out=open(name, 'a+')
    out.write('{}\n\n'.format(len(element)))
    for count_i, i in enumerate(element):
        #print(f"index: {count_i}, geo: {geo}", flush = True)
        #print(f"element: {i}, xyz: {geo[count_i]}\n", flush = True)
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
            if (_ in el_metals for _ in bond_elements):# and (bond_order == 0):
                #print(f"FOUND METAL! bond_elements: {bond_elements}", flush = True)
                bond_order = 1

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
    #Zhao's note: this seems to generate quite different inchikey if you write to xyz file, need to see why#
    #They do result in different inchikeys, must be the bonding info, consider changing it in sep_mols
    count = 0
    for group in groups:
        N_atom=len(group)
        mol=copy.deepcopy(molecule)
        mol.elements=[E[ind] for ind in group]
        mol.bond_mats=[bond_mat[group][:, group]]
        mol.geo=np.zeros([N_atom, 3])
        mol.adj_mat=adj_mat[group][:, group]
        for count_i, i in enumerate(group): mol.geo[count_i, :]=G[i, :]
        mol_write_yp(".tmp.mol", mol)
        mol=next(pybel.readfile("mol", ".tmp.mol"))
        inchi=mol.write(format='inchikey').strip().split()[0]
        inchikey+=[inchi]
        os.system("mv .tmp.mol " + inchi + ".mol")
        #os.system("rm .tmp.mol")
        count += 1

    if len(groups) == 1:
        return inchikey[0][:14]
    else:
        return '-'.join(sorted([i[:14] for i in inchikey]))                   
