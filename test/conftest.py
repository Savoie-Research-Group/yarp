"""
Persistent test fixtures for YARP testing suite
"""
import yaml
import pytest
from pathlib import Path
import pickle

# YAML input files
@pytest.fixture
def no_initial_struct(tmp_path):
    # 1. Load the real YAML
    yaml_path = Path(__file__).parent / "main_inputs" / "invalid" / "no_initial_struct.yaml"
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # 2. Define the safe temporary output path
    safe_output = tmp_path / "test_no_initial_struct_output.pkl"
    safe_status = tmp_path / "test_no_initial_struct_STATUS.json"

    # 3. Overwrite the nested key
    if 'initialize' in data:
        data['initialize']['output'] = str(safe_output)
        data['initialize']['status'] = str(safe_status)
    else:
        # Fallback if the YAML structure changes in the future
        pytest.fail("The input YAML does not contain an 'initialize' block.")

    return data

@pytest.fixture
def species_noenum(tmp_path):
    # 1. Load the real YAML
    yaml_path = Path(__file__).parent / "main_inputs" / "invalid" / "species_noenum.yaml"
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # 2. Define the safe temporary output path
    safe_output = tmp_path / "test_species_noenum_output.pkl"
    safe_status = tmp_path / "test_species_noenum_STATUS.json"

    # 3. Overwrite the nested key
    if 'initialize' in data:
        data['initialize']['output'] = str(safe_output)
        data['initialize']['status'] = str(safe_status)
    else:
        # Fallback if the YAML structure changes in the future
        pytest.fail("The input YAML does not contain an 'initialize' block.")

    return data

@pytest.fixture
def slurm_no_queue(tmp_path):
    # 1. Load the real YAML
    yaml_path = Path(__file__).parent / "main_inputs" / "invalid" / "slurm_no_queue.yaml"
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # 2. Define the safe temporary output path
    safe_output = tmp_path / "test_slurm_no_queue_output.pkl"
    safe_status = tmp_path / "test_slurm_no_queue_STATUS.json"

    # 3. Overwrite the nested key
    if 'initialize' in data:
        data['initialize']['output'] = str(safe_output)
        data['initialize']['status'] = str(safe_status)
    else:
        # Fallback if the YAML structure changes in the future
        pytest.fail("The input YAML does not contain an 'initialize' block.")

    return data

@pytest.fixture
def sge_no_queue(tmp_path):
    # 1. Load the real YAML
    yaml_path = Path(__file__).parent / "main_inputs" / "invalid" / "sge_no_queue.yaml"
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # 2. Define the safe temporary output path
    safe_output = tmp_path / "test_sge_no_queue_output.pkl"
    safe_status = tmp_path / "test_sge_no_queue_STATUS.json"

    # 3. Overwrite the nested key
    if 'initialize' in data:
        data['initialize']['output'] = str(safe_output)
        data['initialize']['status'] = str(safe_status)
    else:
        # Fallback if the YAML structure changes in the future
        pytest.fail("The input YAML does not contain an 'initialize' block.")

    return data

@pytest.fixture
def enum_egat_llpath_llrefine(tmp_path):
    # 1. Load the real YAML
    yaml_path = Path(__file__).parent / "main_inputs" / "enum_egat_llpath_llrefine.yaml"
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # 2. Define the safe temporary output path
    safe_output = tmp_path / "test_enum_egat_llpath_llrefine_output.pkl"
    safe_status = tmp_path / "test_enum_egat_llpath_llrefine_STATUS.json"

    # 3. Overwrite the nested key
    if 'initialize' in data:
        data['initialize']['output'] = str(safe_output)
        data['initialize']['status'] = str(safe_status)
    else:
        # Fallback if the YAML structure changes in the future
        pytest.fail("The input YAML does not contain an 'initialize' block.")

    return data


@pytest.fixture
def enum_min_options(tmp_path):
    # 1. Load the real YAML
    yaml_path = Path(__file__).parent / "main_inputs" / "enum_min_options.yaml"
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # 2. Define the safe temporary output path
    safe_output = tmp_path / "test_enum_min_options_output.pkl"
    safe_status = tmp_path / "test_enum_min_options_STATUS.json"

    # 3. Overwrite the nested key
    if 'initialize' in data:
        data['initialize']['output'] = str(safe_output)
        data['initialize']['status'] = str(safe_status)
    else:
        # Fallback if the YAML structure changes in the future
        pytest.fail("The input YAML does not contain an 'initialize' block.")

    return data

@pytest.fixture
def enum_full_options(tmp_path):
    # 1. Load the real YAML
    yaml_path = Path(__file__).parent / "main_inputs" / "enum_full_options.yaml"
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # 2. Define the safe temporary output path
    safe_output = tmp_path / "test_enum_full_options_output.pkl"
    safe_status = tmp_path / "test_enum_full_options_STATUS.json"

    # 3. Overwrite the nested key
    if 'initialize' in data:
        data['initialize']['output'] = str(safe_output)
        data['initialize']['status'] = str(safe_status)
    else:
        # Fallback if the YAML structure changes in the future
        pytest.fail("The input YAML does not contain an 'initialize' block.")

    return data

# Pickle files
@pytest.fixture
def glucose_single_path():
    """Returns a dictionary object of the reactions contained in glucose pickle file."""
    file = str(Path(__file__).parent / "pickles" / "glucose_single_path.pkl")
    return pickle.load(open(file, 'rb'))

@pytest.fixture
def glucose_multi_path():
    """Returns a dictionary object of the reactions contained in glucose pickle file."""
    file = str(Path(__file__).parent / "pickles" / "glucose_multi_path.pkl")
    return pickle.load(open(file, 'rb'))

@pytest.fixture
def d1_path():
    """Returns a string object to"""
    return str(Path(__file__).parent / "pickles" / "haa_heavy_b2f2_d1.pkl")

@pytest.fixture
def khp_d1():
    """Returns a dictionary object of the reactions contained in 3HP b2f2 (depth 1) pickle file."""
    file = str(Path(__file__).parent / "pickles" / "3hp_heavy_b2f2_d1.pkl")
    return pickle.load(open(file, 'rb'))

# Molecule files
@pytest.fixture
def ethene_xyz():
    """Returns a string object of the absolute path to ethene XYZ file."""
    return str(Path(__file__).parent / "molecules" / "ethene.xyz")

@pytest.fixture
def ammonium_xyz():
    """Returns a string object of the absolute path to ammonium XYZ file."""
    return str(Path(__file__).parent / "molecules" / "ammonium.xyz")

@pytest.fixture
def nitrate_xyz():
    """Returns a string object of the absolute path to nitrate XYZ file."""
    return str(Path(__file__).parent / "molecules" / "nitrate.xyz")

@pytest.fixture
def ethanol_mol():
    """Returns a string object of the absolute path to ethanol MOL file."""
    return str(Path(__file__).parent / "molecules" / "ethanol.mol")

@pytest.fixture
def acetate_mol():
    """Returns a string object of the absolute path to acetate MOL file."""
    return str(Path(__file__).parent / "molecules" / "acetate.mol")

@pytest.fixture
def betaine_mol():
    """Returns a string object of the absolute path to betaine MOL file."""
    return str(Path(__file__).parent / "molecules" / "betaine.mol")


@pytest.fixture
def test_xyz_dir():
    """Returns a Path object of the absolute path to the test XYZ directory."""
    return Path(__file__).parent / "molecules" / "batch_xyz_rxn"


@pytest.fixture
def test_smiles_file():
    """Returns a Path object of the absolute path to the test SMILES file."""
    return Path(__file__).parent / "molecules" / "batch_SMILES_rxn.txt"

# SMILES
@pytest.fixture
def ethene_smi():
    """Returns the smiles string for ethene."""
    return "C=C"

@pytest.fixture
def haa_canon_smi():
    return 'O=CCO'

@pytest.fixture
def haa_full_map_smi():
    return '[C:0]([C:1]([O:7][H:11])([H:13])[H:14])(=[O:6])[H:12]'

@pytest.fixture
def haa_heavy_map_smi():
    return '[CH:1]([CH2:4][OH:3])(=[O:2])'

@pytest.fixture
def haa_heavy_map_explicitH_smi():
    return '[C:1]([C:4]([O:3][H])([H])[H])(=[O:2])[H]'

@pytest.fixture
def rad_canon_smi():
    return '[CH2]C'

@pytest.fixture
def rad_canon_map_smi():
    return '[CH2:2][CH3:1]'

@pytest.fixture
def rad_explicitH_smi():
    return 'C([C]([H])[H])([H])([H])[H]'

@pytest.fixture
def rad_full_map_smi():
    return '[C:1]([C:2]([H:3])[H:7])([H:5])([H:6])[H:4]'

@pytest.fixture
def anion_canon_smi():
    return 'CC(=O)[O-]'

@pytest.fixture
def anion_canon_map_smi():
    return '[CH3:2][C:3](=[O:1])[O-:4]'

@pytest.fixture
def anion_explicitH_smi():
    return '[O-]C(=O)C([H])([H])[H]'

@pytest.fixture
def anion_full_map_smi():
    return '[O-:3][C:4](=[O:5])[C:6]([H:13])([H:14])[H:12]'

@pytest.fixture
def aromatic_canon_smi():
    return 'Oc1ccoc1'

@pytest.fixture
def aromatic_full_map_smi():
    return '[c:0]1([H:15])[c:4]([O:9][H:13])[c:5]([H:20])[c:1]([H:16])[o:6]1'

@pytest.fixture
def benzene_smi():
    return "c1ccccc1"

@pytest.fixture
def benz_rad_cat_smi():
    return "[CH]1C=CC=C[CH+]1"

#======= Find Lewis Structures Tests =========

# Ions
@pytest.fixture
def methyl_3_buteneium_xyz():
    """Returns a string object of the absolute path to 1-methyl-3-buteneium XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "ions" / "1-methyl-3-buteneium.xyz")

@pytest.fixture
def methyl_4_chloro_3_buteneium_xyz():
    """Returns a string object of the absolute path to 1-methyl-4-chloro-3-buteneium XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "ions" / "1-methyl-4-chloro-3-buteneium.xyz")

@pytest.fixture
def methyl_4_fluoro_3_buteneium_xyz():
    """Returns a string object of the absolute path to 1-methyl-4-fluoro-3-buteneium XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "ions" / "1-methyl-4-fluoro-3-buteneium.xyz")

@pytest.fixture
def propanenitrene_xyz():
    """Returns a string object of the absolute path to propanenitreneium XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "ions" / "2-propanenitrene.xyz")

@pytest.fixture
def cp_anion_xyz():
    """Returns a string object of the absolute path to cyclopentadienyl anion XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "ions" / "cp_anion.xyz")

@pytest.fixture
def cyclopropenium_xyz():
    """Returns a string object of the absolute path to cyclopropenium XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "ions" / "cyclopropenium.xyz")

@pytest.fixture
def ketonate_xyz():
    """Returns a string object of the absolute path to ketonate XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "ions" / "ketonate.xyz")

@pytest.fixture
def methylmethyleneoxide_xyz():
    """Returns a string object of the absolute path to methylmethyleneoxide XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "ions" / "methylmethyleneoxide.xyz")

@pytest.fixture
def propeneonate_xyz():
    """Returns a string object of the absolute path to propeneonate XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "ions" / "propeneonate.xyz")

@pytest.fixture
def pyrylium_xyz():
    """Returns a string object of the absolute path to pyrylium XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "ions" / "pyrylium.xyz")

@pytest.fixture
def toluenium_xyz():
    """Returns a string object of the absolute path to toluenium XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "ions" / "toluenium.xyz")

# Multi-Structure Systems
@pytest.fixture
def bimole1_far_xyz():
    """Returns a string object of the absolute path to bimole1_far XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "multi_structure" / "bimole_far.xyz")

@pytest.fixture
def bimole1_one_xyz():
    """Returns a string object of the absolute path to bimole1_one XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "multi_structure" / "bimole_one.xyz")

@pytest.fixture
def bimole1_two_xyz():
    """Returns a string object of the absolute path to bimole1_two XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "multi_structure" / "bimole_two.xyz")

@pytest.fixture
def bimole1_xyz():
    """Returns a string object of the absolute path to bimole1 XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "multi_structure" / "bimole.xyz")

# Radicals
@pytest.fixture
def methyl_3_butene_radical_xyz():
    """Returns a string object of the absolute path to 1-methyl-3-butene radical XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "radicals" / "1-methyl-3-butene_radical.xyz")

@pytest.fixture
def propene_radical_xyz():
    """Returns a string object of the absolute path to propene radical XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "radicals" / "propene_radical.xyz")

# Rings
@pytest.fixture
def adamantum_xyz():
    """Returns a string object of the absolute path to adamantum XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "rings" / "adamantum.xyz")

@pytest.fixture
def anthracene_xyz():
    """Returns a string object of the absolute path to anthracene XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "rings" / "anthracene.xyz")

@pytest.fixture
def azulene_xyz():
    """Returns a string object of the absolute path to azulene XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "rings" / "azulene.xyz")

@pytest.fixture
def benzene_xyz():
    """Returns a string object of the absolute path to benzene XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "rings" / "benzene.xyz")

@pytest.fixture
def biphenylene_xyz():
    """Returns a string object of the absolute path to biphenylene XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "rings" / "biphenylene.xyz")

@pytest.fixture
def bis_cyclohexane_xyz():
    """Returns a string object of the absolute path to bis-cyclohexane XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "rings" / "bis-cyclohexane.xyz")

@pytest.fixture
def napthalene_xyz():
    """Returns a string object of the absolute path to napthalene XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "rings" / "napthalene.xyz")

@pytest.fixture
def o_xylene_xyz():
    """Returns a string object of the absolute path to o-xylene XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "rings" / "o-xylene.xyz")

@pytest.fixture
def pyridine_aldehyde_xyz():
    """Returns a string object of the absolute path to pyridine_aldehyde XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "rings" / "pyridine_aldehyde.xyz")

@pytest.fixture
def thiophene_xyz():
    """Returns a string object of the absolute path to thiophene XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "rings" / "thiophene.xyz")

# Zwitterions
@pytest.fixture
def co_xyz():
    """Returns a string object of the absolute path to co XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "zwitterions" / "CO.xyz")

@pytest.fixture
def cyclopenteneamine_xyz():
    """Returns a string object of the absolute path to cyclopenteneamine XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "zwitterions" / "cyclopenteneamine.xyz")

@pytest.fixture
def diazomethane_xyz():
    """Returns a string object of the absolute path to diazomethane XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "zwitterions" / "diazomethane.xyz")

@pytest.fixture
def sulfoxide_xyz():
    """Returns a string object of the absolute path to sulfoxide XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "zwitterions" / "sulfoxide.xyz")

@pytest.fixture
def trimethyl_phosphine_oxide_xyz():
    """Returns a string object of the absolute path to trimethyl_phosphine_oxide XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "zwitterions" / "trimethyl_phosphine_oxide.xyz")

# "Special" Cases
@pytest.fixture
def bredt_xyz():
    """Returns a string object of the absolute path to bredt XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "special_cases" / "bredt.xyz")

@pytest.fixture
def decyne_xyz():
    """Returns a string object of the absolute path to decyne XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "special_cases" / "decyne.xyz")

@pytest.fixture
def sulfone_xyz():
    """Returns a string object of the absolute path to sulfone XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "special_cases" / "sulfone.xyz")

# Other
@pytest.fixture
def chloroform_xyz():
    """Returns a string object of the absolute path to chloroform XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "standard" / "chloroform.xyz")

@pytest.fixture
def ec_xyz():
    """Returns a string object of the absolute path to ec XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "standard" / "ec.xyz")

@pytest.fixture
def ester_xyz():
    """Returns a string object of the absolute path to ester XYZ file."""
    return str(Path(__file__).parent / "molecules" / "find_lewis_structures" / "standard" / "ester.xyz")
