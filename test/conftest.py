"""
Persistent test fixtures for YARP testing suite
"""
import yaml
import pytest
from pathlib import Path
import pickle
import omegaconf
from yarp.reaction.egat.predict_from_smiles import load_model

# YAML input files
@pytest.fixture
def enum_min_options(tmp_path):
    # 1. Load the real YAML
    yaml_path = Path(__file__).parent / "main_inputs" / "enum_min_options.yaml"
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # 2. Define the safe temporary output path
    safe_output = tmp_path / "test_enum_min_options_output.pkl"

    # 3. Overwrite the nested key
    if 'initialize' in data:
        data['initialize']['output'] = str(safe_output)
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

    # 3. Overwrite the nested key
    if 'initialize' in data:
        data['initialize']['output'] = str(safe_output)
    else:
        # Fallback if the YAML structure changes in the future
        pytest.fail("The input YAML does not contain an 'initialize' block.")

    return data

@pytest.fixture
def egat_min_options(tmp_path):
    # 1. Load the real YAML
    yaml_path = Path(__file__).parent / "main_inputs" / "egat_min_options.yaml"
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # 2. Define the safe temporary output path
    safe_output = tmp_path / "test_egat_min_options_output.pkl"

    # 3. Overwrite the nested key
    if 'initialize' in data:
        data['initialize']['output'] = str(safe_output)
    else:
        # Fallback if the YAML structure changes in the future
        pytest.fail("The input YAML does not contain an 'initialize' block.")

    return data

# Pytorch models
@pytest.fixture
def egat_csv():
    return str(Path(__file__).parent / "reaction" / "yarp_predictions.csv")

@pytest.fixture
def egat_pretrain():
    """Return pytorch model"""
    model, args = load_model('test/models/v1.pth', omegaconf.OmegaConf.load('test/models/auto0.yaml'))
    return model, args

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