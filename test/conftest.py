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

@pytest.fixture
def enum_d2_pkl(tmp_path):
    # 1. Load the real YAML
    yaml_path = Path(__file__).parent / "main_inputs" / "d2_pickle.yaml"
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # 2. Define the safe temporary output path
    safe_output = tmp_path / "test_enum_d2_pkl_output.pkl"
    safe_status = tmp_path / "test_enum_d2_pkl_STATUS.json"
    safe_source = Path(__file__).parent / "molecules/d1_3hp_b2f2_heavy/yarp_rxn.pkl"

    # 3. Overwrite the nested key
    if 'initialize' in data:
        data['initialize']['output'] = str(safe_output)
        data['initialize']['status'] = str(safe_status)
        data['initialize']['initial_structure']['source'] = str(safe_source)
    else:
        # Fallback if the YAML structure changes in the future
        pytest.fail("The input YAML does not contain an 'initialize' block.")

    return data

@pytest.fixture
def enum_d2_smi(tmp_path):
    # 1. Load the real YAML
    yaml_path = Path(__file__).parent / "main_inputs" / "d2_smi.yaml"
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # 2. Define the safe temporary output path
    safe_output = tmp_path / "test_enum_d2_pkl_output.pkl"
    safe_status = tmp_path / "test_enum_d2_pkl_STATUS.json"
    safe_source = Path(__file__).parent / "molecules/d1_3hp_b2f2_heavy/rxn_smi.txt"

    # 3. Overwrite the nested key
    if 'initialize' in data:
        data['initialize']['output'] = str(safe_output)
        data['initialize']['status'] = str(safe_status)
        data['initialize']['initial_structure']['source'] = str(safe_source)
    else:
        # Fallback if the YAML structure changes in the future
        pytest.fail("The input YAML does not contain an 'initialize' block.")

    return data

@pytest.fixture
def enum_d2_xyz_dir(tmp_path):
    # 1. Load the real YAML
    yaml_path = Path(__file__).parent / "main_inputs" / "d2_xyz_dir.yaml"
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # 2. Define the safe temporary output path
    safe_output = tmp_path / "test_enum_d2_pkl_output.pkl"
    safe_status = tmp_path / "test_enum_d2_pkl_STATUS.json"
    safe_source = Path(__file__).parent / "molecules/d1_3hp_b2f2_heavy/rxn_xyzs"

    # 3. Overwrite the nested key
    if 'initialize' in data:
        data['initialize']['output'] = str(safe_output)
        data['initialize']['status'] = str(safe_status)
        data['initialize']['initial_structure']['source'] = str(safe_source)
    else:
        # Fallback if the YAML structure changes in the future
        pytest.fail("The input YAML does not contain an 'initialize' block.")

    return data

# Pickle files
def _reaction_hash_cases(filename, expected_count):
    """Load a committed reaction-hash case set when its fixture is requested."""
    source = Path(__file__).parent / "pickles" / filename
    with source.open("rb") as stream:
        payload = pickle.load(stream)
    assert payload["version"] == 2
    cases = payload["cases"]
    assert len(cases) == expected_count
    return cases


@pytest.fixture(scope="module")
def reaction_hash_symmetry_cases():
    return _reaction_hash_cases("reaction_hash_symmetry.pkl", 200)


@pytest.fixture(scope="module")
def reaction_hash_nonisomorphic_cases():
    return _reaction_hash_cases("reaction_hash_nonisomorphic.pkl", 100)


@pytest.fixture(scope="module")
def reaction_hash_direction_cases():
    return _reaction_hash_cases("reaction_hash_direction.pkl", 100)


@pytest.fixture(scope="module")
def reaction_hash_network_reverse_cases():
    return _reaction_hash_cases("reaction_hash_network_reverses.pkl", 5)


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

@pytest.fixture
def khp2pp22_soergel_beam2_cyc3():
    """Returns a dictionary object of the reactions contained in khp2pp22 (depth 3) pickle file."""
    file = str(Path(__file__).parent / "pickles" / "khp2pp22_soergel_beam2_cyc3.pkl")
    return pickle.load(open(file, 'rb'))

# Enumerated products, shared across the geometry/determinism tests
@pytest.fixture(scope="session")
def khp_parent():
    """The ketohydroperoxide reactant, enumerated from in several test modules."""
    import yarp as yp
    return yp.yarpecule("O=CCCOO")

@pytest.fixture(scope="session")
def khp_products(khp_parent):
    """
    Break-2/form-2 products of O=CCCOO, keyed by canonical SMILES.

    Session-scoped because enumeration is the expensive part (~0.1 s) and
    several modules want the same products. Keying by SMILES rather than by
    position in the generator keeps the tests readable and keeps a failure
    message chemically meaningful if enumeration order ever shifts. All 37
    products have distinct canonical SMILES, so the key is unambiguous.
    """
    from yarp.reaction.enum import bnfn

    products = {}
    for prod in bnfn(yarpecules=khp_parent, n=2, hashes={khp_parent.hash},
                     hash_filter=True, lower_score=True, verbose=False):
        prod.get_smiles()
        assert prod.canon_smi not in products, (
            f"canonical SMILES {prod.canon_smi} is no longer unique among the "
            "enumerated products; these fixtures key on it"
        )
        products[prod.canon_smi] = prod
    return products

@pytest.fixture
def khp_reaction(khp_parent, khp_products):
    """
    A real reaction object: KHP -> one of its break-2/form-2 products.

    Function-scoped: `reaction()` deep-copies both states, so every test gets
    conformer dictionaries of its own to write into.
    """
    from yarp.reaction.reaction import reaction

    return reaction(khp_parent, khp_products["C=COCOO"])

@pytest.fixture(scope="session")
def khp_remapped_products(khp_parent):
    """
    Groups of products that are the same molecule under different atom mappings.

    `hash_filter=False` keeps the redundant mappings that `bnfn` would
    otherwise collapse, which is the only way to get two yarpecules that share
    a yarpecule hash but carry different adjacency matrices. Returns
    {canonical SMILES: [yarpecule, ...]} for the groups with more than one
    member.
    """
    from yarp.reaction.enum import bnfn

    groups = {}
    for prod in bnfn(yarpecules=khp_parent, n=2, hashes={khp_parent.hash},
                     hash_filter=False, lower_score=False, verbose=False):
        prod.get_smiles()
        groups.setdefault(prod.canon_smi, []).append(prod)

    remapped = {smi: mols for smi, mols in groups.items() if len(mols) > 1}
    assert remapped, "no duplicated mappings were enumerated"
    return remapped

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


@pytest.fixture
def cyclohexane_dehydrogenation():
    """Return the mapped ring-dehydrogenation reaction used for hash tests."""
    from yarp.yarpecule.input_parsers import load_reactions_from_smiles_file

    source = Path(__file__).parent / "molecules" / "cyclohexane_dehydrogenation.smi"
    reactions = load_reactions_from_smiles_file(source)
    assert len(reactions) == 1
    return next(iter(reactions.values()))

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


# yarpecule initialization: regression cases for the SMILES parser, ring finder and
# geometry fallback. Each fixture holds the edge cases of one test; every case targets a
# fixed regression unless it is marked as a guard.
CYCLIC_CUMULENE = "[C:1]([C:2]1=[N:5][C:3]([H:8])=[C:6]=[C:7]=[C:4]1[H:9])([H:10])([H:11])[H:12]"


@pytest.fixture
def smiles2adjmat_cases():
    """
    (smiles, formula, {(map_i, map_j): bond order}) for smiles2adjmat. Branch bookkeeping
    (a `)` followed by `(` left close_flag set), multi-letter element tokens, two-letter
    bracket aromatics, and H-led element symbols.
    """
    return [
        # N4=H5 on master (OctetError)
        ("[C:1]([H:2])([H:3])(=[N:4][H:5])", "CH3N", {(1, 4): 2, (4, 5): 1}),
        # RGD1 MR_20770_1 reactant fragment: N16=H19 on master
        ("[C:12]([N:13]([H:17])[H:18])(/[C:14](=[N:16]/[H:19])[H:20])=[C:15]([H:21])[H:22]", "C3H6N2",
         {(14, 16): 2, (16, 19): 1, (12, 15): 2}),
        # RGD1 species read "clean" on master, with C5#C6 as a double bond
        ("[C:1](\\[C:2]([H:10])([H:11])[H:12])(=[C:3](\\[N:4]([C:5]#[C:6][H:13])[H:8])[H:9])[H:7]", "C5H7N",
         {(5, 6): 3}),
        # unmapped: the wrong order also changed the inferred H count (master: 8 H)
        ("CC(C)(=NC)", "C4H9N", {}),
        # Kaden: `Clc` / `Brc` read as one unknown element
        ("Clc1ccccc1", "C6ClH5", {}),
        ("Brc1ccc(Br)cc1", "Br2C6H4", {}),
        # `Sc` read as scandium, `In` as indium (silently)
        ("CSc1ccccc1", "C7H8S", {}),
        ("In1cccc1", "C4H4IN", {}),
        # `[se]` read as sulfur (silently)
        ("c1cc[se]c1", "C4H4Se", {}),
        # the H of He counted as a hydrogen (OctetError)
        ("[He]", "He", {}),
    ]


@pytest.fixture
def aromatic_double_bond_cases():
    """
    (smiles, needs a ring double bond?) for needs_aromatic_double_bond, one bool per
    aromatic atom in input order. Lone-pair donors (pyrrole-type n, ring o/s, c-) and
    atoms with an exocyclic double bond must say False.
    """
    return [
        ("c1ccccc1", [True] * 6),                                       # guard
        ("c1ccncc1", [True] * 6),                                       # guard: pyridine n
        ("c1cc[nH]c1", [True, True, True, False, True]),               # pyrrole
        ("c1ccoc1", [True, True, True, False, True]),                  # furan
        ("c1ccsc1", [True, True, True, False, True]),                  # thiophene
        ("C[n+]1ccccc1", [True] * 6),                                   # pyridinium: n+ with 3 neighbours
        ("O=c1cc[nH]cc1", [False, True, True, False, True, True]),     # 4-pyridone: exocyclic =O and [nH]
        ("[n:1]1([H:8])[c:2](=[O:7])[n:6][c:3]([N:4]([H:9])[H:10])[o:5]1",   # RGD1 species
         [False, False, True, True, False]),
        ("[cH-]1cccc1", [False, True, True, True, True]),              # cyclopentadienide
    ]


@pytest.fixture
def ring_cases():
    """
    (smiles, ring sizes) for return_rings(remove_fused=False). The first atom of the fused
    ones is a fusion atom whose lowest ring neighbour is across the shared bond (master
    raised), or master returned a path that does not close.
    """
    return [
        ("c12c(occ1)cccc2", [5, 6, 9]),                                 # benzofuran (master raised)
        ("c12c([nH]cc1)cccc2", [5, 6, 9]),                              # indole (master raised)
        ("[n:1]12[n:2]([n:3]1[H:6])[c:4]2=[O:5]", [3, 3, 4]),           # RGD1 species (master raised)
        ("c1cccc2sc(cc12)O", [5, 6, 9]),                                # master: non-closing 9-atom path
        ("c1ccc2ccccc2c1", [6, 6, 10]),                                 # guard: naphthalene
        ("C1CC2CCC1C2", [5, 5, 6]),                                     # guard: norbornane
        ("C1CCC2(C1)CC2", [3, 5]),                                      # guard: spiro
    ]


@pytest.fixture
def obabel_embed_cases():
    """
    (smiles, bond lengths checked?) for obabel_embed, built from RDKit's kekulized graph.
    The strained ones are RDKit embedding failures from RGD1; UFF does not always keep
    their bonds, so only the geometry itself is checked there.
    """
    return [
        ("CCO", True),
        ("c1ccccc1", True),
        ("[NH4+]", True),                       # charge
        ("[CH3]", True),                        # radical
        ("C[Sc](C)C", True),                    # metal
        ("O=C=C=C=O", True),                    # cumulene chain
        ("C1=NC=C=C=C1", False),                # cyclic cumulene (RDKit cannot embed)
        ("[C-]1=[N+]=C(C)CO1", False),          # nitrile-ylide ring (RDKit cannot embed)
    ]


@pytest.fixture
def obabel_worker_shadows():
    """
    Files in the current directory that would shadow the imports of obabel_embed's worker
    process if it ran without isolation (an old yarp checkout, a local json.py or numpy.py).
    """
    return ["yarp/__init__.py", "json.py", "numpy.py"]


@pytest.fixture
def geo_via_obabel_cases():
    """SMILES whose rdkit molecule (explicit H) rdkit cannot embed, plus two it can (guards)."""
    return [
        "C1=NC=C=C=C1",                         # RGD1-type cyclic cumulene
        "CC1=NC=C=C=C1",
        "[C-]1=[N+]=C(C)CO1",                   # nitrile-ylide ring
        "C1=C=C=NN1C",
        "c1ccccc1",                             # guard: aromatic bonds are kekulized for Open Babel
        "CCO",                                  # guard
    ]


@pytest.fixture
def xyz_from_smiles_cases():
    """
    (smiles, mode, expected) for xyz_from_smiles with its default strict=True: "geometry"
    (a finite geometry, never None) or the exception raised.
    """
    return [
        (CYCLIC_CUMULENE, "yarp", "geometry"),                          # master: ValueError Bad Conformer Id
        (CYCLIC_CUMULENE, "rdkit", "geometry"),                         # master: geometry None
        ("[C:1]1([H:6])([H:7])[C:2]2([H:8])[n:3]3[n:4]1[n:5]32", "yarp", "geometry"),  # master: TypeError
        ("[C:1]=[O:4].[C:2](#[N:3])[H:5]", "yarp", "geometry"),         # master: embedded as formaldehyde
        ("[C:1]([H:2])[H:3]", "yarp", "geometry"),                      # master: carbene embedded as methane
        ("c1ccccc1[N](C)(C)C", "yarp", "OctetError"),                   # master: tried the RDKit geometry fallback
        ("CCO", "yarp", "geometry"),                                    # guard
        ("CCO", "rdkit", "geometry"),                                   # guard
    ]


@pytest.fixture
def yarpecule_cases():
    """
    (smiles, formula) that the default yarpecule() must build in yarp mode, without any
    RDKit fallback. Master's defaults sent each through a fallback, or failed.
    """
    return [
        ("Clc1ccccc1", "C6ClH5"),                                       # master: RDKit mode
        ("c1ccoc1", "C4H4O"),                                           # master: RDKit geometry fallback
        ("c12c(occ1)cccc2", "C8H6O"),                                   # master: RDKit geometry fallback
        ("[C:12]([N:13]([H:17])[H:18])(/[C:14](=[N:16]/[H:19])[H:20])=[C:15]([H:21])[H:22]", "C3H6N2"),  # master: OctetError
        ("[C:1]1([H:6])([H:7])[C:2]2([H:8])[n:3]3[n:4]1[n:5]32", "C2H3N3"),  # master: TypeError
        (CYCLIC_CUMULENE, "C6H5N"),                                     # master: ValueError
        ("CCO", "C2H6O"),                                               # guard
    ]
