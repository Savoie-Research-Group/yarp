"""
Persistent test fixtures for YARP testing suite
"""

import pytest
from pathlib import Path

#######################################
#       Molecular Input Parsing       #
#######################################


@pytest.fixture
def ethene_xyz():
    """Returns a string object of the absolute path to ethene XYZ file."""
    return str(Path(__file__).parent / "molecules" / "ethene.xyz")


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
