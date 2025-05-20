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
