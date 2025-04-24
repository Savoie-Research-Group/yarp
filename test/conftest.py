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
