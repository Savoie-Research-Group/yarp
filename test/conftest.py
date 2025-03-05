import pytest
from pathlib import Path

# All test molecules should be loaded in here as fixtures.
# This way, all tests can access them.


@pytest.fixture
def ch2ch2_xyz():
    """Returns a string object of the absolute path to ethene XYZ file."""
    return str(Path(__file__).parent / "molecules" / "ch2ch2.xyz")


@pytest.fixture
def ch2ch2_smi():
    """Returns the smiles string for ethene."""
    return "C=C"
