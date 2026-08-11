"""Pytest fixtures for EGAT_container tests."""
import sys
from pathlib import Path

import pytest

# Add EGAT_container/src to path
EGAT_ROOT = Path(__file__).resolve().parent.parent
SRC = EGAT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def egat_root():
    return EGAT_ROOT


@pytest.fixture
def reference_csv():
    return str(EGAT_ROOT / "tests" / "reference_predictions.csv")


@pytest.fixture
def test_input_csv():
    """Input CSV with reaction_smiles for testing."""
    return str(EGAT_ROOT / "tests" / "reference_predictions.csv")


@pytest.fixture
def activation_model_path(egat_root):
    return str(egat_root / "models" / "Activation_barrier.pth")


@pytest.fixture
def enthalpy_model_path(egat_root):
    return str(egat_root / "models" / "Enthalpy.pth")


@pytest.fixture
def activation_config_path(egat_root):
    return str(egat_root / "models" / "Activation_barrier.yaml")


@pytest.fixture
def enthalpy_config_path(egat_root):
    return str(egat_root / "models" / "Enthalpy.yaml")
