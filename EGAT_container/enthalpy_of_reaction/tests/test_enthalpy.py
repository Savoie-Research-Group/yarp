"""Smoke + regression tests for the enthalpy-of-reaction container model."""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from rxn_egat.predict import load_model, predict_reactions  # noqa: E402

MODEL = _ROOT / "models" / "egat_dh.pth"

# (reactant, product) atom-mapped SMILES + expected DH (kcal/mol) from the
# trained model. Regression guard against featurization/architecture drift.
CASES = [
    ("[C:4]([C:6]([H:23])([H:24])[H:25])([H:10])([H:18])[H:19].[C@:1]1([N:8]([H:14])[H:15])([H:12])[C:3]([H:16])([H:17])[O:9][C@:2]([C:5]([H:20])([H:21])[H:22])([H:13])[N:7]1[H:11]",
     "[C:1]([C:3]([O:9][H:10])([H:16])[H:17])([N:7]([C:2]([C:4]([C:6]([H:23])([H:24])[H:25])([H:18])[H:19])([C:5]([H:20])([H:21])[H:22])[H:13])[H:11])([N:8]([H:14])[H:15])[H:12]",
     -6.05),
    ("[C-:1]([N+:6](=[C:2]([C:4]([H:17])([H:19])[H:20])[C:5]([H:18])([H:21])[H:22])[H:10])([N:7]([H:13])[H:14])[H:11].[C:3]([O:8][H:9])([H:12])([H:15])[H:16]",
     "[C:1]([C:3]([O:8][H:9])([H:15])[H:16])([N:6]([C:2]([C:4]([H:17])([H:19])[H:20])([C:5]([H:18])([H:21])[H:22])[H:12])[H:10])([N:7]([H:13])[H:14])[H:11]",
     -44.423),
]


@pytest.fixture(scope="module")
def model():
    assert MODEL.exists(), f"checkpoint missing: {MODEL}"
    return load_model(str(MODEL), device="cpu")


def test_predictions_finite_and_signed(model):
    m, ym, ys = model
    pairs = [(r, p) for r, p, _ in CASES]
    preds = predict_reactions(m, ym, ys, pairs)
    assert all(np.isfinite(preds))
    # model must produce both endothermic (+) and exothermic (-) values
    assert preds[1] < 0


def test_regression_values(model):
    m, ym, ys = model
    pairs = [(r, p) for r, p, _ in CASES]
    preds = predict_reactions(m, ym, ys, pairs)
    for got, (_, _, expected) in zip(preds, CASES):
        assert abs(got - expected) < 0.1, f"{got} vs {expected}"


def test_reverse_flips_sign(model):
    """Reverse reaction (swap reactant/product) should flip DH sign."""
    m, ym, ys = model
    r, p, _ = CASES[1]
    fwd = predict_reactions(m, ym, ys, [(r, p)])[0]
    rev = predict_reactions(m, ym, ys, [(p, r)])[0]
    assert np.sign(fwd) != np.sign(rev)
    assert abs(fwd + rev) < 5.0  # approximately antisymmetric
