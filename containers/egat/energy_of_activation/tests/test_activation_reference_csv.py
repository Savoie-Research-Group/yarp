"""
Validate activation model predictions against tests/activation_reference.csv.

This matches the style of YARP's EGAT tests: compare predicted values to the
precomputed Activation_PRED column within a small tolerance.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import omegaconf


def test_activation_matches_activation_reference_csv():
    egat_root = Path(__file__).resolve().parent.parent
    activation_csv = egat_root / "tests" / "activation_reference.csv"
    model_path = egat_root / "models" / "Activation_barrier.pth"
    cfg_path = egat_root / "models" / "Activation_barrier.yaml"

    df = pd.read_csv(activation_csv)
    df = df.head(5).copy()

    # Convert (Rsmiles, Psmiles) -> reaction_smiles (R>>P)
    assert "Rsmiles" in df.columns and "Psmiles" in df.columns
    df["reaction_smiles"] = df["Rsmiles"].astype(str) + ">>" + df["Psmiles"].astype(str)

    # Prepare the minimal CSV format EGAT expects
    # (Keep a human-readable intermediate around if you want to inspect it)
    tmp_csv = egat_root / "tests" / "_tmp_activation_input.csv"
    df[["reaction_smiles"]].to_csv(tmp_csv, index=False)

    # Run prediction through EGAT_container entrypoint logic (directly, not subprocess)
    import sys

    src = egat_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from egat.compile_model import load_compiled_model
    from egat.dataset import FastDataset

    cfg = omegaconf.OmegaConf.load(str(cfg_path))
    omegaconf.OmegaConf.set_struct(cfg, False)

    # The dataset builder consumes an AAM column; we adapt reaction_smiles -> AAM.
    df_aam = pd.DataFrame({"AAM": df["reaction_smiles"].astype(str).tolist()})
    tmp_aam = egat_root / "tests" / "_tmp_activation_aam.csv"
    df_aam.to_csv(tmp_aam, index=False)

    # Use a local, writable data path for any caching
    data_path = egat_root / "tests" / "_tmp_data"
    data_path.mkdir(exist_ok=True)

    cfg.smiles = "AAM"
    cfg.data_path = str(data_path)
    cfg.molecular = False

    model, _ = load_compiled_model(str(model_path))
    model.eval()

    dataset = FastDataset(cfg, dataset=str(tmp_aam))
    preds = []
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample is not None
            _idx, rgraph, pgraph, _ = sample
            preds.append(float(model(rgraph, pgraph).item()))

    expected = df["Activation_PRED"].astype(float).values
    preds = np.asarray(preds, dtype=float)

    tol = 5e-4  # same order as YARP's test tolerance
    diff = np.abs(preds - expected)
    assert np.all(diff < tol), f"Max diff {diff.max():.6f} (tol={tol})"

    # Cleanup temp files
    try:
        tmp_csv.unlink(missing_ok=True)
        tmp_aam.unlink(missing_ok=True)
    except TypeError:
        # Python <3.8 compatibility for missing_ok (not expected here, but harmless)
        if tmp_csv.exists():
            tmp_csv.unlink()
        if tmp_aam.exists():
            tmp_aam.unlink()

