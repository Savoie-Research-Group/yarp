"""
Unit tests for EGAT activation barrier and reaction enthalpy models.
Compares predictions against reference values to ensure model correctness.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import omegaconf

# Ensure src is on path (conftest adds it, but import order may vary)
_EGAT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _EGAT_ROOT / "src"
if str(_SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_SRC))


def _run_predictions(
    input_csv,
    output_csv,
    activation_model,
    activation_config,
    enthalpy_model,
    enthalpy_config,
):
    """Run prediction pipeline and return output DataFrame."""
    from egat.compile_model import load_compiled_model
    from egat.predict_enthalpy_from_smiles import load_model_from_checkpoint
    from egat.dataset import FastDataset

    df_in = pd.read_csv(input_csv)
    col = "reaction_smiles" if "reaction_smiles" in df_in.columns else "reactions"
    reactions = df_in[col].astype(str).tolist()

    tmpdir = Path(tempfile.mkdtemp(prefix="egat_test_"))
    work_csv = tmpdir / "work.csv"
    data_path = tmpdir / "data"
    data_path.mkdir(exist_ok=True)
    pd.DataFrame({"AAM": reactions}).to_csv(work_csv, index=False)

    # Activation
    cfg_a = omegaconf.OmegaConf.load(activation_config)
    omegaconf.OmegaConf.set_struct(cfg_a, False)
    cfg_a.smiles = "AAM"
    cfg_a.data_path = str(data_path)
    cfg_a.molecular = False

    act_model, _ = load_compiled_model(activation_model)
    act_model.eval()

    # Enthalpy
    cfg_h = omegaconf.OmegaConf.load(enthalpy_config)
    omegaconf.OmegaConf.set_struct(cfg_h, False)
    cfg_h.smiles = "AAM"
    cfg_h.data_path = str(data_path)
    cfg_h.molecular = False

    enth_model, _ = load_model_from_checkpoint(enthalpy_model, cfg_h)
    enth_model.eval()

    dataset = FastDataset(cfg_a, dataset=str(work_csv))
    act_preds, enth_preds = [], []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample is None:
                act_preds.append(np.nan)
                enth_preds.append(np.nan)
                continue
            _idx, rgraph, pgraph, _ = sample
            act_preds.append(float(act_model(rgraph, pgraph).item()))
            enth_preds.append(float(enth_model(rgraph, pgraph).item()))

    df_out = pd.DataFrame({
        "reaction_smiles": reactions,
        "activation_barrier": act_preds,
        "reaction_enthalpy": enth_preds,
    })
    df_out.to_csv(output_csv, index=False)
    return df_out


class TestEGATModels:
    """Test that predictions match reference values within tolerance."""

    TOLERANCE = 0.01  # kcal/mol

    def test_activation_barrier_matches_reference(
        self,
        reference_csv,
        activation_model_path,
        activation_config_path,
        enthalpy_model_path,
        enthalpy_config_path,
        tmp_path,
    ):
        """Predictions should match reference_predictions.csv within tolerance."""
        ref = pd.read_csv(reference_csv)
        out_csv = tmp_path / "test_out.csv"

        pred = _run_predictions(
            reference_csv,
            str(out_csv),
            activation_model_path,
            activation_config_path,
            enthalpy_model_path,
            enthalpy_config_path,
        )

        pred_act = pred["activation_barrier"].values
        ref_act = ref["activation_barrier"].values

        assert len(pred_act) == len(ref_act), "Row count mismatch"
        valid = np.isfinite(pred_act) & np.isfinite(ref_act)
        assert np.all(valid), f"Some predictions are NaN: {np.where(~valid)[0]}"

        diff = np.abs(pred_act - ref_act)
        assert np.all(diff < self.TOLERANCE), (
            f"Activation barrier predictions differ from reference (tolerance={self.TOLERANCE}). "
            f"Max diff: {diff.max():.6f}, mean: {diff.mean():.6f}"
        )

    def test_reaction_enthalpy_matches_reference(
        self,
        reference_csv,
        activation_model_path,
        activation_config_path,
        enthalpy_model_path,
        enthalpy_config_path,
        tmp_path,
    ):
        """Enthalpy predictions should match reference within tolerance."""
        ref = pd.read_csv(reference_csv)
        out_csv = tmp_path / "test_out.csv"

        pred = _run_predictions(
            reference_csv,
            str(out_csv),
            activation_model_path,
            activation_config_path,
            enthalpy_model_path,
            enthalpy_config_path,
        )

        pred_enth = pred["reaction_enthalpy"].values
        ref_enth = ref["reaction_enthalpy"].values

        assert len(pred_enth) == len(ref_enth), "Row count mismatch"
        valid = np.isfinite(pred_enth) & np.isfinite(ref_enth)
        assert np.all(valid), f"Some predictions are NaN: {np.where(~valid)[0]}"

        diff = np.abs(pred_enth - ref_enth)
        assert np.all(diff < self.TOLERANCE), (
            f"Reaction enthalpy predictions differ from reference (tolerance={self.TOLERANCE}). "
            f"Max diff: {diff.max():.6f}, mean: {diff.mean():.6f}"
        )
