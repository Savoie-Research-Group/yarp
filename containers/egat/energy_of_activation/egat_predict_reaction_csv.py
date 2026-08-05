#!/usr/bin/env python3
"""
EGAT reaction property inference: activation barrier + reaction enthalpy from CSV.
Self-contained: uses only EGAT_container/src modules.
"""
import argparse
import sys
import os
import tempfile
from pathlib import Path

# Thread env must be set before numpy/torch import. ~8 is the CPU sweet spot for
# these small graphs (more threads add overhead). Override with --threads.
_DEFAULT_THREADS = min(8, os.cpu_count() or 1)
if "EGAT_THREADS" in os.environ:
    _DEFAULT_THREADS = int(os.environ["EGAT_THREADS"])
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, str(_DEFAULT_THREADS))

import numpy as np
import pandas as pd
import torch
import omegaconf

# Ensure src is on path (run from EGAT_container root)
_EGAT_ROOT = Path(__file__).resolve().parent
_SRC = _EGAT_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load_activation_model(activation_model_path: str):
    from egat.compile_model import load_compiled_model

    model, _cfg = load_compiled_model(activation_model_path)
    model.eval()
    return model


def _load_enthalpy_model(enthalpy_model_path: str, config):
    from egat.predict_enthalpy_from_smiles import load_model_from_checkpoint

    model, _ckpt = load_model_from_checkpoint(enthalpy_model_path, config)
    model.eval()
    return model


def _predict_with_fastdataset(config, csv_path: str, activation_model=None, enthalpy_model=None):
    """Returns (activation_preds, enthalpy_preds)."""
    from egat.dataset import FastDataset

    dataset = FastDataset(config, dataset=csv_path)
    activation_preds = [] if activation_model is not None else None
    enthalpy_preds = [] if enthalpy_model is not None else None

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample is None:
                if activation_preds is not None:
                    activation_preds.append(np.nan)
                if enthalpy_preds is not None:
                    enthalpy_preds.append(np.nan)
                continue
            _idx, rgraph, pgraph, _strings = sample
            if activation_preds is not None:
                activation_preds.append(float(activation_model(rgraph, pgraph).item()))
            if enthalpy_preds is not None:
                enthalpy_preds.append(float(enthalpy_model(rgraph, pgraph).item()))

    return activation_preds, enthalpy_preds


def main():
    p = argparse.ArgumentParser(
        description="EGAT: activation barrier + reaction enthalpy from reaction_smiles CSV"
    )
    p.add_argument("--input", required=True, help="Input CSV path")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument(
        "--reactions-col",
        default="reaction_smiles",
        help="Column name for reaction SMILES (default: reaction_smiles)",
    )
    p.add_argument(
        "--activation-model",
        default=str(_EGAT_ROOT / "models" / "Activation_barrier.pth"),
        help="Path to activation model (.pth)",
    )
    p.add_argument(
        "--enthalpy-model",
        default=str(_EGAT_ROOT / "models" / "Enthalpy.pth"),
        help="Path to enthalpy model (.pth)",
    )
    p.add_argument(
        "--activation-config",
        default=str(_EGAT_ROOT / "models" / "Activation_barrier.yaml"),
        help="YAML config for activation model",
    )
    p.add_argument(
        "--enthalpy-config",
        default=str(_EGAT_ROOT / "models" / "Enthalpy.yaml"),
        help="YAML config for enthalpy model",
    )
    p.add_argument("--no-activation", action="store_true", help="Skip activation barrier")
    p.add_argument("--no-enthalpy", action="store_true", help="Skip reaction enthalpy")
    p.add_argument("--self-test", action="store_true", help="Run smoke test")
    p.add_argument("--threads", type=int, default=_DEFAULT_THREADS,
                       help=f"Torch intra-op threads (default {_DEFAULT_THREADS}; ~8 is fastest).")
    args = p.parse_args()

    torch.set_num_threads(max(1, args.threads))

    if args.self_test:
        sample_csv = _EGAT_ROOT / "examples" / "sample_reactions.csv"
        if not sample_csv.exists():
            sample_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                "reaction_smiles": [
                    "[C:0]([O:1][H:2])>>[C:0](=[O:1])[H:2]",
                    "[C:0]([C:1]([C:2](=[O:5])[H:9])([H:10])[H:11])([O:3][O:4][H:6])([H:7])[H:8]>>[C:0]([C:2]1([H:9])[C:1]([H:10])([H:11])[O:5]1)([O:3][O:4][H:6])([H:7])[H:8]",
                ]
            }).to_csv(sample_csv, index=False)
        args.input = str(sample_csv)
        args.output = str(_EGAT_ROOT / "examples" / "sample_out.csv")

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_in = pd.read_csv(in_path)
    if args.reactions_col not in df_in.columns:
        for alt in ("reactions", "reaction", "AAM"):
            if alt in df_in.columns:
                args.reactions_col = alt
                break
        else:
            raise ValueError(
                f"Input CSV must have column '{args.reactions_col}' (or reactions/reaction/AAM). "
                f"Found: {list(df_in.columns)}"
            )

    reactions = df_in[args.reactions_col].astype(str).tolist()
    do_activation = not args.no_activation
    do_enthalpy = not args.no_enthalpy
    if not (do_activation or do_enthalpy):
        raise ValueError("Set at least one of activation or enthalpy.")

    tmpdir_obj = tempfile.TemporaryDirectory(prefix="egat_")
    tmpdir = Path(tmpdir_obj.name)
    work_csv = tmpdir / "input_aam.csv"
    pd.DataFrame({"AAM": reactions}).to_csv(work_csv, index=False)
    data_path = tmpdir / "data"
    data_path.mkdir(exist_ok=True)

    activation_preds = None
    enthalpy_preds = None

    if do_activation:
        cfg_a = omegaconf.OmegaConf.load(args.activation_config)
        omegaconf.OmegaConf.set_struct(cfg_a, False)
        cfg_a.smiles = "AAM"
        cfg_a.data_path = str(data_path)
        cfg_a.molecular = False
        activation_model = _load_activation_model(args.activation_model)
        activation_preds, _ = _predict_with_fastdataset(
            cfg_a, str(work_csv), activation_model=activation_model
        )

    if do_enthalpy:
        cfg_h = omegaconf.OmegaConf.load(args.enthalpy_config)
        omegaconf.OmegaConf.set_struct(cfg_h, False)
        cfg_h.smiles = "AAM"
        cfg_h.data_path = str(data_path)
        cfg_h.molecular = False
        enthalpy_model = _load_enthalpy_model(args.enthalpy_model, cfg_h)
        _, enthalpy_preds = _predict_with_fastdataset(
            cfg_h, str(work_csv), enthalpy_model=enthalpy_model
        )

    df_out = pd.DataFrame(
        {
            "reaction_smiles": reactions,
            "activation_barrier": activation_preds if activation_preds is not None else [np.nan] * len(reactions),
            "reaction_enthalpy": enthalpy_preds if enthalpy_preds is not None else [np.nan] * len(reactions),
        }
    )
    df_out.to_csv(out_path, index=False)
    tmpdir_obj.cleanup()

    if args.self_test:
        df_chk = pd.read_csv(out_path)
        ok_act = np.isfinite(df_chk["activation_barrier"].values).any()
        ok_ent = np.isfinite(df_chk["reaction_enthalpy"].values).any()
        if do_activation and not ok_act:
            raise RuntimeError("Self-test failed: activation predictions all NaN/inf")
        if do_enthalpy and not ok_ent:
            raise RuntimeError("Self-test failed: enthalpy predictions all NaN/inf")
        print("Self-test OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
