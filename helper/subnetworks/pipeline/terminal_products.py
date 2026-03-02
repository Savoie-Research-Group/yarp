#!/usr/bin/env python3
"""Write terminal product list for one YARP network pickle."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from rdkit import Chem
from yarp.network.network import network

from add_dummy_reverse_barriers import load_pickle_payload
from subnetwork_gen import select_start_yarpecules


def default_config_path():
    cwd_cfg = Path("pipeline/configs/pipeline_config.yaml")
    if cwd_cfg.exists():
        return cwd_cfg
    return Path(__file__).resolve().parent / "configs" / "pipeline_config.yaml"


def load_config(config_path):
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg if isinstance(cfg, dict) else {}


def normalize_smiles(smiles):
    smi = str(smiles or "").strip()
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def _smiles(yp):
    smi = getattr(yp, "canon_smi", None)
    return normalize_smiles(smi) if smi is not None else None


def write_table(df, out_path):
    out_path = Path(out_path)
    try:
        df.to_parquet(out_path, index=False)
        return out_path
    except Exception as exc:
        fallback = out_path.with_suffix(".pkl")
        df.to_pickle(fallback)
        print(
            f"[warning] parquet write failed ({exc.__class__.__name__}); "
            f"wrote pickle instead: {fallback}"
        )
        return fallback


def main():
    parser = argparse.ArgumentParser(description="List terminal products for one network pickle.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--network-pickle", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path()
    cfg = load_config(config_path)
    preview_rows = max(1, int(cfg.get("table_preview_rows", 10)))

    sg = (cfg.get("subnetwork_gen", {}) or {})
    network_cfg = (sg.get("network", {}) or {})
    start_cfg = (sg.get("start", {}) or {})

    payload = load_pickle_payload(Path(args.network_pickle))
    crn = network(
        rxns=payload,
        dG_lot=network_cfg.get("dG_lot", "egat"),
    )
    terminal_yp = crn.get_terminal_species(verbose=bool(network_cfg.get("terminal_verbose", False)))
    start_candidates = select_start_yarpecules(crn, start_cfg)
    if not start_candidates:
        raise RuntimeError("No start species could be selected.")

    start_idx = int(start_cfg.get("index", 0))
    if start_idx < 0 or start_idx >= len(start_candidates):
        raise IndexError(f"start.index={start_idx} is out of range for {len(start_candidates)} candidates.")
    start_yp = start_candidates[start_idx]

    rows = []
    for yp in terminal_yp:
        rows.append(
            {
                "product_hash": str(getattr(yp, "hash", "")),
                "product_smiles": _smiles(yp),
                "start_hash": str(getattr(start_yp, "hash", "")),
                "start_smiles": _smiles(start_yp),
                "network_pickle": str(Path(args.network_pickle).resolve()),
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    written = write_table(df, out)
    print(f"Wrote terminal products: {written} (rows={len(rows)})")
    print(df.head(preview_rows).to_string(index=False))


if __name__ == "__main__":
    main()
