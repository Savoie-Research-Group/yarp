"""
End-to-end python script for full network generation from a SMILES text file.

Usage:
    python network_gen.py --config /path/to/config.yaml
"""

import argparse
import importlib.util
import logging
import pickle
import re
import subprocess
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import yaml
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def init_logging(verbose=False, name=None):
    level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(filename)s:%(funcName)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


def load_yaml(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must parse to a YAML mapping (dict).")
    return data


def resolve_path(path_value, base_dir):
    p = Path(path_value).expanduser()
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def required(config, keys):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError("Missing required config key: " + ".".join(keys))
        current = current[key]
    return current


def load_main_yarp_function(main_yarp_path):
    main_yarp_path = Path(main_yarp_path).expanduser().resolve()
    if not main_yarp_path.exists():
        raise FileNotFoundError(f"main_yarp.py not found at: {main_yarp_path}")

    spec = importlib.util.spec_from_file_location("main_yarp_runtime", str(main_yarp_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {main_yarp_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    main_fn = getattr(module, "main", None)
    if main_fn is None:
        raise AttributeError(f"No callable 'main' found in: {main_yarp_path}")
    return main_fn


def txt_to_smi(txt_path):
    log.debug(f"Reading SMILES from text file: {txt_path}")
    txt_path = Path(txt_path)
    if not txt_path.exists():
        log.error(f"Input text file not found: {txt_path}")
        sys.exit(1)

    with open(txt_path, "r") as f:
        smi = f.read().strip()

    if not smi:
        log.error(f"No SMILES string found in: {txt_path}")
        sys.exit(1)

    log.debug(f"Loaded SMILES: {smi}")
    return smi


def normalize_input_smiles(smiles, strip_stereo=True):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("RDKit MolFromSmiles returned None")
        canon = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=not strip_stereo,
        )
        if canon != smiles:
            log.debug(f"Canonicalized SMILES: {smiles} -> {canon}")
        return canon
    except Exception as exc:
        log.warning(f"RDKit canonicalization failed for {smiles}; using original. ({exc})")
        return smiles


def network_prefix(smi, max_length=24, run_tag=None):
    log.debug("Generating network prefix.")
    base = re.sub(r"[^A-Za-z0-9]+", "_", smi).strip("_")
    base = re.sub(r"_+", "_", base)
    base = base[:max_length] if base else "smiles"
    if run_tag is None:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.debug(f"Network prefix generated: {base}__{run_tag}")
    return f"{base}__{run_tag}"


def gen_chained_yarp_again_yamls(
    prefix,
    ers,
    initial_smiles,
    depth,
    out_dir,
    *,
    enum,
    mode,
    separate,
    l_score,
    f_charge,
    strain,
    barrier_cutoff,
    barrier_source,
    stage,
    egat_method,
    egat_model,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bn, fn = 2, 2
    if isinstance(ers, str) and ers.startswith("b") and "f" in ers:
        try:
            bn = int(ers.split("f")[0][1:])
            fn = int(ers.split("f")[1])
        except Exception:
            pass

    written = []
    prev_output_pkl = None

    for d in range(1, int(depth) + 1):
        stem = f"{prefix}__{ers}_d{d}"
        yml_path = out_dir / f"{stem}.yml"

        initial_species = initial_smiles if d == 1 else str(prev_output_pkl)
        output_pkl = (out_dir / f"{stem}.pkl").resolve()

        doc = {
            "initialize": {
                "enumerate": bool(enum),
                "mode": str(mode),
                "initial species": str(initial_species),
                "output": str(output_pkl),
                "bonds to break": int(bn),
                "bonds to form": int(fn),
                "separate products": separate,
                "enumeration filters": {
                    "lewis score": int(l_score),
                    "formal charge": int(f_charge),
                    "discard strained rings": bool(strain),
                    "barrier cutoff": float(barrier_cutoff),
                    "barrier source": str(barrier_source),
                },
            },
            "stages": [str(stage)],
            str(stage): {
                "method": str(egat_method),
                "model": str(egat_model),
            },
        }

        with open(yml_path, "w") as f:
            yaml.safe_dump(doc, f, sort_keys=False)

        written.append(yml_path)
        prev_output_pkl = output_pkl

    return written


def handle_main_yarp(network, yaml_path, log_dir, yp_main_fn):
    log.debug(f"Starting main_yarp for YAML: {yaml_path}")
    yaml_path = Path(yaml_path)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(yaml_path, "r") as f:
        inp = yaml.safe_load(f)

    pkl_field = inp.get("initialize", {}).get("output")
    if not pkl_field:
        log.error(f"No 'initialize: output:' field found in {yaml_path}")
        return None

    expected_pkl = Path(pkl_field).expanduser()
    if not expected_pkl.is_absolute():
        expected_pkl = (yaml_path.parent / expected_pkl).resolve()

    round_stem = yaml_path.stem
    out_file = log_dir / f"{round_stem}.out"
    err_file = log_dir / f"{round_stem}.err"

    with open(out_file, "w") as fout, open(err_file, "w") as ferr:
        try:
            with redirect_stdout(fout), redirect_stderr(ferr):
                yp_main_fn(inp)
        except Exception:
            traceback.print_exc(file=ferr)
            log.error(
                f"main_yarp failed for {network} round {round_stem}. "
                f"See {err_file} for details."
            )
            return None

    if not expected_pkl.exists():
        log.error(f"Expected pickle not found: {expected_pkl}")
        return None

    log.info(
        f"main_yarp completed for {network}\n"
        f"  Round  : {round_stem}\n"
        f"  Logs   : {log_dir}\n"
        f"  Stdout : {out_file}\n"
        f"  Stderr : {err_file}\n"
        f"  Pickle : {expected_pkl}"
    )
    return expected_pkl


def merge_pickles(pkls, out_name):
    merged = {}
    loaded = 0
    out_path = Path(out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.debug(f"Merging pickles: {pkls}")
    for p in pkls:
        with open(p, "rb") as f:
            rxns = pickle.load(f)
        log.debug(f"  loading {Path(p).name}: {len(rxns)} reactions")
        for rxn in rxns.values():
            loaded += 1
            merged[f"rxn_{loaded:07d}"] = rxn

    with open(out_path, "wb") as f:
        pickle.dump(merged, f)

    log.info(f"Merged {len(pkls)} pkls -> {out_path} (loaded={loaded})")
    return out_path


def ya_pickle_to_csv(
    network,
    pickle_path,
    out_dir,
    log_dir,
    export_script_path,
    canon,
    egat,
    flux,
):
    pickle_path = Path(pickle_path)
    out_dir = Path(out_dir)
    log_dir = Path(log_dir)
    export_script_path = Path(export_script_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{network}.csv"

    log.debug(f"Converting pickle to CSV for {network}: {pickle_path}")
    log.debug(
        f"  Output CSV : {csv_path}\n"
        f"  Canonical  : {canon}\n"
        f"  EGAT data  : {egat}\n"
        f"  Flux data  : {flux}"
    )

    cmd = ["python", str(export_script_path), str(pickle_path), str(csv_path)]
    if canon:
        cmd.append("-c")
    if egat:
        cmd.append("-e")
    if flux:
        cmd.append("-f")

    out_file = log_dir / f"{network}_export.out"
    err_file = log_dir / f"{network}_export.err"

    with open(out_file, "w") as fout, open(err_file, "w") as ferr:
        result = subprocess.run(cmd, stdout=fout, stderr=ferr, text=True)

    if result.returncode != 0:
        log.error(
            f"export_rxn_smi failed (exit {result.returncode})\n"
            f"  stdout: {out_file}\n"
            f"  stderr: {err_file}"
        )
        return None

    log.info(f"Reaction CSV written: {csv_path}")
    return csv_path


def validate_lists(ers, depths):
    if not isinstance(ers, list) or not isinstance(depths, list):
        raise ValueError("network.ers and network.depth must both be lists.")
    if len(ers) != len(depths):
        raise ValueError("network.ers and network.depth must be the same length.")
    if len(ers) == 0:
        raise ValueError("network.ers and network.depth cannot be empty.")


def main(config, config_dir):
    input_file = resolve_path(required(config, ["input", "smiles_txt"]), config_dir)
    work_dir = resolve_path(config.get("output", {}).get("work_dir", "."), config_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    ers = required(config, ["network", "ers"])
    depths = required(config, ["network", "depth"])
    validate_lists(ers, depths)

    yarp_cfg = config.get("yarp", {})
    export_cfg = config.get("export", {})
    output_cfg = config.get("output", {})

    main_yarp_path = resolve_path(required(yarp_cfg, ["main_yarp_path"]), config_dir)
    yp_main_fn = load_main_yarp_function(main_yarp_path)

    run_tag = output_cfg.get("run_tag") or datetime.now().strftime("%Y%m%d_%H%M%S")
    max_length = int(output_cfg.get("network_max_length", 24))

    smi_raw = txt_to_smi(input_file)
    smi = normalize_input_smiles(smi_raw, strip_stereo=bool(yarp_cfg.get("strip_stereo", True)))
    network = network_prefix(smi_raw, max_length=max_length, run_tag=run_tag)

    log.info("Starting YARP-Again Network Gen")
    log.debug(
        f"Loaded config:\n"
        f"  Config path   : {config_dir}\n"
        f"  Input file    : {input_file}\n"
        f"  Work dir      : {work_dir}\n"
        f"  CSV export    : {bool(export_cfg.get('enable_csv', False))}\n"
        f"  main_yarp.py  : {main_yarp_path}\n"
        f"  ERS list      : {ers}\n"
        f"  Depth list    : {depths}"
    )

    yamls = []
    for depth, ers_name in zip(depths, ers):
        generated = gen_chained_yarp_again_yamls(
            prefix=network,
            ers=ers_name,
            initial_smiles=smi,
            depth=depth,
            out_dir=work_dir,
            enum=bool(yarp_cfg.get("enumerate", True)),
            mode=yarp_cfg.get("mode", "concerted"),
            separate=yarp_cfg.get("separate_products"),
            l_score=int(yarp_cfg.get("lewis_score", 0)),
            f_charge=int(yarp_cfg.get("formal_charge", 3)),
            strain=bool(yarp_cfg.get("discard_strained_rings", False)),
            barrier_cutoff=float(yarp_cfg.get("barrier_cutoff", 120)),
            barrier_source=yarp_cfg.get("barrier_source", "egat"),
            stage=yarp_cfg.get("stage", "egat"),
            egat_method=yarp_cfg.get("egat_method", "ml_predict"),
            egat_model=yarp_cfg.get("egat_model", "egat_pretrain"),
        )
        log.info(f"Generated {len(generated)} YAMLs for ERS {ers_name} to depth {depth}.")
        yamls.extend(generated)

    pickles = []
    for yml_path in yamls:
        pkl = handle_main_yarp(
            network=network,
            yaml_path=yml_path,
            log_dir=work_dir,
            yp_main_fn=yp_main_fn,
        )
        if pkl is None:
            log.error(f"main_yarp failed for YAML: {yml_path}. Exiting.")
            sys.exit(1)
        pickles.append(pkl)

    if bool(export_cfg.get("enable_csv", False)):
        export_script = resolve_path(required(export_cfg, ["export_script_path"]), config_dir)
        merged_pkl = merge_pickles(
            pickles,
            out_name=work_dir / f"{network}__merged.pkl",
        )
        csv_out = ya_pickle_to_csv(
            network=network,
            pickle_path=merged_pkl,
            out_dir=work_dir,
            log_dir=work_dir,
            export_script_path=export_script,
            canon=bool(export_cfg.get("canon", True)),
            egat=bool(export_cfg.get("egat", True)),
            flux=bool(export_cfg.get("flux", False)),
        )
        if csv_out is None:
            log.error(f"CSV export failed for network: {network}. Exiting.")
            sys.exit(1)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate YARP-Again networks from a YAML config.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: ./config.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        config = load_yaml(config_path)
    except Exception as exc:
        print(f"Failed to load config: {exc}", file=sys.stderr)
        sys.exit(1)

    verbose = bool(config.get("logging", {}).get("verbose", False))
    log = init_logging(verbose=verbose, name=__name__)

    try:
        main(config=config, config_dir=config_path.parent)
    except Exception as exc:
        log.error(f"Fatal error: {exc}")
        log.debug(traceback.format_exc())
        sys.exit(1)
