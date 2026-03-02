#!/usr/bin/env python3
"""
Generate subnetworks from YARP network pickles using a YAML config.

Pipeline:
1. Load one pickle (or all pickles in a directory).
2. Optionally add dummy reverse barriers.
3. Build a yarp.network.network object.
4. Select the initial reagent yarpecule.
5. Enumerate simple paths from start reagent to each terminal species.
6. Save per-terminal subnetworks as pickle files.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

from rdkit import Chem
from yarp.network.network import network
from tqdm.auto import tqdm as tqdm_auto

import yaml

from add_dummy_reverse_barriers import add_dummy_reverse_barriers, load_pickle_payload


def default_config_path():
    cwd_cfg = Path("pipeline/configs/pipeline_config.yaml")
    if cwd_cfg.exists():
        return cwd_cfg
    return Path(__file__).resolve().parent / "configs" / "pipeline_config.yaml"


def resolve_path(path_text, config_dir):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (config_dir / path).resolve()


def load_config(config_path):
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Create pipeline/configs/pipeline_config.yaml or pass --config /path/to/pipeline_config.yaml."
        )

    with config_path.open("r") as f:
        raw_cfg = yaml.safe_load(f) or {}
    if not isinstance(raw_cfg, dict):
        raise ValueError("Config YAML must be a mapping/object at top level.")

    # Unified pipeline config support: use the subnetwork_gen section if present.
    cfg = raw_cfg.get("subnetwork_gen", raw_cfg)
    if not isinstance(cfg, dict):
        raise ValueError("subnetwork_gen config section must be a mapping/object.")

    return cfg, config_path.parent.resolve()


def collect_input_pickles(cfg, config_dir):
    input_cfg = cfg.get("input", {}) or {}
    input_pickle = input_cfg.get("pickle")
    input_dir = input_cfg.get("directory")
    glob_pattern = input_cfg.get("glob", "*.pkl")
    recursive = bool(input_cfg.get("recursive", False))

    if input_pickle and input_dir:
        raise ValueError("Set only one of input.pickle or input.directory.")
    if not input_pickle and not input_dir:
        raise ValueError("Config must set input.pickle or input.directory.")

    if input_pickle:
        path = resolve_path(str(input_pickle), config_dir)
        if not path.exists():
            raise FileNotFoundError(f"Input pickle not found: {path}")
        return [path]

    root = resolve_path(str(input_dir), config_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {root}")

    if recursive:
        pickles = sorted(root.rglob(glob_pattern))
    else:
        pickles = sorted(root.glob(glob_pattern))

    pickles = [p for p in pickles if p.is_file()]
    if not pickles:
        raise RuntimeError(f"No pickle files found in {root} with pattern '{glob_pattern}'.")
    return pickles


def _safe_label(text):
    return "".join(ch if (ch.isalnum() or ch in "-._") else "_" for ch in str(text))


def split_smiles(smiles):
    return [part.strip() for part in str(smiles or "").split(".") if part.strip()]


def normalize_smiles(smiles):
    smi = str(smiles or "").strip()
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def normalize_smiles_text(smiles):
    parts = []
    for part in split_smiles(smiles):
        normalized = normalize_smiles(part)
        if normalized:
            parts.append(normalized)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return ".".join(sorted(parts))


def _node_id(yp_obj, style):
    style = str(style).lower().strip()
    if style == "hash":
        return str(getattr(yp_obj, "hash"))

    if style in {"inchi", "inchikey", "inchi_key"}:
        # Do not recalculate; only use if present.
        inchi = getattr(yp_obj, "inchi", None)
        if inchi:
            return str(inchi)
        return str(getattr(yp_obj, "hash"))

    raise ValueError("Node ID style must be 'hash' or 'inchi'.")


def _unique_mols_by_hash(mols):
    unique = {}
    for mol in mols:
        unique[mol.hash] = mol
    return list(unique.values())


def _species_lookup_from_reactions(crn):
    lookup = {}
    for rxn in crn.rxns.values():
        for mol in list(rxn.reactant.species) + list(rxn.product.species):
            lookup[mol.hash] = mol
    return lookup


def _auto_detect_start_species(crn):
    species_lookup = _species_lookup_from_reactions(crn)
    initial_hashes = [
        n for n, attr in crn.crn.nodes(data=True)
        if attr.get("type") == "species"
        and crn.crn.in_degree(n) == 0
        and crn.crn.out_degree(n) > 0
    ]

    # Deterministic order across runs.
    initial_hashes = sorted(initial_hashes, key=lambda x: str(x))
    start_mols = [species_lookup[h] for h in initial_hashes if h in species_lookup]
    return _unique_mols_by_hash(start_mols)


def select_start_yarpecules(crn, start_cfg):
    reaction_id_query = start_cfg.get("reaction_id")
    smiles_query = start_cfg.get("smiles")
    auto_detect = bool(start_cfg.get("auto_detect", True))
    fallback_first_reactant = bool(start_cfg.get("fallback_first_reactant", True))

    matches = []
    for rxn in crn.rxns.values():
        rxn_id = str(getattr(rxn, "id", ""))
        rxn_hash = str(getattr(rxn, "hash", ""))
        reactant_smiles = [getattr(m, "canon_smi", "") for m in rxn.reactant.species]

        hit = False
        if reaction_id_query is not None:
            q = str(reaction_id_query).strip()
            if q and (q == rxn_id or q == rxn_hash):
                hit = True

        if smiles_query is not None:
            s = str(smiles_query).strip()
            if s and s in reactant_smiles:
                hit = True

        if hit:
            matches.extend(list(rxn.reactant.species))

    if not matches and auto_detect:
        matches = _auto_detect_start_species(crn)

    if not matches and fallback_first_reactant:
        for rxn in crn.rxns.values():
            if rxn.reactant.species:
                matches = list(rxn.reactant.species)
                break

    return _unique_mols_by_hash(matches)


def build_paths(crn, start_yp, terminal_yp, path_cfg):
    cutoff = path_cfg.get("cutoff")
    verbose = bool(path_cfg.get("verbose", False))
    show_progress = bool(path_cfg.get("show_progress", True))

    terminal_iter = terminal_yp
    if show_progress:
        terminal_iter = tqdm_auto(terminal_yp, desc="Finding paths to terminal species")

    paths = {}
    for end_yp in terminal_iter:
        paths[end_yp] = crn.get_simple_paths(start_yp, end_yp, cutoff=cutoff, verbose=verbose)
    return paths


def filter_terminal_species(terminal_yp, subnet_cfg):
    only_end_hash = subnet_cfg.get("only_end_hash")
    only_end_smiles = subnet_cfg.get("only_end_smiles")
    only_end_smiles_norm = normalize_smiles_text(only_end_smiles)

    if only_end_hash is None and only_end_smiles_norm is None:
        return list(terminal_yp)

    selected = []
    for yp in terminal_yp:
        yp_hash = str(getattr(yp, "hash", ""))
        yp_smiles_norm = normalize_smiles_text(getattr(yp, "canon_smi", None))
        if only_end_hash is not None and yp_hash != str(only_end_hash):
            continue
        if only_end_smiles_norm is not None and yp_smiles_norm != only_end_smiles_norm:
            continue
        selected.append(yp)
    return selected


def _yarpecule_metadata(yp_obj):
    return {
        "hash": str(getattr(yp_obj, "hash", "")),
        "inchi": getattr(yp_obj, "inchi", None),
        "canon_smi": getattr(yp_obj, "canon_smi", None),
    }


def _yarpecule_smiles(yp_obj):
    smi = getattr(yp_obj, "canon_smi", None)
    if smi is None:
        return None
    return str(smi)


def _path_metadata(path, rxn_key_by_hash):
    steps = []
    for step_i in sorted(path.keys()):
        rxn = path[step_i]
        if rxn is None:
            continue
        rxn_hash = rxn.hash
        steps.append(
            {
                "step": int(step_i),
                "rxn_hash": rxn_hash,
                "rxn_key": rxn_key_by_hash.get(rxn_hash, rxn_hash),
                "rxn_id": getattr(rxn, "id", None),
            }
        )
    return steps


def save_subnetworks(source_pickle,rxns_payload,crn,start_yp,paths,subnet_cfg,log_cfg):
    min_paths_to_save = int(subnet_cfg.get("min_paths_to_save", 1))
    output_root = Path(subnet_cfg.get("output_root", "subnetworks"))
    node_id_style = subnet_cfg.get("node_id_style", "inchi")
    prefix_with_source_name = bool(subnet_cfg.get("prefix_with_source_name", True))
    max_saved_to_print = int(log_cfg.get("max_saved_to_print", 25))
    only_end_hash = subnet_cfg.get("only_end_hash")
    only_end_smiles = subnet_cfg.get("only_end_smiles")

    start_dir_label = _safe_label(_node_id(start_yp, node_id_style))
    start_file_label = _safe_label(_node_id(start_yp, node_id_style))

    outdir = output_root / start_dir_label
    outdir.mkdir(parents=True, exist_ok=True)

    # Preserve original pickle dictionary key style when available.
    rxn_key_by_hash = {}
    rxn_obj_by_hash = {}
    if isinstance(rxns_payload, dict):
        for k, rxn in rxns_payload.items():
            h = getattr(rxn, "hash", None)
            if h is not None:
                rxn_key_by_hash[h] = k
                rxn_obj_by_hash[h] = rxn
    else:
        for rxn in rxns_payload:
            h = getattr(rxn, "hash", None)
            if h is not None:
                rxn_key_by_hash[h] = h
                rxn_obj_by_hash[h] = rxn

    saved_files = []
    skipped_groups = 0

    for end_yp, path_list in paths.items():
        if only_end_hash and str(getattr(end_yp, "hash", "")) != str(only_end_hash):
            continue
        if only_end_smiles and str(getattr(end_yp, "canon_smi", "")) != str(only_end_smiles):
            continue
        if len(path_list) < min_paths_to_save:
            skipped_groups += 1
            continue

        sub_rxns = {}
        unique_path_records = {}
        path_multiplicity = {}
        for path in path_list:
            path_steps = _path_metadata(path, rxn_key_by_hash)
            if path_steps:
                signature = tuple(
                    step.get("rxn_key")
                    for step in sorted(path_steps, key=lambda s: int(s.get("step", 0)))
                )
                if signature and signature not in unique_path_records:
                    unique_path_records[signature] = path_steps
                    path_multiplicity[signature] = 0
                if signature:
                    path_multiplicity[signature] += 1

            for step_i in sorted(path.keys()):
                rxn = path[step_i]
                if rxn is None:
                    continue
                rxn_hash = rxn.hash
                key = rxn_key_by_hash.get(rxn_hash, rxn_hash)
                obj = rxn_obj_by_hash.get(rxn_hash, rxn)
                sub_rxns[key] = obj

        if not sub_rxns:
            skipped_groups += 1
            continue
        path_records = list(unique_path_records.values())

        end_label = _safe_label(_node_id(end_yp, node_id_style))
        if prefix_with_source_name:
            fname = f"{source_pickle.stem}__{start_file_label}_to_{end_label}.pkl"
        else:
            fname = f"{start_file_label}_to_{end_label}.pkl"

        out_file = outdir / fname
        payload = {
            "rxns": sub_rxns,
            "paths": path_records,
            "metadata": {
                "source_pickle": str(source_pickle),
                "start": _yarpecule_metadata(start_yp),
                "end": _yarpecule_metadata(end_yp),
                "start_smiles": _yarpecule_smiles(start_yp),
                "initial_reagent_smiles": _yarpecule_smiles(start_yp),
                "end_smiles": _yarpecule_smiles(end_yp),
                "n_paths_raw": len(path_list),
                "n_paths_unique": len(path_records),
                "n_paths": len(path_records),
                "path_signature_multiplicity": {
                    " -> ".join(str(x) for x in sig): int(count)
                    for sig, count in path_multiplicity.items()
                },
                "min_paths_to_save": min_paths_to_save,
                "node_id_style": node_id_style,
            },
        }
        with out_file.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        saved_files.append(
            (out_file, len(path_records), len(sub_rxns), getattr(end_yp, "canon_smi", "<missing>"))
        )

    print(f"  Saved subnetworks: {len(saved_files)}")
    print(f"  Skipped end-node groups: {skipped_groups}")
    print(f"  Output directory: {outdir.resolve()}")
    print(f"  Sample saved subnetworks (up to {max_saved_to_print}):")
    for out_file, n_paths, n_rxns, end_smi in saved_files[:max_saved_to_print]:
        print(f"  - {out_file.name} | end_smi= {end_smi} | n_paths= {n_paths} | n_rxns= {n_rxns}")

    return saved_files


def maybe_add_dummy_barriers(payload, source_pickle, cfg, config_dir):
    barrier_cfg = cfg.get("dummy_reverse_barriers", {}) or {}
    enabled = bool(barrier_cfg.get("enabled", False))
    if not enabled:
        return payload, None

    seed = barrier_cfg.get("seed")
    save_reverse = bool(barrier_cfg.get("save_reverse", True))
    suffix = barrier_cfg.get("suffix", "_full_network")
    generated_dir = resolve_path(
        str(barrier_cfg.get("output_dir", "generated_networks")),
        config_dir,
    )

    output_path = None
    if save_reverse:
        generated_dir.mkdir(parents=True, exist_ok=True)
        output_path = generated_dir / f"{source_pickle.stem}{suffix}{source_pickle.suffix}"

    updated_payload, written_path, _, _ = add_dummy_reverse_barriers(
        payload,
        output_path=output_path,
        seed=seed,
        verbose=bool(barrier_cfg.get("verbose", False)),
    )
    return updated_payload, written_path


def process_one_pickle(source_pickle, cfg, config_dir):
    print(f"\nProcessing: {source_pickle}")
    payload = load_pickle_payload(source_pickle)

    payload, generated_path = maybe_add_dummy_barriers(payload, source_pickle, cfg, config_dir)
    if generated_path is not None:
        print(f"  Generated network pickle: {generated_path}")

    network_cfg = cfg.get("network", {}) or {}
    dG_lot = network_cfg.get("dG_lot", "egat")
    terminal_verbose = bool(network_cfg.get("terminal_verbose", False))



    crn = network(rxns=payload, dG_lot=dG_lot)
    terminal_yp = crn.get_terminal_species(verbose=terminal_verbose)
    print(f"  species: {crn.n_species}")
    print(f"  reactions: {crn.n_rxns}")
    print(f"  terminal species (total): {len(terminal_yp)}")

    start_cfg = cfg.get("start", {}) or {}
    start_candidates = select_start_yarpecules(crn, start_cfg)
    if not start_candidates:
        raise RuntimeError("No starting reagent matches were found.")

    start_index = int(start_cfg.get("index", 0))
    if start_index < 0 or start_index >= len(start_candidates):
        raise IndexError(
            f"start.index={start_index} is out of range for {len(start_candidates)} start candidates."
        )
    start_yp = start_candidates[start_index]
    print(f"  selected start: {start_yp.canon_smi} | hash={start_yp.hash}")

    subnet_cfg = dict(cfg.get("subnetworks", {}) or {})
    target_terminal_yp = filter_terminal_species(terminal_yp, subnet_cfg)
    print(f"  terminal species (path targets): {len(target_terminal_yp)}")
    if not target_terminal_yp:
        print("  no terminal species matched only_end filters; skipping path build.")
        return {
            "source_pickle": source_pickle,
            "generated_pickle": generated_path,
            "n_species": crn.n_species,
            "n_reactions": crn.n_rxns,
            "n_terminal": len(terminal_yp),
            "n_saved_subnetworks": 0,
        }

    path_cfg = cfg.get("paths", {}) or {}
    paths = build_paths(crn, start_yp, target_terminal_yp, path_cfg)
    print(f"  built path groups: {len(paths)}")

    subnet_cfg["output_root"] = str(
        resolve_path(str(subnet_cfg.get("output_root", "subnetworks")), config_dir)
    )
    log_cfg = cfg.get("logging", {}) or {}
    saved_files = save_subnetworks(
        source_pickle=source_pickle,
        rxns_payload=payload,
        crn=crn,
        start_yp=start_yp,
        paths=paths,
        subnet_cfg=subnet_cfg,
        log_cfg=log_cfg,
    )

    return {
        "source_pickle": source_pickle,
        "generated_pickle": generated_path,
        "n_species": crn.n_species,
        "n_reactions": crn.n_rxns,
        "n_terminal": len(terminal_yp),
        "n_saved_subnetworks": len(saved_files),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate path-based sub-networks from YARP pickles.")
    parser.add_argument("--config", default=None,
        help="Path to pipeline_config.yaml (default: ./pipeline/configs/pipeline_config.yaml)")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path()
    cfg, config_dir = load_config(config_path)

    pickles = collect_input_pickles(cfg, config_dir)
    print(f"Found {len(pickles)} input pickle(s).")

    summaries = []
    for pkl in pickles:
        summaries.append(process_one_pickle(pkl, cfg, config_dir))

    print("\nRun summary:")
    for s in summaries:
        print(
            f"- {s['source_pickle'].name}: "
            f"species={s['n_species']}, rxns={s['n_reactions']}, "
            f"terminal={s['n_terminal']}, saved_subnetworks={s['n_saved_subnetworks']}"
        )


if __name__ == "__main__":
    main()
