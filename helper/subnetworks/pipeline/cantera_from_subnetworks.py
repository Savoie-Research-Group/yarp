#!/usr/bin/env python3
"""Convert subnetwork pickles to Cantera input YAML with embedded mapping metadata."""

import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path

import yaml
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
    # Unified pipeline config support: use the cantera_from_subnetworks section if present.
    cfg = raw_cfg.get("cantera_from_subnetworks", raw_cfg)
    if not isinstance(cfg, dict):
        raise ValueError("cantera_from_subnetworks config section must be a mapping/object.")
    return cfg, config_path.parent.resolve()


def apply_runtime_module_paths(cfg, cfg_dir):
    runtime_cfg = cfg.get("runtime", {}) or {}
    module_paths = runtime_cfg.get("module_paths", [])
    if module_paths is None:
        return
    if isinstance(module_paths, str):
        module_paths = [module_paths]

    for path_text in module_paths:
        path = resolve_path(str(path_text), cfg_dir)
        if not path.exists():
            continue
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def is_reaction_like(obj):
    return hasattr(obj, "reactant") and hasattr(obj, "product") and (
        hasattr(obj, "barrier") or hasattr(obj, "forward_barrier")
    )


def iter_reactions(container):
    seen_ids = set()
    stack = [container]
    while stack:
        current = stack.pop()
        obj_id = id(current)
        if obj_id in seen_ids:
            continue
        seen_ids.add(obj_id)

        if is_reaction_like(current):
            yield current
            continue

        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, (list, tuple, set)):
            stack.extend(current)


def collect_subnetwork_pickles(input_cfg, cfg_dir):
    root = resolve_path(str(input_cfg.get("root", "subnetworks")), cfg_dir)
    glob_pattern = str(input_cfg.get("glob", "*.pkl"))
    recursive = bool(input_cfg.get("recursive", True))

    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Input subnetworks root not found: {root}")

    if recursive:
        pickles = sorted(p for p in root.rglob(glob_pattern) if p.is_file())
    else:
        pickles = sorted(p for p in root.glob(glob_pattern) if p.is_file())

    if not pickles:
        raise RuntimeError(f"No subnetwork pickle files found under {root} with pattern '{glob_pattern}'.")
    return root, pickles


def load_subnetwork_payload(pickle_path):
    try:
        with pickle_path.open("rb") as f:
            payload = pickle.load(f)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Failed to unpickle {pickle_path} because a required module is missing. "
            "Activate the Python environment with YARP installed and retry."
        ) from exc

    if isinstance(payload, dict) and "rxns" in payload:
        rxns = payload.get("rxns", {}) or {}
        paths = payload.get("paths", []) or []
        metadata = payload.get("metadata", {}) or {}
        if not isinstance(rxns, dict):
            raise ValueError(f"`rxns` payload in {pickle_path} must be a dictionary.")
        return rxns, paths, metadata

    reactions = list(iter_reactions(payload))
    if not reactions:
        raise RuntimeError(f"No reaction objects found in pickle payload: {pickle_path}")
    return {idx: rxn for idx, rxn in enumerate(reactions, start=1)}, [], {}


def collect_path_reaction_keys(paths):
    keys = set()
    for path in (paths or []):
        if not isinstance(path, list):
            continue
        for row in path:
            if not isinstance(row, dict):
                continue
            rxn_key = row.get("rxn_key")
            if rxn_key is not None:
                keys.add(str(rxn_key))
    return keys


def choose_reverse_seed_keys(rxns, paths, start_smiles=None, end_smiles=None):
    _ = start_smiles  # kept for signature compatibility
    end_parts = set(split_smiles(end_smiles or ""))

    reverse_keys = set()
    for key, rxn in rxns.items():
        reactant_smiles = extract_smiles_from_state(getattr(rxn, "reactant", None))
        product_smiles = extract_smiles_from_state(getattr(rxn, "product", None))
        reactant_parts = set(reactant_smiles)
        product_parts = set(product_smiles)

        # Keep final-product-forming and product-consuming reactions forward-only.
        if end_parts and (end_parts & product_parts):
            continue
        if end_parts and (end_parts & reactant_parts):
            continue
        reverse_keys.add(key)
    return reverse_keys


def ordered_reaction_items(rxns, reverse_seed_keys):
    reverse_seed_text = {str(k) for k in reverse_seed_keys}

    reverse_items = []
    forward_only_items = []
    for key, rxn in rxns.items():
        if key in reverse_seed_keys or str(key) in reverse_seed_text:
            reverse_items.append((key, rxn))
        else:
            forward_only_items.append((key, rxn))

    ordered = reverse_items + forward_only_items
    return ordered, len(reverse_items)


def split_smiles(smiles):
    return [part.strip() for part in str(smiles).split(".") if part.strip()]


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


def extract_smiles_from_state(state):
    if state is None:
        return []

    species = getattr(state, "species", None)
    if species:
        out = []
        for sp in species:
            smi = getattr(sp, "canon_smi", None) or getattr(sp, "_canon_smi", None)
            if isinstance(smi, str) and smi:
                out.extend([s for s in (normalize_smiles(part) for part in split_smiles(smi)) if s])
        if out:
            return out

    smi = getattr(state, "canon_smi", None) or getattr(state, "_canon_smi", None)
    if isinstance(smi, str) and smi:
        return [s for s in (normalize_smiles(part) for part in split_smiles(smi)) if s]
    return []


def species_composition_from_obj(species_obj):
    elements = getattr(species_obj, "elements", None)
    if not elements:
        return None
    counts = {}
    for element in elements:
        symbol = str(element or "").strip()
        if not symbol:
            continue
        # YARP stores lowercase aromatic symbols (e.g., "c"); Cantera expects standard element case.
        symbol = symbol[0].upper() + symbol[1:].lower()
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts or None


def extract_species_entries_from_state(state):
    if state is None:
        return []

    species = getattr(state, "species", None)
    if species:
        entries = []
        for sp in species:
            smi = getattr(sp, "canon_smi", None) or getattr(sp, "_canon_smi", None)
            if isinstance(smi, str) and smi:
                for part in split_smiles(smi):
                    normalized = normalize_smiles(part)
                    if normalized:
                        entries.append((normalized, species_composition_from_obj(sp)))
        if entries:
            return entries

    smi = getattr(state, "canon_smi", None) or getattr(state, "_canon_smi", None)
    if isinstance(smi, str) and smi:
        out = []
        for part in split_smiles(smi):
            normalized = normalize_smiles(part)
            if normalized:
                out.append((normalized, None))
        return out
    return []


def ch_counts_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    mol_h = Chem.AddHs(mol)
    counts = {}
    for atom in mol_h.GetAtoms():
        symbol = atom.GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def to_float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_barrier_value(obj, attr_name, label):
    raw = getattr(obj, attr_name, None)
    if raw is None:
        return None
    if isinstance(raw, dict):
        if label and label in raw:
            return to_float_or_none(raw[label])
        if len(raw) == 1:
            return to_float_or_none(next(iter(raw.values())))
        return None
    return to_float_or_none(raw)


def convert_energy_to_kcal(value, units):
    u = str(units or "kcal/mol").strip().lower()
    if u in {"kcal/mol", "kcal"}:
        return float(value)
    if u in {"kj/mol", "kj"}:
        return float(value) * 0.239005736
    return float(value)


def _canonical_species_signature(entries):
    return tuple(sorted(s for s, _ in entries if s))


def dedupe_reaction_items(
    reaction_items,
    reverse_seed_count,
    *,
    forward_label,
    reverse_label,
    energy_units,
):
    seen = {}
    order = []

    for idx, (orig_key, rxn) in enumerate(reaction_items, start=1):
        reverse_enabled = idx <= int(reverse_seed_count)
        reactant_entries = extract_species_entries_from_state(getattr(rxn, "reactant", None))
        product_entries = extract_species_entries_from_state(getattr(rxn, "product", None))
        signature = (
            _canonical_species_signature(reactant_entries),
            _canonical_species_signature(product_entries),
        )

        forward_val = extract_barrier_value(rxn, "forward_barrier", forward_label)
        if forward_val is None:
            forward_val = extract_barrier_value(rxn, "barrier", forward_label)
        forward_kcal = (
            convert_energy_to_kcal(forward_val, energy_units)
            if forward_val is not None
            else float("inf")
        )
        reverse_val = extract_barrier_value(rxn, "reverse_barrier", reverse_label or forward_label)
        has_reverse = reverse_val is not None

        candidate = {
            "orig_key": orig_key,
            "rxn": rxn,
            "forward_kcal": float(forward_kcal),
            "has_reverse": bool(has_reverse),
            "reverse_enabled": bool(reverse_enabled),
            "merged_orig_keys": [str(orig_key)],
            "merged_rxn_ids": [str(getattr(rxn, "id", ""))],
            "merged_rxn_hashes": [str(getattr(rxn, "hash", ""))],
            "signature": {
                "reactants": list(signature[0]),
                "products": list(signature[1]),
            },
        }

        if signature not in seen:
            seen[signature] = candidate
            order.append(signature)
            continue

        existing = seen[signature]
        existing["reverse_enabled"] = bool(existing["reverse_enabled"] or reverse_enabled)
        existing["merged_orig_keys"].append(str(orig_key))
        existing["merged_rxn_ids"].append(str(getattr(rxn, "id", "")))
        existing["merged_rxn_hashes"].append(str(getattr(rxn, "hash", "")))

        need_reverse = bool(existing["reverse_enabled"])
        existing_ok = (not need_reverse) or bool(existing["has_reverse"])
        candidate_ok = (not need_reverse) or bool(candidate["has_reverse"])
        should_replace = False
        if candidate_ok and not existing_ok:
            should_replace = True
        elif candidate_ok == existing_ok and float(candidate["forward_kcal"]) < float(existing["forward_kcal"]):
            should_replace = True

        if should_replace:
            existing["orig_key"] = candidate["orig_key"]
            existing["rxn"] = candidate["rxn"]
            existing["forward_kcal"] = candidate["forward_kcal"]
            existing["has_reverse"] = candidate["has_reverse"]

    deduped_entries = [seen[sig] for sig in order]
    reverse_entries = [entry for entry in deduped_entries if entry["reverse_enabled"]]
    forward_entries = [entry for entry in deduped_entries if not entry["reverse_enabled"]]
    ordered_entries = reverse_entries + forward_entries

    deduped_items = [(entry["orig_key"], entry["rxn"]) for entry in ordered_entries]
    dedupe_meta_by_orig_key = {}
    for entry in ordered_entries:
        dedupe_meta_by_orig_key[str(entry["orig_key"])] = {
            "merged_orig_keys": sorted({k for k in entry["merged_orig_keys"] if k}),
            "merged_rxn_ids": sorted({k for k in entry["merged_rxn_ids"] if k}),
            "merged_rxn_hashes": sorted({k for k in entry["merged_rxn_hashes"] if k}),
            "signature": entry["signature"],
        }
    dedupe_collisions = sum(
        1 for entry in ordered_entries if len(entry["merged_orig_keys"]) > 1
    )
    return deduped_items, len(reverse_entries), dedupe_meta_by_orig_key, dedupe_collisions


def embed_metadata_in_yaml(yaml_path, embedded):
    with yaml_path.open("r") as f:
        payload = yaml.safe_load(f) or {}
    payload["yaks_metadata"] = embedded
    with yaml_path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _metadata_smiles(metadata, key):
    value = (metadata or {}).get(f"{key}_smiles")
    if value:
        return normalize_smiles_text(value)
    state = (metadata or {}).get(key) or {}
    if isinstance(state, dict):
        return normalize_smiles_text(state.get("canon_smi"))
    return None


def build_cantera_structs(
    reaction_items,
    reverse_seed_count,
    *,
    forward_label,
    reverse_label,
    reverse_fallback_factor,
    energy_units,
    dedupe_meta_by_orig_key=None,
):
    if not reaction_items:
        raise RuntimeError("No reactions available for conversion.")

    species_order = []
    species_composition = {}

    def register_species(smiles, composition=None):
        if smiles not in species_composition:
            species_order.append(smiles)
            species_composition[smiles] = composition or {}
        elif composition and not species_composition[smiles]:
            species_composition[smiles] = composition
        return smiles

    reaction_dict = {}
    reaction_map = {}

    for idx, (orig_key, rxn) in enumerate(reaction_items, start=1):
        reactant_entries = extract_species_entries_from_state(getattr(rxn, "reactant", None))
        product_entries = extract_species_entries_from_state(getattr(rxn, "product", None))

        reactants = [register_species(s, comp) for s, comp in reactant_entries]
        products = [register_species(s, comp) for s, comp in product_entries]

        forward = extract_barrier_value(rxn, "forward_barrier", forward_label)
        if forward is None:
            forward = extract_barrier_value(rxn, "barrier", forward_label)
        if forward is None:
            rxn_id = getattr(rxn, "id", f"reaction_{idx}")
            raise RuntimeError(
                f"Missing forward barrier for reaction {rxn_id!r} and label {forward_label!r}."
            )
        forward = convert_energy_to_kcal(forward, energy_units)

        reverse = None
        if idx <= reverse_seed_count:
            reverse = extract_barrier_value(rxn, "reverse_barrier", reverse_label or forward_label)
            if reverse is None and reverse_fallback_factor is not None:
                reverse = forward * reverse_fallback_factor
            if reverse is None:
                rxn_id = getattr(rxn, "id", f"reaction_{idx}")
                raise RuntimeError(
                    f"Missing reverse barrier for reverse-enabled reaction {rxn_id!r}. "
                    "Either add reverse barriers in the pickle or set rate.reverse_fallback_factor."
                )
            reverse = convert_energy_to_kcal(reverse, energy_units)

        d_g = (forward - reverse) if reverse is not None else 0.0
        rxn_key = f"reaction_{idx}"
        reaction_dict[rxn_key] = {
            "reactants": reactants,
            "products": products,
            "barrier": float(forward),
            "dG": float(d_g),
        }
        reaction_map[rxn_key] = {
            "orig_key": str(orig_key),
            "rxn_id": str(getattr(rxn, "id", "")),
            "rxn_hash": str(getattr(rxn, "hash", "")),
            "reverse_enabled": bool(idx <= reverse_seed_count),
        }
        merged_meta = (dedupe_meta_by_orig_key or {}).get(str(orig_key)) or {}
        if merged_meta:
            reaction_map[rxn_key]["merged_orig_keys"] = merged_meta.get("merged_orig_keys", [])
            reaction_map[rxn_key]["merged_rxn_ids"] = merged_meta.get("merged_rxn_ids", [])
            reaction_map[rxn_key]["merged_rxn_hashes"] = merged_meta.get("merged_rxn_hashes", [])
            reaction_map[rxn_key]["dedupe_signature"] = merged_meta.get("signature", {})

    compound_dict = {}
    for smi in species_order:
        if smi in compound_dict:
            continue
        compound_dict[smi] = species_composition.get(smi) or ch_counts_from_smiles(smi)

    n_rxns = len(reaction_items)
    if reverse_seed_count <= 0:
        depth_dict = {"1": list(range(1, n_rxns + 1))}
        curr_depth = 1
    else:
        depth_dict = {
            "1": list(range(1, reverse_seed_count + 1)),
            "2": list(range(reverse_seed_count + 1, n_rxns + 1)),
        }
        curr_depth = 2

    expl_nodes = sorted(compound_dict.keys())
    return (
        compound_dict,
        reaction_dict,
        depth_dict,
        curr_depth,
        expl_nodes,
        species_order,
        reaction_map,
    )


def call_write_yaml_and_move_output(
    *,
    out_path,
    compound_dict,
    reaction_dict,
    depth_dict,
    curr_depth,
    expl_nodes,
    initial_species,
    write_cfg,
):
    from cantera_helpers import write_yaml

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    try:
        os.chdir(out_path.parent)
        write_yaml(
            compound_dict,
            reaction_dict,
            depth_dict,
            curr_depth,
            expl_nodes,
            kinetics=write_cfg.get("kinetics", "gas"),
            reactions=write_cfg.get("reactions", "all"),
            model=write_cfg.get("model", "constant-cp"),
            EOS=write_cfg.get("EOS", "ideal-gas"),
            T=write_cfg.get("T", 373),
            P=write_cfg.get("P", 1),
            initial_species=initial_species,
        )
    finally:
        os.chdir(cwd)

    generated = out_path.parent / "cantera_input.yaml"
    if not generated.exists():
        raise RuntimeError("cantera_helpers.write_yaml did not create cantera_input.yaml as expected.")
    if generated.resolve() != out_path.resolve():
        shutil.move(str(generated), str(out_path))


def process_subnetwork_pickle(
    subnetwork_path,
    input_root,
    cfg,
    cfg_dir,
):
    rxns, paths, metadata = load_subnetwork_payload(subnetwork_path)
    start_smiles = _metadata_smiles(metadata, "start")
    end_smiles = _metadata_smiles(metadata, "end")
    path_reaction_keys = collect_path_reaction_keys(paths)
    if path_reaction_keys:
        rxns_selected = {
            key: rxn
            for key, rxn in (rxns or {}).items()
            if str(key) in path_reaction_keys
        }
        if not rxns_selected:
            rxns_selected = dict(rxns or {})
    else:
        rxns_selected = dict(rxns or {})
    reverse_seed_keys = choose_reverse_seed_keys(
        rxns_selected,
        paths,
        start_smiles=start_smiles,
        end_smiles=end_smiles,
    )
    reaction_items_all = list(rxns_selected.items())

    reaction_items_dict = {k: rxn for k, rxn in reaction_items_all}
    reaction_items, reverse_seed_count = ordered_reaction_items(reaction_items_dict, reverse_seed_keys)
    rate_cfg = cfg.get("rate", {}) or {}
    forward_label = str(rate_cfg.get("forward_barrier_label", "egat"))
    reverse_label = rate_cfg.get("reverse_barrier_label")
    energy_units = str(rate_cfg.get("energy_units", "kcal/mol"))
    reverse_fallback_factor = rate_cfg.get("reverse_fallback_factor")
    reverse_fallback_factor = (
        float(reverse_fallback_factor) if reverse_fallback_factor is not None else None
    )
    dedupe_after_normalization = bool(cfg.get("dedupe_after_normalization", True))
    n_rxns_before_dedupe = len(reaction_items)
    dedupe_meta_by_orig_key = {}
    dedupe_collisions = 0
    if dedupe_after_normalization:
        (
            reaction_items,
            reverse_seed_count,
            dedupe_meta_by_orig_key,
            dedupe_collisions,
        ) = dedupe_reaction_items(
            reaction_items,
            reverse_seed_count,
            forward_label=forward_label,
            reverse_label=reverse_label,
            energy_units=energy_units,
        )

    (
        compound_dict,
        reaction_dict,
        depth_dict,
        curr_depth,
        expl_nodes,
        species_order,
        reaction_map,
    ) = build_cantera_structs(
        reaction_items,
        reverse_seed_count,
        forward_label=forward_label,
        reverse_label=reverse_label,
        reverse_fallback_factor=reverse_fallback_factor,
        energy_units=energy_units,
        dedupe_meta_by_orig_key=dedupe_meta_by_orig_key,
    )
    start_species = start_smiles if start_smiles in compound_dict else next(iter(compound_dict.keys()))

    out_cfg = cfg.get("output", {}) or {}
    output_root = resolve_path(str(out_cfg.get("root", "subnetwork_cantera_yaml")), cfg_dir)
    preserve_tree = bool(out_cfg.get("preserve_tree", True))

    if preserve_tree:
        rel = subnetwork_path.relative_to(input_root)
        out_yaml = output_root / rel.with_suffix(".yaml")
    else:
        out_yaml = output_root / f"{subnetwork_path.stem}.yaml"

    call_write_yaml_and_move_output(
        out_path=out_yaml,
        compound_dict=compound_dict,
        reaction_dict=reaction_dict,
        depth_dict=depth_dict,
        curr_depth=curr_depth,
        expl_nodes=expl_nodes,
        initial_species=start_species,
        write_cfg=cfg.get("write_yaml", {}) or {},
    )

    embed_metadata_in_yaml(
        out_yaml,
        {
            "source_subnetwork_pickle": str(subnetwork_path),
            "start": (metadata or {}).get("start", {}),
            "end": (metadata or {}).get("end", {}),
            "start_smiles": _metadata_smiles(metadata, "start"),
            "end_smiles": _metadata_smiles(metadata, "end"),
            "start_species_name": start_species,
            "n_paths": len(paths),
            "n_rxns_selected_from_paths": int(len(rxns_selected)),
            "n_rxn_keys_on_paths": int(len(path_reaction_keys)),
            "energy_units_in": energy_units,
            "species_names": species_order,
            "reaction_map": reaction_map,
            "species_aliasing": "disabled",
            "dedupe_after_normalization": dedupe_after_normalization,
            "n_rxns_before_dedupe": int(n_rxns_before_dedupe),
            "n_rxns_after_dedupe": int(len(reaction_items)),
            "dedupe_collisions": int(dedupe_collisions),
        },
    )

    return {
        "subnetwork_pickle": subnetwork_path,
        "output_yaml": out_yaml,
        "n_rxns": len(reaction_items),
        "n_rxns_before_dedupe": int(n_rxns_before_dedupe),
        "n_dedupe_collisions": int(dedupe_collisions),
        "n_reverse_rxns": reverse_seed_count,
        "n_paths": len(paths),
        "n_rxns_selected_from_paths": int(len(rxns_selected)),
        "n_rxn_keys_on_paths": int(len(path_reaction_keys)),
        "start": (metadata or {}).get("start", {}),
        "end": (metadata or {}).get("end", {}),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert all path-aware subnetworks to Cantera YAML files using "
            "cantera_helpers.write_yaml."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to pipeline_config.yaml "
            "(default: ./pipeline/configs/pipeline_config.yaml)"
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path()
    cfg, cfg_dir = load_config(config_path)
    apply_runtime_module_paths(cfg, cfg_dir)

    input_cfg = cfg.get("input", {}) or {}
    input_root, pickles = collect_subnetwork_pickles(input_cfg, cfg_dir)
    print(f"Found {len(pickles)} subnetwork pickle(s) in {input_root}")

    summaries = []
    for pkl in pickles:
        summary = process_subnetwork_pickle(pkl, input_root, cfg, cfg_dir)
        summaries.append(summary)

    print("\nConversion summary:")
    max_print = int((cfg.get("logging", {}) or {}).get("max_print", 25))
    for s in summaries[:max_print]:
        end_smi = (s["end"] or {}).get("canon_smi", "")
        print(
            f"- {s['subnetwork_pickle'].name} -> {s['output_yaml'].name} | "
            f"rxns={s['n_rxns']} (from {s['n_rxns_before_dedupe']}) | "
            f"path_rxns={s['n_rxns_selected_from_paths']} | "
            f"dedupe_collisions={s['n_dedupe_collisions']} | "
            f"reverse_rxns={s['n_reverse_rxns']} | "
            f"paths={s['n_paths']} | end={end_smi}"
        )

    if len(summaries) > max_print:
        print(f"... and {len(summaries) - max_print} more files")


if __name__ == "__main__":
    main()
