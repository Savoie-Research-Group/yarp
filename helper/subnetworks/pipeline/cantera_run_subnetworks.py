#!/usr/bin/env python3
"""Run Cantera over subnetwork YAMLs and write one run YAML plus pickle metadata updates."""

import argparse
import csv
import math
import pickle
import sys
from collections import Counter
from pathlib import Path

import yaml
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smiles_utils import normalize_smiles_text, split_smiles


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
    with config_path.open("r") as f:
        raw_cfg = yaml.safe_load(f) or {}
    cfg = raw_cfg.get("cantera_run_subnetworks", raw_cfg) if isinstance(raw_cfg, dict) else {}
    if not isinstance(cfg, dict):
        raise ValueError("cantera_run_subnetworks config section must be a mapping/object.")
    return cfg, config_path.parent.resolve()


def apply_runtime_module_paths(cfg, cfg_dir):
    runtime_cfg = cfg.get("runtime", {}) or {}
    module_paths = runtime_cfg.get("module_paths", [])
    if isinstance(module_paths, str):
        module_paths = [module_paths]
    for path_text in module_paths:
        path = resolve_path(str(path_text), cfg_dir)
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def collect_cantera_yamls(cfg, cfg_dir):
    input_cfg = cfg.get("input", {}) or {}
    root = resolve_path(str(input_cfg.get("cantera_yaml_root", "./subnetwork_cantera_yaml")), cfg_dir)
    glob_pattern = str(input_cfg.get("glob", "*.yaml"))
    recursive = bool(input_cfg.get("recursive", True))

    if recursive:
        files = sorted(p for p in root.rglob(glob_pattern) if p.is_file())
    else:
        files = sorted(p for p in root.glob(glob_pattern) if p.is_file())

    keep = []
    for path in files:
        name = path.name
        if name.endswith(".species_map.yaml"):
            continue
        if name.endswith(".reaction_map.yaml"):
            continue
        if name.endswith(".run.yaml"):
            continue
        if name.endswith(".reactor_debug.yaml"):
            continue
        keep.append(path)

    batch_cfg = cfg.get("batch", {}) or {}
    start = int(batch_cfg.get("start", 0))
    max_files = batch_cfg.get("max_files")
    if max_files is None:
        selected = keep[start:]
    else:
        selected = keep[start:start + int(max_files)]
    return root, keep, selected


def load_yaml(path):
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(path, payload):
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def write_reaction_flux_table(path, rows):
    fieldnames = [
        "reaction_index",
        "reaction_name",
        "equation",
        "orig_key",
        "rxn_id",
        "rxn_hash",
        "flux_to_final_species",
        "flux_to_final_species_std",
        "cumulative_abs_flux",
        "cumulative_abs_flux_std",
        "final_rate_of_progress",
        "final_rate_of_progress_std",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_final_product_flux_table(path, rows):
    fieldnames = [
        "reaction_index",
        "reaction_name",
        "equation",
        "orig_key",
        "rxn_id",
        "rxn_hash",
        "source_type",
        "from_smiles",
        "to_smiles",
        "flux_to_final_species",
        "flux_to_final_species_std",
        "cumulative_abs_flux",
        "cumulative_abs_flux_std",
        "final_rate_of_progress",
        "final_rate_of_progress_std",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_flux_timeseries_table(path, rows):
    fieldnames = [
        "time_s",
        "reaction_index",
        "reaction_name",
        "equation",
        "orig_key",
        "rxn_id",
        "rxn_hash",
        "cumulative_abs_flux",
        "cumulative_abs_flux_std",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_reactor_debug_yaml(path, payload):
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def print_table_preview(label, rows, max_rows=5):
    max_rows = max(1, int(max_rows))
    print(f"{label}: rows={len(rows)}")
    if not rows:
        return
    for row in rows[:max_rows]:
        preview = ", ".join(f"{k}={v}" for k, v in row.items())
        print(f"  {preview}")


def parse_reaction_index(reaction_name):
    return int(str(reaction_name).split("_")[-1]) - 1


def get_smiles_for_state(state):
    smi = getattr(state, "canon_smi", None)
    if smi:
        return normalize_smiles_text(smi) or str(smi)
    parts = []
    species = getattr(state, "species", None) or []
    for sp in species:
        sub = getattr(sp, "canon_smi", None)
        if sub:
            normalized = normalize_smiles_text(sub) or str(sub)
            parts.append(normalized)
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ".".join(sorted(parts))


def split_equation_side(side):
    return [tok.strip() for tok in str(side or "").split(" + ") if tok.strip()]


def parse_equation_token(token):
    text = str(token or "").strip()
    if not text:
        return None, 0.0
    parts = text.split(maxsplit=1)
    if len(parts) == 2:
        try:
            coeff = float(parts[0])
            species = parts[1].strip()
            if species:
                return species, coeff
        except Exception:
            pass
    return text, 1.0


def equation_counters(equation):
    text = str(equation or "")
    if "=>" not in text:
        return Counter(), Counter()
    left, right = text.split("=>", 1)
    left_counter = Counter()
    right_counter = Counter()
    for tok in split_equation_side(left):
        species, coeff = parse_equation_token(tok)
        if species:
            left_counter[species] += float(coeff)
    for tok in split_equation_side(right):
        species, coeff = parse_equation_token(tok)
        if species:
            right_counter[species] += float(coeff)
    return left_counter, right_counter


def equation_states(equation):
    """Parse normalized full reactant/product state SMILES from equation text."""
    text = str(equation or "")
    if "=>" not in text:
        return None, None
    left, right = text.split("=>", 1)
    left_parts = []
    right_parts = []
    for tok in split_equation_side(left):
        species, coeff = parse_equation_token(tok)
        if not species:
            continue
        smi = normalize_smiles_text(species)
        if not smi:
            continue
        n_rep = 1
        try:
            c = float(coeff)
            c_int = int(round(c))
            if c_int >= 1 and abs(c - c_int) < 1.0e-8:
                n_rep = c_int
        except Exception:
            n_rep = 1
        left_parts.extend([smi] * n_rep)
    for tok in split_equation_side(right):
        species, coeff = parse_equation_token(tok)
        if not species:
            continue
        smi = normalize_smiles_text(species)
        if not smi:
            continue
        n_rep = 1
        try:
            c = float(coeff)
            c_int = int(round(c))
            if c_int >= 1 and abs(c - c_int) < 1.0e-8:
                n_rep = c_int
        except Exception:
            n_rep = 1
        right_parts.extend([smi] * n_rep)
    left_parts = sorted([s for s in left_parts if s])
    right_parts = sorted([s for s in right_parts if s])
    left_state = ".".join(left_parts) if left_parts else None
    right_state = ".".join(right_parts) if right_parts else None
    return left_state, right_state


def get_meta_smiles(primary_meta, fallback_meta, key):
    value = (primary_meta or {}).get(f"{key}_smiles")
    if value is None:
        value = ((primary_meta or {}).get(key) or {}).get("canon_smi")
    if value is None:
        value = (fallback_meta or {}).get(f"{key}_smiles")
    if value is None:
        value = ((fallback_meta or {}).get(key) or {}).get("canon_smi")
    return normalize_smiles_text(value)


def get_reaction_by_key(rxns, rxn_key):
    if rxn_key in rxns:
        return rxns[rxn_key]
    key_text = str(rxn_key)
    for key, rxn in rxns.items():
        if str(key) == key_text:
            return rxn
    return None


def collect_terminal_product_states_from_paths(paths, rxns):
    terminal_states = set()
    for path_rows in (paths or []):
        steps = sorted(
            [row for row in path_rows if isinstance(row, dict)],
            key=lambda row: int(row.get("step", 0)),
        )
        if not steps:
            continue
        last = steps[-1]
        rxn = get_reaction_by_key(rxns or {}, last.get("rxn_key"))
        terminal_state = None
        if rxn is not None:
            terminal_state = normalize_smiles_text(get_smiles_for_state(getattr(rxn, "product", None)))
        if not terminal_state:
            terminal_state = normalize_smiles_text(last.get("product_smiles"))
        if terminal_state:
            terminal_states.add(terminal_state)
    return sorted(terminal_states)


def collect_nonterminal_path_states(paths, rxns):
    nonterminal_states = set()
    for path_rows in (paths or []):
        steps = sorted(
            [row for row in path_rows if isinstance(row, dict)],
            key=lambda row: int(row.get("step", 0)),
        )
        if not steps:
            continue
        for idx, step in enumerate(steps):
            rxn = get_reaction_by_key(rxns or {}, step.get("rxn_key"))
            reactant_state = None
            product_state = None
            if rxn is not None:
                reactant_state = normalize_smiles_text(get_smiles_for_state(getattr(rxn, "reactant", None)))
                product_state = normalize_smiles_text(get_smiles_for_state(getattr(rxn, "product", None)))
            if not reactant_state:
                reactant_state = normalize_smiles_text(step.get("reactant_smiles"))
            if not product_state:
                product_state = normalize_smiles_text(step.get("product_smiles"))
            if reactant_state:
                nonterminal_states.add(reactant_state)
            if idx < (len(steps) - 1) and product_state:
                nonterminal_states.add(product_state)
    return sorted(nonterminal_states)


def collect_terminal_completion_species(terminal_states, end_smiles=None, nonterminal_states=None):
    states = list(terminal_states or [])
    if end_smiles:
        matching = [s for s in states if end_smiles in split_smiles(normalize_smiles_text(s))]
        if matching:
            states = matching
    nonterminal_species = set()
    for state in (nonterminal_states or []):
        for part in split_smiles(normalize_smiles_text(state)):
            if part:
                nonterminal_species.add(part)
    ordered = []
    seen = set()
    for state in states:
        for part in split_smiles(normalize_smiles_text(state)):
            if part != end_smiles and part in nonterminal_species:
                continue
            if not part or part in seen:
                continue
            seen.add(part)
            ordered.append(part)
    return ordered


def terminal_product_fraction_scale(end_smiles, terminal_product_states, mode="max"):
    """Estimate target-species mole-fraction scale from terminal-state stoichiometry.

    Repeated tokens in a terminal state (e.g. ``P.[H][H].[H][H]``) are counted
    with multiplicity, so this correctly yields ``1/3`` for ``P`` in that case.
    """
    if not end_smiles:
        return 1.0
    fractions = []
    for state in (terminal_product_states or []):
        parts = split_smiles(normalize_smiles_text(state))
        if not parts:
            continue
        total = len(parts)
        if total <= 0:
            continue
        end_count = sum(1 for part in parts if part == end_smiles)
        if end_count > 0:
            fractions.append(float(end_count) / float(total))
    if not fractions:
        return 1.0
    mode_text = str(mode or "max").strip().lower()
    if mode_text in {"min", "lowest", "conservative"}:
        return min(fractions)
    if mode_text in {"mean", "avg", "average"}:
        return float(sum(fractions) / len(fractions))
    if mode_text in {"first"}:
        return fractions[0]
    # default: highest reachable target fraction among observed terminal states.
    return max(fractions)


def resolve_terminal_completion_species(
    terminal_species_cfg,
    terminal_completion_mode,
    end_smiles,
    terminal_product_states,
    nonterminal_states=None,
):
    mode = str(terminal_completion_mode or "auto").strip().lower()
    cfg_value = terminal_species_cfg
    if isinstance(cfg_value, str):
        text = cfg_value.strip().lower()
        if text in {"", "none", "null"}:
            return []
        if text not in {"auto", "final", "final_species"}:
            return [normalize_smiles_text(cfg_value) or str(cfg_value)]
    elif isinstance(cfg_value, (list, tuple, set)):
        species = []
        for item in cfg_value:
            smi = normalize_smiles_text(item) or str(item).strip()
            if smi:
                species.append(smi)
        return species
    elif cfg_value is not None:
        smi = normalize_smiles_text(cfg_value) or str(cfg_value).strip()
        return [smi] if smi else []

    if mode in {"product", "species", "target", "final_species"}:
        return [end_smiles] if end_smiles else []

    state_species = collect_terminal_completion_species(
        terminal_product_states,
        end_smiles=end_smiles,
        nonterminal_states=nonterminal_states,
    )
    if mode in {"terminal_state", "terminal_state_sum", "state", "full_state"}:
        return state_species if state_species else ([end_smiles] if end_smiles else [])

    # auto: steady-state accumulation in target product P.
    if end_smiles:
        return [end_smiles]
    return state_species if state_species else []


def build_path_flux_records(paths, rxns, flux_to_final_by_key, start_smiles=None, end_smiles=None):
    collapsed = {}
    for path_rows in paths:
        steps = [row for row in path_rows if isinstance(row, dict)]
        steps = sorted(steps, key=lambda row: int(row.get("step", 0)))
        step_flux_rows = []

        for row_i, row in enumerate(steps):
            rxn_key = row.get("rxn_key")
            key_text = str(rxn_key)
            rxn = get_reaction_by_key(rxns, key_text)
            if rxn is None:
                continue
            reactant = get_smiles_for_state(rxn.reactant)
            product = get_smiles_for_state(rxn.product)
            step_flux_rows.append(
                {
                    "step": int(row.get("step", row_i + 1)),
                    "rxn_key": key_text,
                    "reactant_smiles": reactant,
                    "product_smiles": product,
                    "flux_to_final_species": float(flux_to_final_by_key.get(key_text, 0.0)),
                }
            )

        terminal_flux = 0.0
        terminal_rxn_key = None
        if step_flux_rows:
            terminal_flux = float(step_flux_rows[-1]["flux_to_final_species"])
            terminal_rxn_key = step_flux_rows[-1]["rxn_key"]

        if not step_flux_rows:
            continue

        signature = tuple(step["rxn_key"] for step in step_flux_rows)
        chain_start = start_smiles or step_flux_rows[0]["reactant_smiles"]
        chain_nodes = [chain_start] + [step["product_smiles"] for step in step_flux_rows]
        if end_smiles:
            chain_nodes[-1] = end_smiles
        smiles_path = " -> ".join(chain_nodes)

        if signature not in collapsed:
            collapsed[signature] = {
                "path_index": 0,
                "smiles_path": smiles_path,
                "terminal_rxn_key": terminal_rxn_key,
                "terminal_flux_to_final_species": terminal_flux,
                "steps": step_flux_rows,
                "multiplicity": 1,
            }
        else:
            collapsed[signature]["multiplicity"] += 1

    path_records = []
    for i, record in enumerate(collapsed.values(), start=1):
        record["path_index"] = i
        path_records.append(record)
    return path_records


def build_reaction_part_maps(rxns):
    maps = {}
    for key, rxn in (rxns or {}).items():
        reactant_counter = Counter(split_smiles(get_smiles_for_state(getattr(rxn, "reactant", None))))
        product_counter = Counter(split_smiles(get_smiles_for_state(getattr(rxn, "product", None))))
        maps[str(key)] = (reactant_counter, product_counter)
    return maps


def species_flux_timeseries_from_rows(flux_ts_rows, reaction_part_maps):
    by_time = {}
    for row in flux_ts_rows:
        try:
            time_s = float(row.get("time_s", 0.0))
            flux_val = float(row.get("cumulative_abs_flux", 0.0))
            flux_std = float(row.get("cumulative_abs_flux_std", 0.0))
        except Exception:
            continue
        key = str(row.get("orig_key", ""))
        equation = str(row.get("equation", ""))
        by_time.setdefault(time_s, [])
        by_time[time_s].append((key, equation, flux_val, flux_std))

    rows = []
    for time_s in sorted(by_time.keys()):
        flux_in = Counter()
        flux_out = Counter()
        flux_in_var = Counter()
        flux_out_var = Counter()
        for key, equation, flux_val, flux_std in by_time[time_s]:
            reactant_counter, product_counter = equation_counters(equation)
            if not reactant_counter and not product_counter:
                reactant_counter, product_counter = reaction_part_maps.get(key, (Counter(), Counter()))
            if not reactant_counter and not product_counter:
                continue
            for smi, coeff in reactant_counter.items():
                scale = float(coeff)
                flux_out[smi] += flux_val * scale
                flux_out_var[smi] += (flux_std * scale) ** 2
            for smi, coeff in product_counter.items():
                scale = float(coeff)
                flux_in[smi] += flux_val * scale
                flux_in_var[smi] += (flux_std * scale) ** 2
        for smi in sorted(set(flux_in.keys()) | set(flux_out.keys())):
            rows.append(
                {
                    "time_s": float(time_s),
                    "species_smiles": smi,
                    "cumulative_in_flux": float(flux_in.get(smi, 0.0)),
                    "cumulative_in_flux_std": float(math.sqrt(max(flux_in_var.get(smi, 0.0), 0.0))),
                    "cumulative_out_flux": float(flux_out.get(smi, 0.0)),
                    "cumulative_out_flux_std": float(math.sqrt(max(flux_out_var.get(smi, 0.0), 0.0))),
                }
            )
    return rows


def final_species_flux_lookup(species_flux_rows):
    if not species_flux_rows:
        return {}
    final_time = max(float(row.get("time_s", 0.0)) for row in species_flux_rows)
    lookup = {}
    for row in species_flux_rows:
        if float(row.get("time_s", 0.0)) != final_time:
            continue
        smi = str(row.get("species_smiles", ""))
        lookup[smi] = {
            "in": float(row.get("cumulative_in_flux", 0.0)),
            "in_std": float(row.get("cumulative_in_flux_std", 0.0)),
            "out": float(row.get("cumulative_out_flux", 0.0)),
            "out_std": float(row.get("cumulative_out_flux_std", 0.0)),
        }
    return lookup


def collect_terminal_product_states(path_flux_records):
    terminal_states = set()
    for rec in (path_flux_records or []):
        steps = sorted((rec.get("steps") or []), key=lambda row: int(row.get("step", 0)))
        if not steps:
            continue
        terminal_state = normalize_smiles_text(steps[-1].get("product_smiles"))
        if terminal_state:
            terminal_states.add(terminal_state)
    return sorted(terminal_states)


def collect_on_target_intermediate_states(path_flux_records, start_smiles, terminal_states):
    terminal_state_set = set(terminal_states or [])
    on_target = set()
    for rec in (path_flux_records or []):
        for step in (rec.get("steps") or []):
            reactant_state = normalize_smiles_text(step.get("reactant_smiles"))
            product_state = normalize_smiles_text(step.get("product_smiles"))
            if reactant_state and reactant_state != start_smiles and reactant_state not in terminal_state_set:
                on_target.add(reactant_state)
            if product_state and product_state != start_smiles and product_state not in terminal_state_set:
                on_target.add(product_state)
    return sorted(on_target)


def collect_direct_on_target_intermediate_states(path_flux_records, start_smiles, terminal_states):
    terminal_state_set = set(terminal_states or [])
    direct = set()
    for rec in (path_flux_records or []):
        steps = sorted((rec.get("steps") or []), key=lambda row: int(row.get("step", 0)))
        if not steps:
            continue
        first_product_state = normalize_smiles_text(steps[0].get("product_smiles"))
        if first_product_state and first_product_state != start_smiles and first_product_state not in terminal_state_set:
            direct.add(first_product_state)
    return sorted(direct)


def collect_species_from_states(states):
    species = set()
    for state in (states or []):
        for smi in split_smiles(normalize_smiles_text(state)):
            if smi:
                species.add(smi)
    return sorted(species)


def check_directional_presence_for_on_target(
    rxns,
    included_keys,
    reaction_map,
    start_smiles,
    terminal_states,
    on_target_intermediate_states,
):
    keys = {str(k) for k in (included_keys or set())}
    terminal_state_set = set(terminal_states or [])
    reverse_enabled_keys = {
        str((v or {}).get("orig_key"))
        for v in (reaction_map or {}).values()
        if bool((v or {}).get("reverse_enabled", False))
        and (v or {}).get("orig_key") is not None
    }
    direction_rows = []
    for inter_state in on_target_intermediate_states:
        has_r_to_i = False
        has_i_to_r = False
        has_i_to_p = False
        for key, rxn in (rxns or {}).items():
            key_text = str(key)
            if keys and key_text not in keys:
                continue
            reactant_state = normalize_smiles_text(get_smiles_for_state(getattr(rxn, "reactant", None)))
            product_state = normalize_smiles_text(get_smiles_for_state(getattr(rxn, "product", None)))
            if start_smiles and reactant_state == start_smiles and product_state == inter_state:
                has_r_to_i = True
                if key_text in reverse_enabled_keys:
                    has_i_to_r = True
            if start_smiles and reactant_state == inter_state and product_state == start_smiles:
                has_i_to_r = True
            if terminal_state_set and reactant_state == inter_state and product_state in terminal_state_set:
                has_i_to_p = True
        direction_rows.append(
            {
                "intermediate_state": inter_state,
                "has_r_to_i": bool(has_r_to_i),
                "has_i_to_r": bool(has_i_to_r),
                "has_i_to_p": bool(has_i_to_p),
            }
        )
    return direction_rows


def print_flux_direction_debug(
    *,
    final_species_flux,
    start_smiles,
    end_smiles,
    on_target_intermediates,
    direction_rows,
    threshold,
):
    print(
        "[flux-debug] thresholds:",
        f"cumulative_flux_threshold={threshold}",
        f"on_target_intermediates={len(on_target_intermediates)}",
        f"direct_on_target_intermediates={len(direction_rows)}",
    )
    checks = []
    reagent_flux = final_species_flux.get(start_smiles, {"in": 0.0, "out": 0.0}) if start_smiles else {"in": 0.0, "out": 0.0}
    product_flux = final_species_flux.get(end_smiles, {"in": 0.0, "out": 0.0}) if end_smiles else {"in": 0.0, "out": 0.0}

    checks.append(
        {
            "species": start_smiles,
            "role": "reagent",
            "in": float(reagent_flux.get("in", 0.0)),
            "out": float(reagent_flux.get("out", 0.0)),
            "ok": bool(
                float(reagent_flux.get("in", 0.0)) > threshold
                and float(reagent_flux.get("out", 0.0)) > threshold
            ),
            "expectation": "R should have both in/out flux",
        }
    )
    for inter in on_target_intermediates:
        vals = final_species_flux.get(inter, {"in": 0.0, "out": 0.0})
        checks.append(
            {
                "species": inter,
                "role": "on_target_intermediate",
                "in": float(vals.get("in", 0.0)),
                "out": float(vals.get("out", 0.0)),
                "ok": bool(float(vals.get("in", 0.0)) > threshold and float(vals.get("out", 0.0)) > threshold),
                "expectation": "On-target I should have both in/out flux",
            }
        )
    checks.append(
        {
            "species": end_smiles,
            "role": "product",
            "in": float(product_flux.get("in", 0.0)),
            "out": float(product_flux.get("out", 0.0)),
            "ok": bool(float(product_flux.get("in", 0.0)) > threshold and float(product_flux.get("out", 0.0)) <= threshold),
            "expectation": "P should have in flux and ~zero out flux",
        }
    )

    for row in checks:
        print(
            "[flux-debug]",
            f"role={row['role']}",
            f"species={row['species']}",
            f"in={row['in']:.6e}",
            f"out={row['out']:.6e}",
            f"ok={row['ok']}",
            f"expect={row['expectation']}",
        )

    for row in direction_rows:
        print(
            "[flux-debug]",
            f"intermediate_state={row['intermediate_state']}",
            f"R_to_I={row['has_r_to_i']}",
            f"I_to_R={row['has_i_to_r']}",
            f"I_to_P={row['has_i_to_p']}",
        )

    failures = [row for row in checks if not row["ok"]]
    if failures:
        print(f"[flux-debug] WARNING: flux-direction expectation failed for {len(failures)} species.")
    missing_pairs = [row for row in direction_rows if not (row["has_r_to_i"] and row["has_i_to_r"])]
    if missing_pairs:
        print(
            f"[flux-debug] WARNING: missing R<->I reaction directions for {len(missing_pairs)} on-target intermediates."
        )
    return checks, failures, missing_pairs


def run_one_yaml(yaml_path, yaml_root, cfg, cfg_dir):
    from cantera_helpers import build_and_run_reactor

    input_cfg = cfg.get("input", {}) or {}
    pickle_root = resolve_path(str(input_cfg.get("subnetwork_pickle_root", "./subnetworks")), cfg_dir)
    rel = yaml_path.relative_to(yaml_root)
    pickle_path = pickle_root / rel.with_suffix(".pkl")

    cantera_yaml_payload = load_yaml(yaml_path)
    yaks_metadata = cantera_yaml_payload.get("yaks_metadata", {}) or {}
    reaction_map = yaks_metadata.get("reaction_map", {}) or {}
    included_keys = {
        str((v or {}).get("orig_key"))
        for v in reaction_map.values()
        if (v or {}).get("orig_key") is not None
    }
    start_species_name = yaks_metadata.get("start_species_name")

    # Backward-compatible fallback for existing sidecar-map outputs.
    if not reaction_map:
        reaction_map_path = yaml_path.with_suffix(".reaction_map.yaml")
        if reaction_map_path.exists():
            reaction_map = load_yaml(reaction_map_path)

    with pickle_path.open("rb") as f:
        payload = pickle.load(f)

    metadata = payload.get("metadata", {}) or {}
    start_smiles = get_meta_smiles(metadata, yaks_metadata, "start")
    end_smiles = get_meta_smiles(metadata, yaks_metadata, "end")

    if not start_species_name:
        start_species_name = start_smiles

    reactor_cfg = cfg.get("reactor", {}) or {}
    terminal_completion_mode = str(reactor_cfg.get("terminal_completion_mode", "auto"))
    terminal_product_states_seed = list(metadata.get("terminal_product_states", []) or [])
    if not terminal_product_states_seed:
        terminal_product_states_seed = list(yaks_metadata.get("terminal_product_states", []) or [])
    if not terminal_product_states_seed:
        terminal_product_states_seed = collect_terminal_product_states_from_paths(
            payload.get("paths", []),
            payload.get("rxns", {}),
        )
    if not terminal_product_states_seed and end_smiles:
        terminal_product_states_seed = [end_smiles]
    nonterminal_path_states_seed = collect_nonterminal_path_states(
        payload.get("paths", []),
        payload.get("rxns", {}),
    )

    completion_terminal_species = resolve_terminal_completion_species(
        reactor_cfg.get("terminal_species", None),
        terminal_completion_mode=terminal_completion_mode,
        end_smiles=end_smiles,
        terminal_product_states=terminal_product_states_seed,
        nonterminal_states=nonterminal_path_states_seed,
    )
    completion_tol_cfg = float(reactor_cfg.get("completion_tol", 1e-6))
    completion_target_scale_mode = str(reactor_cfg.get("completion_target_scale_mode", "max"))
    completion_target_scale = terminal_product_fraction_scale(
        end_smiles,
        terminal_product_states_seed,
        mode=completion_target_scale_mode,
    )
    completion_target_value = (1.0 - completion_tol_cfg) * completion_target_scale
    completion_target_value = max(0.0, min(1.0, completion_target_value))
    min_completion_time_cfg = float(reactor_cfg.get("min_completion_time", 0.0))
    # Preserve order while deduplicating.
    completion_terminal_species = list(dict.fromkeys(s for s in completion_terminal_species if s))
    terminal_species_cfg = None
    if completion_terminal_species:
        terminal_species_cfg = (
            completion_terminal_species[0]
            if len(completion_terminal_species) == 1
            else completion_terminal_species
        )

    if completion_terminal_species:
        print(
            "[completion-debug]",
            f"mode={terminal_completion_mode}",
            f"terminal_states={terminal_product_states_seed}",
            f"nonterminal_states={len(nonterminal_path_states_seed)}",
            f"terminal_species={completion_terminal_species}",
            f"completion_target_scale_mode={completion_target_scale_mode}",
            f"completion_target_scale={completion_target_scale:.6g}",
            f"completion_target_value={completion_target_value:.6g}",
            f"min_completion_time={min_completion_time_cfg:.6g}",
        )
    else:
        print(
            "[completion-debug] mode="
            f"{terminal_completion_mode} terminal_species=[] (completion stop disabled)"
        )
    extend_seconds_raw = reactor_cfg.get("extend_seconds", None)
    if isinstance(extend_seconds_raw, str) and extend_seconds_raw.strip().lower() in {"", "none", "null"}:
        extend_seconds_raw = None
    extend_seconds_cfg = None if extend_seconds_raw is None else float(extend_seconds_raw)
    output_cfg = cfg.get("output", {}) or {}
    try:
        net_states, states, details = build_and_run_reactor(
            str(yaml_path),
            float(reactor_cfg.get("time_sim", 100.0)),
            float(reactor_cfg.get("time_step", 0.1)),
            str(reactor_cfg.get("rule", "css")),
            int(reactor_cfg.get("curr_depth", 1)),
            uncertainty=bool(reactor_cfg.get("uncertainty", True)),
            uncertainty_cycles=int(reactor_cfg.get("uncertainty_cycles", 30)),
            scale=float(reactor_cfg.get("scale", 3.0)),
            write_excel=bool(reactor_cfg.get("write_excel", False)),
            terminal_species=terminal_species_cfg,
            fraction_basis=str(reactor_cfg.get("fraction_basis", "X")),
            completion_tol=completion_tol_cfg,
            completion_target=completion_target_value,
            completion_hold_steps=int(reactor_cfg.get("completion_hold_steps", 5)),
            completion_dxdt_tol=float(reactor_cfg.get("completion_dxdt_tol", 0.0)),
            min_completion_time=min_completion_time_cfg,
            debug_fraction=bool(reactor_cfg.get("debug_fraction", False)),
            extend_if_not_complete=bool(reactor_cfg.get("extend_if_not_complete", False)),
            extend_seconds=extend_seconds_cfg,
            max_time_multiplier=float(reactor_cfg.get("max_time_multiplier", 10.0)),
            return_details=True,
        )
    except Exception as exc:
        error_text = f"{exc.__class__.__name__}: {exc}"
        print(f"[warning] solver failure for {yaml_path.name}: {error_text}")

        run_payload = {
            "run": {
                "input_yaml": str(yaml_path),
                "subnetwork_pickle": str(pickle_path),
                "rule": reactor_cfg.get("rule", "css"),
                "time_sim": float(reactor_cfg.get("time_sim", 100.0)),
                "time_step": float(reactor_cfg.get("time_step", 0.1)),
                "uncertainty": bool(reactor_cfg.get("uncertainty", True)),
                "uncertainty_cycles": int(reactor_cfg.get("uncertainty_cycles", 30)),
                "scale": float(reactor_cfg.get("scale", 3.0)),
                "terminal_completion_mode": terminal_completion_mode,
                "completion_terminal_species": completion_terminal_species,
                "completion_target_value": float(completion_target_value),
                "min_completion_time": float(min_completion_time_cfg),
                "status": "solver_failed",
                "solver_error": error_text,
            }
        }

        if bool(output_cfg.get("write_run_yaml", True)):
            run_yaml_suffix = str(output_cfg.get("run_yaml_suffix", ".run.yaml"))
            run_yaml_path = yaml_path.with_suffix(run_yaml_suffix)
            dump_yaml(run_yaml_path, run_payload)
        # Always emit an empty to_final table so downstream steps can continue.
        table_suffix = str(output_cfg.get("final_product_flux_table_suffix", ".to_final.csv"))
        table_path = yaml_path.with_suffix(table_suffix)
        write_final_product_flux_table(table_path, [])
        # Emit empty optional tables if they are enabled.
        if bool(output_cfg.get("write_reaction_flux_table", False)):
            rxn_suffix = str(output_cfg.get("reaction_flux_table_suffix", ".flux.csv"))
            write_reaction_flux_table(yaml_path.with_suffix(rxn_suffix), [])
        if bool(output_cfg.get("write_flux_timeseries_table", True)):
            ts_suffix = str(output_cfg.get("flux_timeseries_table_suffix", ".flux_timeseries.csv"))
            write_flux_timeseries_table(yaml_path.with_suffix(ts_suffix), [])

        if bool(output_cfg.get("update_subnetwork_pickle_metadata", True)):
            payload.setdefault("metadata", {})
            payload["metadata"]["cantera_run"] = {
                "input_yaml": str(yaml_path),
                "status": "solver_failed",
                "solver_error": error_text,
                "terminal_completion_mode": terminal_completion_mode,
                "completion_terminal_species": completion_terminal_species,
                "completion_target_value": float(completion_target_value),
                "min_completion_time": float(min_completion_time_cfg),
            }
            with pickle_path.open("wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        return {
            "yaml_path": yaml_path,
            "pickle_path": pickle_path,
            "final_species_smiles": "solver_failed",
            "final_species_concentration": float("nan"),
            "all_reaction_flux_zero": True,
            "flux_direction_failures": 0,
            "missing_direction_pairs": 0,
            "flux_direction_check_enabled": False,
            "solver_failed": True,
            "solver_error": error_text,
        }

    species = [normalize_smiles_text(s) or str(s) for s in (details.get("species", []) or [])]
    if end_smiles in species:
        final_species_idx = species.index(end_smiles)
        final_species_name = end_smiles
    else:
        final_species_idx = int(max(range(len(details["final_x_mean"])), key=lambda i: details["final_x_mean"][i]))
        final_species_name = species[final_species_idx]

    final_conc = float(details["final_x_mean"][final_species_idx])
    final_conc_std = float(details["final_x_std"][final_species_idx])
    initial_x = details.get("species_x_ts_mean", [[]]) or [[]]
    initial_x = initial_x[0] if initial_x else []
    start_species_initial_x = None
    if start_species_name in species and species.index(start_species_name) < len(initial_x):
        start_species_initial_x = float(initial_x[species.index(start_species_name)])

    flux_to_final = details["species_rxn_prod_flux_mean"][final_species_idx]
    flux_to_final_std = details["species_rxn_prod_flux_std"][final_species_idx]
    rxn_cum_abs_flux = details["rxn_cum_abs_flux_mean"]
    rxn_cum_abs_flux_std = details["rxn_cum_abs_flux_std"]
    rxn_final_rate = details["rxn_final_rate_mean"]
    rxn_final_rate_std = details["rxn_final_rate_std"]

    reaction_names = sorted(reaction_map.keys(), key=parse_reaction_index) if reaction_map else []
    if not reaction_names:
        reaction_names = [f"reaction_{i+1}" for i in range(len(rxn_cum_abs_flux))]
    if not included_keys:
        included_keys = {str(k) for k in (payload.get("rxns", {}) or {}).keys()}
    reaction_metrics = {}
    reaction_rows = []
    final_product_rows = []
    flux_to_final_by_key = {}
    flux_to_final_std_by_key = {}
    total_flux_by_key = {}
    final_rate_by_key = {}

    end_in_product_count = 0
    end_in_reactant_count = 0
    end_reverse_match_count = 0
    rxn_lookup_miss_count = 0

    for reaction_name in reaction_names:
        rxn_idx = parse_reaction_index(reaction_name)
        if rxn_idx >= len(flux_to_final) or rxn_idx >= len(rxn_cum_abs_flux):
            continue
        mapping = (reaction_map.get(reaction_name) or {})
        orig_key = str(mapping.get("orig_key", reaction_name))
        equation = details.get("reactions", [])[rxn_idx] if rxn_idx < len(details.get("reactions", [])) else ""
        flux_val = float(flux_to_final[rxn_idx])
        total_flux_val = float(rxn_cum_abs_flux[rxn_idx])
        final_rate_val = float(rxn_final_rate[rxn_idx])
        rxn = get_reaction_by_key(payload.get("rxns", {}), orig_key)
        reactant_smiles = get_smiles_for_state(getattr(rxn, "reactant", None)) if rxn is not None else ""
        product_smiles = get_smiles_for_state(getattr(rxn, "product", None)) if rxn is not None else ""
        if rxn is None:
            rxn_lookup_miss_count += 1
        eq_from_state, eq_to_state = equation_states(equation)
        if not reactant_smiles and eq_from_state:
            reactant_smiles = eq_from_state
        if not product_smiles and eq_to_state:
            product_smiles = eq_to_state
        reactant_parts = split_smiles(reactant_smiles)
        product_parts = split_smiles(product_smiles)
        ends_in_product = bool(end_smiles and end_smiles in product_parts)
        ends_in_reactant = bool(end_smiles and end_smiles in reactant_parts)
        reverse_enabled = bool(mapping.get("reverse_enabled", False))
        include_as_reverse_terminal = bool(ends_in_reactant and reverse_enabled and not ends_in_product)
        if ends_in_product:
            end_in_product_count += 1
        if ends_in_reactant:
            end_in_reactant_count += 1
        if include_as_reverse_terminal:
            end_reverse_match_count += 1

        effective_from_smiles = product_smiles if include_as_reverse_terminal else reactant_smiles
        effective_to_smiles = reactant_smiles if include_as_reverse_terminal else product_smiles
        effective_from_parts = split_smiles(effective_from_smiles)
        source_type = "R" if (start_smiles and start_smiles in effective_from_parts) else "I"

        flux_to_final_by_key[orig_key] = flux_val
        flux_to_final_std_by_key[orig_key] = float(flux_to_final_std[rxn_idx])
        total_flux_by_key[orig_key] = total_flux_val
        final_rate_by_key[orig_key] = final_rate_val
        reaction_metrics[reaction_name] = {
            "orig_key": orig_key,
            "rxn_id": mapping.get("rxn_id"),
            "rxn_hash": mapping.get("rxn_hash"),
            "flux_to_final_species": flux_val,
            "flux_to_final_species_std": float(flux_to_final_std[rxn_idx]),
            "cumulative_abs_flux": total_flux_val,
            "cumulative_abs_flux_std": float(rxn_cum_abs_flux_std[rxn_idx]),
            "final_rate_of_progress": final_rate_val,
            "final_rate_of_progress_std": float(rxn_final_rate_std[rxn_idx]),
        }
        reaction_rows.append(
            {
                "reaction_index": rxn_idx + 1,
                "reaction_name": reaction_name,
                "equation": equation,
                "orig_key": orig_key,
                "rxn_id": mapping.get("rxn_id"),
                "rxn_hash": mapping.get("rxn_hash"),
                "flux_to_final_species": flux_val,
                "flux_to_final_species_std": float(flux_to_final_std[rxn_idx]),
                "cumulative_abs_flux": total_flux_val,
                "cumulative_abs_flux_std": float(rxn_cum_abs_flux_std[rxn_idx]),
                "final_rate_of_progress": final_rate_val,
                "final_rate_of_progress_std": float(rxn_final_rate_std[rxn_idx]),
            }
        )
        if ends_in_product:
            final_product_rows.append(
                {
                    "reaction_index": rxn_idx + 1,
                    "reaction_name": reaction_name,
                    "equation": equation,
                    "orig_key": orig_key,
                    "rxn_id": mapping.get("rxn_id"),
                    "rxn_hash": mapping.get("rxn_hash"),
                    "source_type": source_type,
                    "from_smiles": effective_from_smiles,
                    "to_smiles": effective_to_smiles,
                    "flux_to_final_species": flux_val,
                    "flux_to_final_species_std": float(flux_to_final_std[rxn_idx]),
                    "cumulative_abs_flux": total_flux_val,
                    "cumulative_abs_flux_std": float(rxn_cum_abs_flux_std[rxn_idx]),
                    "final_rate_of_progress": final_rate_val,
                    "final_rate_of_progress_std": float(rxn_final_rate_std[rxn_idx]),
                }
            )
        elif include_as_reverse_terminal:
            final_product_rows.append(
                {
                    "reaction_index": rxn_idx + 1,
                    "reaction_name": reaction_name,
                    "equation": equation,
                    "orig_key": orig_key,
                    "rxn_id": mapping.get("rxn_id"),
                    "rxn_hash": mapping.get("rxn_hash"),
                    "source_type": source_type,
                    "from_smiles": effective_from_smiles,
                    "to_smiles": effective_to_smiles,
                    "flux_to_final_species": flux_val,
                    "flux_to_final_species_std": float(flux_to_final_std[rxn_idx]),
                    "cumulative_abs_flux": total_flux_val,
                    "cumulative_abs_flux_std": float(rxn_cum_abs_flux_std[rxn_idx]),
                    "final_rate_of_progress": final_rate_val,
                    "final_rate_of_progress_std": float(rxn_final_rate_std[rxn_idx]),
                }
            )

    path_flux_records = build_path_flux_records(
        payload.get("paths", []),
        payload.get("rxns", {}),
        flux_to_final_by_key,
        start_smiles=start_smiles,
        end_smiles=end_smiles,
    )
    terminal_product_states = collect_terminal_product_states(path_flux_records)
    if not terminal_product_states and end_smiles:
        terminal_product_states = [end_smiles]
    path_intermediate_states = collect_on_target_intermediate_states(
        path_flux_records,
        start_smiles,
        terminal_product_states,
    )
    direct_path_intermediate_states = collect_direct_on_target_intermediate_states(
        path_flux_records,
        start_smiles,
        terminal_product_states,
    )
    path_direction_rows = check_directional_presence_for_on_target(
        payload.get("rxns", {}),
        included_keys,
        reaction_map,
        start_smiles,
        terminal_product_states,
        path_intermediate_states,
    )
    direction_by_state = {
        str(row.get("intermediate_state")): row
        for row in path_direction_rows
        if row.get("intermediate_state")
    }
    on_target_intermediate_states = sorted(
        state
        for state, row in direction_by_state.items()
        if bool(row.get("has_r_to_i")) and bool(row.get("has_i_to_p"))
    )
    direction_rows = [direction_by_state[state] for state in on_target_intermediate_states if state in direction_by_state]
    direct_on_target_intermediate_states = sorted(
        set(on_target_intermediate_states) & set(direct_path_intermediate_states)
    )
    if not direct_on_target_intermediate_states:
        direct_on_target_intermediate_states = list(on_target_intermediate_states)
    on_target_intermediates = collect_species_from_states(on_target_intermediate_states)
    direct_on_target_intermediates = collect_species_from_states(direct_on_target_intermediate_states)
    time_grid = details.get("time_grid", []) or []
    rxn_cum_abs_flux_ts_mean = details.get("rxn_cum_abs_flux_ts_mean", []) or []
    rxn_cum_abs_flux_ts_std = details.get("rxn_cum_abs_flux_ts_std", []) or []
    flux_ts_rows = []
    n_reactions = len(rxn_cum_abs_flux)
    for rxn_idx in range(n_reactions):
        reaction_name = f"reaction_{rxn_idx + 1}"
        mapping = (reaction_map.get(reaction_name) or {})
        mapped_orig_key = mapping.get("orig_key")
        orig_key = str(mapped_orig_key) if mapped_orig_key is not None else reaction_name
        equation = details.get("reactions", [])[rxn_idx] if rxn_idx < len(details.get("reactions", [])) else ""
        for t_i, time_s in enumerate(time_grid):
            if t_i >= len(rxn_cum_abs_flux_ts_mean):
                continue
            row_vals = rxn_cum_abs_flux_ts_mean[t_i]
            row_std = rxn_cum_abs_flux_ts_std[t_i] if t_i < len(rxn_cum_abs_flux_ts_std) else []
            if rxn_idx >= len(row_vals):
                continue
            flux_ts_rows.append(
                {
                    "time_s": float(time_s),
                    "reaction_index": rxn_idx + 1,
                    "reaction_name": reaction_name,
                    "equation": equation,
                    "orig_key": orig_key,
                    "rxn_id": mapping.get("rxn_id"),
                    "rxn_hash": mapping.get("rxn_hash"),
                    "cumulative_abs_flux": float(row_vals[rxn_idx]),
                    "cumulative_abs_flux_std": float(row_std[rxn_idx]) if rxn_idx < len(row_std) else 0.0,
                }
            )
    reaction_part_maps = build_reaction_part_maps(payload.get("rxns", {}))
    species_flux_rows = species_flux_timeseries_from_rows(flux_ts_rows, reaction_part_maps)
    final_species_flux = final_species_flux_lookup(species_flux_rows)
    debug_flux_direction = bool(reactor_cfg.get("debug_flux_direction", False))
    debug_flux_threshold = float(reactor_cfg.get("debug_flux_threshold", 0.0))
    flux_direction_checks = []
    flux_direction_failures = []
    missing_direction_pairs = []
    if debug_flux_direction:
        (
            flux_direction_checks,
            flux_direction_failures,
            missing_direction_pairs,
        ) = print_flux_direction_debug(
            final_species_flux=final_species_flux,
            start_smiles=start_smiles,
            end_smiles=end_smiles,
            on_target_intermediates=on_target_intermediates,
            direction_rows=direction_rows,
            threshold=debug_flux_threshold,
        )
    final_product_rows_sorted = sorted(
        final_product_rows,
        key=lambda row: row["cumulative_abs_flux"],
        reverse=True,
    )
    if not final_product_rows_sorted:
        print(
            "[to_final-debug]",
            f"rows=0",
            f"end_smiles={end_smiles}",
            f"n_reactions={len(reaction_names)}",
            f"end_in_product={end_in_product_count}",
            f"end_in_reactant={end_in_reactant_count}",
            f"reverse_terminal_matches={end_reverse_match_count}",
            f"rxn_lookup_miss={rxn_lookup_miss_count}",
        )
    max_abs_cum_flux = max((abs(float(v)) for v in rxn_cum_abs_flux), default=0.0)
    nonzero_flux_count = sum(1 for v in rxn_cum_abs_flux if abs(float(v)) > 0.0)

    run_payload = {
        "run": {
            "input_yaml": str(yaml_path),
            "subnetwork_pickle": str(pickle_path),
            "rule": reactor_cfg.get("rule", "css"),
            "time_sim": float(reactor_cfg.get("time_sim", 100.0)),
            "time_step": float(reactor_cfg.get("time_step", 0.1)),
            "uncertainty": bool(reactor_cfg.get("uncertainty", True)),
            "uncertainty_cycles": int(reactor_cfg.get("uncertainty_cycles", 30)),
            "scale": float(reactor_cfg.get("scale", 3.0)),
            "final_species_name": final_species_name,
            "final_species_smiles": final_species_name,
            "final_species_concentration": final_conc,
            "final_species_concentration_std": final_conc_std,
            "start_species_name": start_species_name,
            "start_species_initial_concentration": start_species_initial_x,
            "terminal_completion_mode": terminal_completion_mode,
            "terminal_product_states_for_completion": terminal_product_states_seed,
            "nonterminal_path_states_for_completion": nonterminal_path_states_seed,
            "completion_terminal_species": completion_terminal_species,
            "completion_target_scale_mode": completion_target_scale_mode,
            "completion_target_scale": float(completion_target_scale),
            "completion_target_value": float(completion_target_value),
            "min_completion_time": float(min_completion_time_cfg),
            "completion_tol": float(completion_tol_cfg),
            "completion_hold_steps": int(reactor_cfg.get("completion_hold_steps", 5)),
            "completion_dxdt_tol": float(reactor_cfg.get("completion_dxdt_tol", 0.0)),
            "fraction_basis": str(reactor_cfg.get("fraction_basis", "X")),
            "adaptive_extension": details.get("adaptive_extension", {}),
            "confidence_runs_via_barrier_perturbation": bool(reactor_cfg.get("uncertainty", True)),
        },
        "paths": path_flux_records,
        "reactions": reaction_metrics,
        "reactions_to_final_product": final_product_rows_sorted,
        "species_flux_final": final_species_flux,
        "terminal_product_states": terminal_product_states,
        "path_intermediate_states": path_intermediate_states,
        "direct_path_intermediate_states": direct_path_intermediate_states,
        "on_target_intermediate_states": on_target_intermediate_states,
        "direct_on_target_intermediate_states": direct_on_target_intermediate_states,
        "on_target_intermediates": on_target_intermediates,
        "direct_on_target_intermediates": direct_on_target_intermediates,
        "flux_direction_checks": flux_direction_checks,
    }

    preview_rows = int(output_cfg.get("table_preview_rows", 5))
    if bool(output_cfg.get("write_run_yaml", True)):
        run_yaml_suffix = str(output_cfg.get("run_yaml_suffix", ".run.yaml"))
        run_yaml_path = yaml_path.with_suffix(run_yaml_suffix)
        dump_yaml(run_yaml_path, run_payload)
    if bool(output_cfg.get("write_reaction_flux_table", False)):
        table_suffix = str(output_cfg.get("reaction_flux_table_suffix", ".flux.csv"))
        table_path = yaml_path.with_suffix(table_suffix)
        sorted_rows = sorted(reaction_rows, key=lambda row: row["cumulative_abs_flux"], reverse=True)
        write_reaction_flux_table(table_path, sorted_rows)
        print_table_preview(f"{table_path.name}", sorted_rows, max_rows=preview_rows)
    if bool(output_cfg.get("write_final_product_flux_table", True)):
        table_suffix = str(output_cfg.get("final_product_flux_table_suffix", ".to_final.csv"))
        table_path = yaml_path.with_suffix(table_suffix)
        write_final_product_flux_table(table_path, final_product_rows_sorted)
        print_table_preview(f"{table_path.name}", final_product_rows_sorted, max_rows=preview_rows)
    if bool(output_cfg.get("write_flux_timeseries_table", True)):
        table_suffix = str(output_cfg.get("flux_timeseries_table_suffix", ".flux_timeseries.csv"))
        table_path = yaml_path.with_suffix(table_suffix)
        write_flux_timeseries_table(table_path, flux_ts_rows)
        print_table_preview(f"{table_path.name}", flux_ts_rows, max_rows=preview_rows)
    if bool(output_cfg.get("write_reactor_debug_yaml", False)):
        debug_suffix = str(output_cfg.get("reactor_debug_yaml_suffix", ".reactor_debug.yaml"))
        debug_path = yaml_path.with_suffix(debug_suffix)
        top_n = int(output_cfg.get("reactor_debug_top_n", 10))
        top_initial = sorted(
            (
                {"species": name, "x": float(x_val)}
                for name, x_val in zip(species, initial_x)
            ),
            key=lambda row: row["x"],
            reverse=True,
        )[:top_n]
        top_final = sorted(
            (
                {"species": name, "x": float(x_val)}
                for name, x_val in zip(species, details.get("final_x_mean", []))
            ),
            key=lambda row: row["x"],
            reverse=True,
        )[:top_n]
        top_flux = sorted(reaction_rows, key=lambda row: abs(float(row["cumulative_abs_flux"])), reverse=True)[:top_n]
        write_reactor_debug_yaml(
            debug_path,
            {
                "run": run_payload["run"],
                "diagnostics": {
                    "n_species": len(species),
                    "n_reactions": len(rxn_cum_abs_flux),
                    "n_nonzero_reactions_by_cumulative_flux": int(nonzero_flux_count),
                    "max_abs_cumulative_flux": float(max_abs_cum_flux),
                    "all_reaction_flux_zero": bool(max_abs_cum_flux == 0.0),
                    "n_on_target_intermediates": int(len(on_target_intermediates)),
                    "n_direct_on_target_intermediates": int(len(direct_on_target_intermediates)),
                    "n_path_intermediate_states": int(len(path_intermediate_states)),
                    "n_direct_path_intermediate_states": int(len(direct_path_intermediate_states)),
                    "n_on_target_intermediate_states": int(len(on_target_intermediate_states)),
                    "n_direct_on_target_intermediate_states": int(len(direct_on_target_intermediate_states)),
                    "flux_direction_check_enabled": bool(debug_flux_direction),
                    "flux_direction_check_failures": int(len(flux_direction_failures)),
                    "missing_reagent_intermediate_direction_pairs": int(len(missing_direction_pairs)),
                    "adaptive_extension": details.get("adaptive_extension", {}),
                },
                "top_species_initial_x": top_initial,
                "top_species_final_x": top_final,
                "top_reactions_by_abs_cumulative_flux": top_flux,
                "species_flux_final": final_species_flux,
                "terminal_product_states": terminal_product_states,
                "path_intermediate_states": path_intermediate_states,
                "direct_path_intermediate_states": direct_path_intermediate_states,
                "on_target_intermediate_states": on_target_intermediate_states,
                "direct_on_target_intermediate_states": direct_on_target_intermediate_states,
                "on_target_intermediates": on_target_intermediates,
                "direct_on_target_intermediates": direct_on_target_intermediates,
                "flux_direction_checks": flux_direction_checks,
                "on_target_direction_presence": direction_rows,
            },
        )

    if bool(output_cfg.get("update_subnetwork_pickle_metadata", True)):
        final_x_mean = details.get("final_x_mean", []) or []
        final_x_std = details.get("final_x_std", []) or []
        species_x_ts_mean = details.get("species_x_ts_mean", []) or []
        species_x_ts_std = details.get("species_x_ts_std", []) or []
        time_grid = details.get("time_grid", []) or []
        final_conc_by_species = {}
        final_conc_std_by_species = {}
        species_ts_by_species = {}
        species_ts_std_by_species = {}
        for i, species_name in enumerate(species):
            if i >= len(final_x_mean):
                continue
            x_val = float(final_x_mean[i])
            x_std = float(final_x_std[i]) if i < len(final_x_std) else 0.0
            final_conc_by_species[str(species_name)] = x_val
            final_conc_std_by_species[str(species_name)] = x_std
            ts_vals = [float(row[i]) for row in species_x_ts_mean if i < len(row)]
            ts_std_vals = [float(row[i]) for row in species_x_ts_std if i < len(row)]
            species_ts_by_species[str(species_name)] = ts_vals
            species_ts_std_by_species[str(species_name)] = ts_std_vals

        payload.setdefault("metadata", {})
        payload["metadata"]["final_species_name"] = final_species_name
        payload["metadata"]["final_species_token"] = final_species_name
        payload["metadata"]["final_species_smiles"] = final_species_name
        payload["metadata"]["final_species_concentration"] = final_conc
        payload["metadata"]["final_species_concentration_std"] = final_conc_std
        payload["metadata"]["final_species_concentration_by_species"] = final_conc_by_species
        payload["metadata"]["final_species_concentration_std_by_species"] = final_conc_std_by_species
        payload["metadata"]["final_species_concentration_by_token"] = final_conc_by_species
        payload["metadata"]["final_species_concentration_std_by_token"] = final_conc_std_by_species
        payload["metadata"]["final_species_concentration_by_smiles"] = final_conc_by_species
        payload["metadata"]["final_species_concentration_std_by_smiles"] = final_conc_std_by_species
        payload["metadata"]["species_time_grid_s"] = [float(t) for t in time_grid]
        payload["metadata"]["species_concentration_timeseries_by_smiles"] = species_ts_by_species
        payload["metadata"]["species_concentration_timeseries_std_by_smiles"] = species_ts_std_by_species
        payload["metadata"]["reaction_flux_to_final_by_key"] = flux_to_final_by_key
        payload["metadata"]["reaction_flux_to_final_std_by_key"] = flux_to_final_std_by_key
        payload["metadata"]["reaction_total_cumulative_flux_by_key"] = total_flux_by_key
        payload["metadata"]["reaction_final_rate_by_key"] = final_rate_by_key
        payload["metadata"]["path_flux_records"] = path_flux_records
        payload["metadata"]["reactions_to_final_product"] = final_product_rows_sorted
        payload["metadata"]["species_flux_final"] = final_species_flux
        payload["metadata"]["terminal_product_states"] = terminal_product_states
        payload["metadata"]["path_intermediate_states"] = path_intermediate_states
        payload["metadata"]["direct_path_intermediate_states"] = direct_path_intermediate_states
        payload["metadata"]["on_target_intermediate_states"] = on_target_intermediate_states
        payload["metadata"]["direct_on_target_intermediate_states"] = direct_on_target_intermediate_states
        payload["metadata"]["on_target_intermediates"] = on_target_intermediates
        payload["metadata"]["direct_on_target_intermediates"] = direct_on_target_intermediates
        payload["metadata"]["on_target_direction_presence"] = direction_rows
        phase_state = {}
        try:
            phase_state = (
                ((cantera_yaml_payload or {}).get("phases", []) or [{}])[0].get("state", {}) or {}
            )
        except Exception:
            phase_state = {}

        payload["metadata"]["cantera_run"] = {
            "input_yaml": str(yaml_path),
            "rule": reactor_cfg.get("rule", "css"),
            "time_sim": float(reactor_cfg.get("time_sim", 100.0)),
            "time_step": float(reactor_cfg.get("time_step", 0.1)),
            "uncertainty": bool(reactor_cfg.get("uncertainty", True)),
            "uncertainty_cycles": int(reactor_cfg.get("uncertainty_cycles", 30)),
            "scale": float(reactor_cfg.get("scale", 3.0)),
            "debug_flux_direction": bool(debug_flux_direction),
            "debug_flux_threshold": float(debug_flux_threshold),
            "extend_if_not_complete": bool(reactor_cfg.get("extend_if_not_complete", False)),
            "extend_seconds": extend_seconds_cfg,
            "max_time_multiplier": float(reactor_cfg.get("max_time_multiplier", 10.0)),
            "terminal_completion_mode": terminal_completion_mode,
            "terminal_product_states_for_completion": terminal_product_states_seed,
            "nonterminal_path_states_for_completion": nonterminal_path_states_seed,
            "completion_terminal_species": completion_terminal_species,
            "completion_target_scale_mode": completion_target_scale_mode,
            "completion_target_scale": float(completion_target_scale),
            "completion_target_value": float(completion_target_value),
            "min_completion_time": float(min_completion_time_cfg),
            "completion_tol": float(completion_tol_cfg),
            "completion_hold_steps": int(reactor_cfg.get("completion_hold_steps", 5)),
            "completion_dxdt_tol": float(reactor_cfg.get("completion_dxdt_tol", 0.0)),
            "fraction_basis": str(reactor_cfg.get("fraction_basis", "X")),
            "temperature_K": (
                float(phase_state.get("T"))
                if phase_state.get("T") is not None
                else None
            ),
            "pressure_atm": (
                float(phase_state.get("P"))
                if phase_state.get("P") is not None
                else None
            ),
            "adaptive_extension": details.get("adaptive_extension", {}),
            "confidence_runs_via_barrier_perturbation": bool(reactor_cfg.get("uncertainty", True)),
        }
        with pickle_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    return {
        "yaml_path": yaml_path,
        "pickle_path": pickle_path,
        "final_species_smiles": final_species_name,
        "final_species_concentration": final_conc,
        "all_reaction_flux_zero": bool(max_abs_cum_flux == 0.0),
        "flux_direction_failures": int(len(flux_direction_failures)),
        "missing_direction_pairs": int(len(missing_direction_pairs)),
        "flux_direction_check_enabled": bool(debug_flux_direction),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Cantera reactor simulations for subnetwork YAML files."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to pipeline_config.yaml (default: ./pipeline/configs/pipeline_config.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path()
    cfg, cfg_dir = load_config(config_path)
    apply_runtime_module_paths(cfg, cfg_dir)

    yaml_root, all_files, selected = collect_cantera_yamls(cfg, cfg_dir)
    print(f"Found {len(all_files)} subnetwork Cantera YAML(s) in {yaml_root}")
    print(f"Selected {len(selected)} YAML(s) for this run")

    summaries = []
    use_tqdm = bool((cfg.get("logging", {}) or {}).get("progress_bar", True)) and tqdm is not None
    iterator = selected
    if use_tqdm:
        iterator = tqdm(selected, total=len(selected), desc="Cantera runs", unit="run")

    for i, yaml_path in enumerate(iterator, start=1):
        summary = run_one_yaml(yaml_path, yaml_root, cfg, cfg_dir)
        summaries.append(summary)
        line = (
            f"[{i}/{len(selected)}] {yaml_path.name} | "
            f"final={summary['final_species_smiles']} | "
            f"x_final={summary['final_species_concentration']:.6e}"
        )
        if summary.get("all_reaction_flux_zero"):
            line += " | WARNING: all reaction cumulative fluxes are zero"
        if summary.get("solver_failed"):
            line += f" | solver_failed={summary.get('solver_error', '')}"
        if summary.get("flux_direction_check_enabled"):
            line += (
                f" | flux_dir_failures={summary.get('flux_direction_failures', 0)}"
                f" | missing_RI_pairs={summary.get('missing_direction_pairs', 0)}"
            )
        if use_tqdm and tqdm is not None:
            tqdm.write(line)
        else:
            print(line)

    max_print = int((cfg.get("logging", {}) or {}).get("max_print", 50))
    print("\nRun summary:")
    for s in summaries[:max_print]:
        print(
            f"- {s['yaml_path'].name} -> {s['pickle_path'].name} | "
            f"final={s['final_species_smiles']} | x_final={s['final_species_concentration']:.6e}"
            f" | flux_dir_failures={s.get('flux_direction_failures', 0)}"
            f" | missing_RI_pairs={s.get('missing_direction_pairs', 0)}"
            f" | solver_failed={bool(s.get('solver_failed', False))}"
        )
    if len(summaries) > max_print:
        print(f"... and {len(summaries) - max_print} more files")


if __name__ == "__main__":
    main()
