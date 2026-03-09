#!/usr/bin/env python3
"""Build compact product-level and retained-timeseries tables for one subnetwork."""

import argparse
import json
import math
import os
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from rdkit import Chem


def default_config_path():
    """Return the default pipeline config path."""
    cwd_cfg = Path("pipeline/configs/pipeline_config.yaml")
    if cwd_cfg.exists():
        return cwd_cfg
    return Path(__file__).resolve().parent / "configs" / "pipeline_config.yaml"


def load_config(config_path):
    """Load YAML config as a plain dict."""
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg if isinstance(cfg, dict) else {}


def split_smiles(smiles):
    """Split dot-delimited SMILES into components."""
    if smiles is None:
        return []
    try:
        if pd.isna(smiles):
            return []
    except Exception:
        pass
    return [part.strip() for part in str(smiles).split(".") if part.strip()]


def split_equation_side(side):
    """Split one Cantera equation side into species tokens."""
    return [tok.strip() for tok in str(side or "").split(" + ") if tok.strip()]


def parse_equation_token(token):
    """Parse one equation token into (species_text, stoich_coeff)."""
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
    """Parse reactant/product species counters from a Cantera equation string."""
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
    """Parse normalized full reactant/product state SMILES from a Cantera equation."""
    text = str(equation or "")
    if "=>" not in text:
        return pd.NA, pd.NA
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
    left_state = ".".join(left_parts) if left_parts else pd.NA
    right_state = ".".join(right_parts) if right_parts else pd.NA
    return left_state, right_state


def parse_reaction_index(reaction_name):
    """Parse Cantera reaction name like reaction_12 into zero-based index."""
    try:
        return int(str(reaction_name).split("_")[-1]) - 1
    except Exception:
        return -1


def normalize_smiles(smiles):
    """Normalize one SMILES by removing stereochemistry."""
    smi = str(smiles or "").strip()
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def normalize_smiles_text(smiles):
    """Normalize a (possibly multi-component) SMILES string."""
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


def smiles_counter(smiles):
    """Convert a SMILES string to a component counter."""
    return Counter(split_smiles(smiles))


def state_smiles(state):
    """Build normalized SMILES string from a YARP state object."""
    smi = getattr(state, "canon_smi", None)
    if smi:
        return normalize_smiles_text(smi) or str(smi)
    parts = []
    for sp in (getattr(state, "species", None) or []):
        sub = getattr(sp, "canon_smi", None)
        if sub:
            normalized = normalize_smiles_text(sub) or str(sub)
            parts.append(normalized)
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ".".join(sorted(parts))


def reaction_matches(rxn, reactant_smiles, product_smiles):
    """Match a reaction by exact normalized reactant/product multisets."""
    if not reactant_smiles or not product_smiles:
        return False
    reactant_query = smiles_counter(normalize_smiles_text(reactant_smiles) or "")
    product_query = smiles_counter(normalize_smiles_text(product_smiles) or "")
    reactant_state = smiles_counter(state_smiles(getattr(rxn, "reactant", None)))
    product_state = smiles_counter(state_smiles(getattr(rxn, "product", None)))
    return reactant_state == reactant_query and product_state == product_query


def barrier_value(value, label):
    """Extract a scalar barrier value from scalar/dict storage."""
    if value is None:
        return pd.NA
    if isinstance(value, dict):
        if label in value:
            return float(value[label])
        for v in value.values():
            try:
                return float(v)
            except Exception:
                continue
        return pd.NA
    try:
        return float(value)
    except Exception:
        return pd.NA


def forward_barrier(rxn, label):
    """Get forward barrier from reaction object."""
    val = getattr(rxn, "forward_barrier", None)
    if val is None:
        val = getattr(rxn, "barrier", None)
    return barrier_value(val, label)


def reverse_barrier(rxn, label):
    """Get reverse barrier from reaction object."""
    return barrier_value(getattr(rxn, "reverse_barrier", None), label)


def pick_reaction(rxns, reactant_smiles, product_smiles, preferred_keys=None):
    """Pick a matching reaction, preferring keys included in the YAML map."""
    preferred = {str(k) for k in (preferred_keys or set())}
    fallback = None
    for key, rxn in sorted((rxns or {}).items(), key=lambda kv: str(kv[0])):
        if reaction_matches(rxn, reactant_smiles, product_smiles):
            if str(key) in preferred:
                return str(key), rxn
            if fallback is None:
                fallback = (str(key), rxn)
    return fallback if fallback is not None else (None, None)


def get_meta_smiles(meta, key):
    """Read normalized SMILES from subnetwork metadata."""
    value = meta.get(f"{key}_smiles")
    if value is None:
        value = (meta.get(key) or {}).get("canon_smi")
    return normalize_smiles_text(value)


def state_parts(state):
    """Return normalized component parts for a reaction state."""
    return split_smiles(state_smiles(state))


def species_hash_lookup(rxns):
    """Map normalized species SMILES to observed yarpecule hash values."""
    lookup = {}
    for rxn in (rxns or {}).values():
        for state in (getattr(rxn, "reactant", None), getattr(rxn, "product", None)):
            for sp in (getattr(state, "species", None) or []):
                smi = normalize_smiles_text(getattr(sp, "canon_smi", None))
                if not smi:
                    continue
                lookup.setdefault(smi, set())
                sp_hash = getattr(sp, "hash", None)
                if sp_hash is not None:
                    lookup[smi].add(str(sp_hash))
    return {k: sorted(v) for k, v in lookup.items()}


def state_hash_lookup(rxns):
    """Map normalized full-state SMILES to observed state hash values."""
    lookup = {}
    for rxn in (rxns or {}).values():
        for state in (getattr(rxn, "reactant", None), getattr(rxn, "product", None)):
            smi = normalize_smiles_text(state_smiles(state))
            if not smi:
                continue
            lookup.setdefault(smi, set())
            state_hash = getattr(state, "hash", None)
            if state_hash is not None:
                lookup[smi].add(str(state_hash))
    return {k: sorted(v) for k, v in lookup.items()}


def hash_candidates_for_smiles(smiles, *, state_hashes=None, species_hashes=None):
    """Return ordered unique hash candidates for a normalized state SMILES string."""
    normalized = normalize_smiles_text(smiles)
    if not normalized:
        return []
    values = []
    for value in (state_hashes or {}).get(normalized, []):
        text = clean_text(value)
        if text:
            values.append(text)
    for part in split_smiles(normalized):
        for value in (species_hashes or {}).get(part, []):
            text = clean_text(value)
            if text:
                values.append(text)
    if not values:
        return []
    return sorted(dict.fromkeys(values))


def collect_all_intermediates(rxns, reagent_smiles, product_smiles):
    """Collect all species in the subnetwork except reagent/product."""
    species = set()
    for rxn in (rxns or {}).values():
        for part in state_parts(getattr(rxn, "reactant", None)):
            if part:
                species.add(part)
        for part in state_parts(getattr(rxn, "product", None)):
            if part:
                species.add(part)
    species.discard(reagent_smiles)
    species.discard(product_smiles)
    return sorted(species)


def collect_on_target_intermediates(path_records, reagent_smiles, product_smiles):
    """Collect intermediates that appear on any saved path to product."""
    on_target = set()
    for rec in (path_records or []):
        for step in (rec.get("steps") or []):
            for part in split_smiles(normalize_smiles_text(step.get("reactant_smiles"))):
                if part and part != reagent_smiles and part != product_smiles:
                    on_target.add(part)
            for part in split_smiles(normalize_smiles_text(step.get("product_smiles"))):
                if part and part != reagent_smiles and part != product_smiles:
                    on_target.add(part)
    return sorted(on_target)


def collect_direct_on_target_intermediates(path_records, reagent_smiles, product_smiles):
    """Collect first-step intermediates directly downstream of reagent."""
    direct = set()
    for rec in (path_records or []):
        steps = sorted((rec.get("steps") or []), key=lambda row: int(row.get("step", 0)))
        if not steps:
            continue
        for part in split_smiles(normalize_smiles_text(steps[0].get("product_smiles"))):
            if part and part != reagent_smiles and part != product_smiles:
                direct.add(part)
    return sorted(direct)


def collect_terminal_product_states(path_records):
    """Collect full terminal product state SMILES from the final step of each path."""
    terminal_states = set()
    for rec in (path_records or []):
        steps = sorted((rec.get("steps") or []), key=lambda row: int(row.get("step", 0)))
        if not steps:
            continue
        state = normalize_smiles_text(steps[-1].get("product_smiles"))
        if state:
            terminal_states.add(state)
    return sorted(terminal_states)


def collect_on_target_intermediate_states(path_records, reagent_smiles, terminal_states):
    """Collect full intermediate states on paths, excluding reagent and terminal states."""
    terminal_state_set = set(terminal_states or [])
    on_target = set()
    for rec in (path_records or []):
        for step in (rec.get("steps") or []):
            reactant_state = normalize_smiles_text(step.get("reactant_smiles"))
            product_state = normalize_smiles_text(step.get("product_smiles"))
            if reactant_state and reactant_state != reagent_smiles and reactant_state not in terminal_state_set:
                on_target.add(reactant_state)
            if product_state and product_state != reagent_smiles and product_state not in terminal_state_set:
                on_target.add(product_state)
    return sorted(on_target)


def collect_direct_on_target_intermediate_states(path_records, reagent_smiles, terminal_states):
    """Collect full first-step product states downstream of reagent."""
    terminal_state_set = set(terminal_states or [])
    direct = set()
    for rec in (path_records or []):
        steps = sorted((rec.get("steps") or []), key=lambda row: int(row.get("step", 0)))
        if not steps:
            continue
        first_state = normalize_smiles_text(steps[0].get("product_smiles"))
        if first_state and first_state != reagent_smiles and first_state not in terminal_state_set:
            direct.add(first_state)
    return sorted(direct)


def collect_species_from_states(states):
    """Expand full-state SMILES into unique component species."""
    species = set()
    for state in (states or []):
        for part in split_smiles(normalize_smiles_text(state)):
            if part:
                species.add(part)
    return sorted(species)


def get_reaction_by_key(rxns, rxn_key):
    """Lookup a reaction object by key text."""
    if rxn_key in (rxns or {}):
        return rxns[rxn_key]
    key_text = str(rxn_key)
    for key, rxn in (rxns or {}).items():
        if str(key) == key_text:
            return rxn
    return None


def iter_yaml_reactions(rxns, reaction_map):
    """Yield included YAML reactions with full states, component sets, and reverse flags."""
    if reaction_map:
        ordered = sorted(
            reaction_map.items(),
            key=lambda kv: (
                0,
                int(str(kv[0]).split("_")[-1]),
            ) if str(kv[0]).startswith("reaction_") else (1, str(kv[0])),
        )
        for _, mapping in ordered:
            orig_key = str((mapping or {}).get("orig_key", ""))
            if not orig_key:
                continue
            rxn = get_reaction_by_key(rxns, orig_key)
            if rxn is None:
                continue
            reactant_state = state_smiles(getattr(rxn, "reactant", None))
            product_state = state_smiles(getattr(rxn, "product", None))
            yield {
                "orig_key": orig_key,
                "reverse_enabled": bool((mapping or {}).get("reverse_enabled", False)),
                "reactant_state": reactant_state,
                "product_state": product_state,
                "reactant_parts": set(split_smiles(reactant_state)),
                "product_parts": set(split_smiles(product_state)),
                "rxn": rxn,
            }
        return

    for key, rxn in (rxns or {}).items():
        reactant_state = state_smiles(getattr(rxn, "reactant", None))
        product_state = state_smiles(getattr(rxn, "product", None))
        yield {
            "orig_key": str(key),
            "reverse_enabled": bool(False),
            "reactant_state": reactant_state,
            "product_state": product_state,
            "reactant_parts": set(split_smiles(reactant_state)),
            "product_parts": set(split_smiles(product_state)),
            "rxn": rxn,
        }


def direction_flags_for_state(yaml_reactions, reagent_state, terminal_states, current_state):
    """Return state-level direction flags (R<->I and I->P) for one intermediate state."""
    has_r_to_i = False
    has_i_to_r = False
    has_i_to_p = False
    terminal_state_set = set(terminal_states or [])
    for row in (yaml_reactions or []):
        reactant_state = normalize_smiles_text(row.get("reactant_state"))
        product_state = normalize_smiles_text(row.get("product_state"))
        reverse_enabled = bool(row.get("reverse_enabled", False))
        if reagent_state and reactant_state == reagent_state and product_state == current_state:
            has_r_to_i = True
            if reverse_enabled:
                has_i_to_r = True
        if reagent_state and reactant_state == current_state and product_state == reagent_state:
            has_i_to_r = True
        if terminal_state_set and reactant_state == current_state and product_state in terminal_state_set:
            has_i_to_p = True
    return has_r_to_i, has_i_to_r, has_i_to_p


def has_direct_r_to_p(yaml_reactions, reagent_smiles, product_smiles):
    """Check whether a direct reagent-to-product reaction exists in YAML."""
    for row in yaml_reactions:
        if reagent_smiles in row["reactant_parts"] and product_smiles in row["product_parts"]:
            return True, bool(row["reverse_enabled"])
    return False, False


def build_reaction_part_maps(rxns):
    """Map reaction key to reactant/product component counters."""
    maps = {}
    for key, rxn in (rxns or {}).items():
        reactant_counter = smiles_counter(state_smiles(getattr(rxn, "reactant", None)))
        product_counter = smiles_counter(state_smiles(getattr(rxn, "product", None)))
        maps[str(key)] = (reactant_counter, product_counter)
    return maps


def build_reaction_state_map(rxns):
    """Map reaction key to full normalized reactant/product state strings."""
    mapping = {}
    for key, rxn in (rxns or {}).items():
        mapping[str(key)] = {
            "from_smiles": state_smiles(getattr(rxn, "reactant", None)),
            "to_smiles": state_smiles(getattr(rxn, "product", None)),
        }
    return mapping


def species_flux_timeseries_from_flux_rows(flux_ts, reaction_part_maps):
    """Build species in/out cumulative flux trajectories from reaction flux table."""
    if flux_ts.empty or not {"time_s", "orig_key", "cumulative_abs_flux"}.issubset(set(flux_ts.columns)):
        return pd.DataFrame(
            columns=[
                "time_s",
                "species_smiles",
                "cumulative_in_flux",
                "cumulative_in_flux_std",
                "cumulative_out_flux",
                "cumulative_out_flux_std",
            ]
        )

    std_col = "cumulative_abs_flux_std" if "cumulative_abs_flux_std" in flux_ts.columns else None
    eq_col = "equation" if "equation" in flux_ts.columns else None
    keep_cols = ["time_s", "orig_key", "cumulative_abs_flux"] + ([std_col] if std_col else []) + ([eq_col] if eq_col else [])
    work = flux_ts[keep_cols].copy()
    work["time_s"] = pd.to_numeric(work["time_s"], errors="coerce")
    work["cumulative_abs_flux"] = pd.to_numeric(work["cumulative_abs_flux"], errors="coerce")
    if std_col:
        work["cumulative_abs_flux_std"] = pd.to_numeric(work["cumulative_abs_flux_std"], errors="coerce").fillna(0.0)
    else:
        work["cumulative_abs_flux_std"] = 0.0
    work["orig_key"] = work["orig_key"].astype(str)
    if eq_col:
        work["equation"] = work["equation"].fillna("").astype(str)
    else:
        work["equation"] = ""
    work = work.dropna(subset=["time_s", "cumulative_abs_flux"])
    if work.empty:
        return pd.DataFrame(
            columns=[
                "time_s",
                "species_smiles",
                "cumulative_in_flux",
                "cumulative_in_flux_std",
                "cumulative_out_flux",
                "cumulative_out_flux_std",
            ]
        )

    if {"time_s", "reaction_index"}.issubset(set(flux_ts.columns)):
        work = (
            work.assign(reaction_index=pd.to_numeric(flux_ts["reaction_index"], errors="coerce"))
            .sort_values(["time_s", "reaction_index"])
            .drop_duplicates(subset=["time_s", "reaction_index"], keep="last")
        )
    elif {"time_s", "reaction_name"}.issubset(set(flux_ts.columns)):
        work = (
            work.assign(reaction_name=flux_ts["reaction_name"].astype(str))
            .sort_values(["time_s", "reaction_name"])
            .drop_duplicates(subset=["time_s", "reaction_name"], keep="last")
        )
    else:
        work = work.sort_values(["time_s", "orig_key"])

    rows = []
    for time_s, grp in work.groupby("time_s", sort=True):
        flux_in = Counter()
        flux_out = Counter()
        flux_in_var = Counter()
        flux_out_var = Counter()
        for _, row in grp.iterrows():
            orig_key = str(row["orig_key"])
            flux_val = float(row["cumulative_abs_flux"])
            flux_std = float(row["cumulative_abs_flux_std"])
            reactant_counter, product_counter = equation_counters(row.get("equation", ""))
            if not reactant_counter and not product_counter:
                reactant_counter, product_counter = reaction_part_maps.get(orig_key, (Counter(), Counter()))
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
    return pd.DataFrame(rows)


def write_table(df, out_path):
    """Write parquet table with pickle fallback."""
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


def to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def safe_json_dumps(value):
    """Serialize nested metadata as compact JSON."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        return str(value)


def clean_text(value):
    """Return stripped text, or empty string for null-like values."""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na", "<na>"}:
        return ""
    return text


def first_nonempty_text(*values):
    """Return first non-empty string value."""
    for value in values:
        text = clean_text(value)
        if text:
            return text
    return ""


def first_phase_state(yaml_payload):
    """Return first phase state mapping from Cantera YAML payload."""
    try:
        phases = (yaml_payload or {}).get("phases", []) or []
        if not phases:
            return {}
        phase0 = phases[0] or {}
        if isinstance(phase0, dict):
            return (phase0.get("state", {}) or {})
    except Exception:
        pass
    return {}


def build_reaction_identity_lookup(rxns):
    """Build lookup for recovering orig_key/rxn_id/rxn_hash from equation states."""
    by_pair = {}
    by_key = {}
    for key, rxn in (rxns or {}).items():
        key_text = str(key)
        rid = str(getattr(rxn, "id", "") or "")
        rhash = str(getattr(rxn, "hash", "") or "")
        reactant_state = normalize_smiles_text(state_smiles(getattr(rxn, "reactant", None)))
        product_state = normalize_smiles_text(state_smiles(getattr(rxn, "product", None)))
        by_key[key_text] = {
            "rxn_id": rid,
            "rxn_hash": rhash,
            "reactant_state": reactant_state,
            "product_state": product_state,
        }
        if reactant_state and product_state:
            by_pair.setdefault((reactant_state, product_state), [])
            by_pair[(reactant_state, product_state)].append((key_text, "forward"))
            by_pair.setdefault((product_state, reactant_state), [])
            by_pair[(product_state, reactant_state)].append((key_text, "reverse"))
    return by_pair, by_key


def downsample_timeseries_df(df, interval_s, group_cols):
    """Downsample time series rows per group while keeping first/last points."""
    if df is None or df.empty:
        return df
    try:
        interval = float(interval_s)
    except Exception:
        return df
    if interval <= 0.0 or "time_s" not in df.columns:
        return df

    work = df.copy()
    work["time_s"] = pd.to_numeric(work["time_s"], errors="coerce")
    work = work.dropna(subset=["time_s"])
    if work.empty:
        return work

    used_groups = [c for c in (group_cols or []) if c in work.columns]
    if not used_groups:
        used_groups = ["_all_rows_group"]
        work["_all_rows_group"] = "all"

    parts = []
    work = work.sort_values(used_groups + ["time_s"], kind="stable")
    for _, grp in work.groupby(used_groups, dropna=False, sort=False):
        if grp.empty:
            continue
        keep_idx = [grp.index[0]]
        next_t = float(grp.iloc[0]["time_s"]) + interval
        for idx, row in grp.iloc[1:].iterrows():
            t = float(row["time_s"])
            if t >= (next_t - 1.0e-15):
                keep_idx.append(idx)
                next_t = t + interval
        if keep_idx[-1] != grp.index[-1]:
            keep_idx.append(grp.index[-1])
        parts.append(work.loc[keep_idx])

    out = pd.concat(parts, ignore_index=True) if parts else work.iloc[0:0].copy()
    if "_all_rows_group" in out.columns:
        out = out.drop(columns=["_all_rows_group"])
    return out.sort_values("time_s", kind="stable").reset_index(drop=True)


def main():
    """Build final per-species table and optional flux timeseries profile table."""
    parser = argparse.ArgumentParser(description="Build compact species datatable.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--subnetwork-pickle", required=True)
    parser.add_argument("--cantera-yaml", required=True)
    parser.add_argument("--to-final-csv", required=True)
    parser.add_argument("--flux-timeseries-csv", required=False, default=None)
    parser.add_argument("--network-pickle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--flux-output", default=None)
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path()
    cfg = load_config(config_path)
    preview_rows = max(1, int(cfg.get("table_preview_rows", 10)))
    output_mode = str(cfg.get("output_mode", "")).strip().lower()
    cleanup_mode = str(cfg.get("cleanup_mode", "clean")).strip().lower()
    datatable_cfg = cfg.get("datatable", {}) or {}
    retained_downsample_seconds = float(datatable_cfg.get("retained_timeseries_downsample_seconds", 0.0) or 0.0)
    if output_mode == "debug":
        apply_retained_downsample = False
    elif output_mode in {"subnetwork", "production"}:
        apply_retained_downsample = retained_downsample_seconds > 0.0
    else:
        apply_retained_downsample = cleanup_mode in {"clean", "debug_keep_failed"} and retained_downsample_seconds > 0.0
    rate_cfg = ((cfg.get("cantera_from_subnetworks", {}) or {}).get("rate", {}) or {})
    barrier_label = str(rate_cfg.get("forward_barrier_label", "egat"))

    with Path(args.subnetwork_pickle).open("rb") as f:
        payload = pickle.load(f)
    meta = payload.get("metadata", {}) or {}
    rxns = payload.get("rxns", {}) or {}

    with Path(args.cantera_yaml).open("r") as f:
        yaml_payload = yaml.safe_load(f) or {}
    phase_state = first_phase_state(yaml_payload)
    cantera_temperature_k = (
        to_float(phase_state.get("T"), default=math.nan)
        if isinstance(phase_state, dict)
        else math.nan
    )
    cantera_pressure_atm = (
        to_float(phase_state.get("P"), default=math.nan)
        if isinstance(phase_state, dict)
        else math.nan
    )
    ymeta = yaml_payload.get("yaks_metadata", {}) or {}
    reaction_map = ymeta.get("reaction_map", {}) or {}
    yaml_reactions = list(iter_yaml_reactions(rxns, reaction_map))
    included_keys = {row["orig_key"] for row in yaml_reactions}
    if not included_keys:
        included_keys = {str(k) for k in (rxns or {}).keys()}
    reverse_enabled_by_key = {}
    for row in yaml_reactions:
        key_text = clean_text((row or {}).get("orig_key", ""))
        if not key_text:
            continue
        reverse_enabled_by_key[key_text] = bool(
            reverse_enabled_by_key.get(key_text, False)
            or bool((row or {}).get("reverse_enabled", False))
        )

    tf = pd.read_csv(args.to_final_csv, low_memory=False)
    flux_ts = pd.DataFrame()
    if args.flux_timeseries_csv:
        flux_ts = pd.read_csv(args.flux_timeseries_csv, low_memory=False)
    if "from_smiles" in tf.columns:
        tf["from_smiles_norm"] = tf["from_smiles"].map(normalize_smiles_text)
    if "to_smiles" in tf.columns:
        tf["to_smiles_norm"] = tf["to_smiles"].map(normalize_smiles_text)
    tf["source_type"] = tf["source_type"] if "source_type" in tf.columns else pd.NA
    tf_to_p_flux_col = "flux_to_final_species" if "flux_to_final_species" in tf.columns else "cumulative_abs_flux"
    tf_to_p_flux_series = (
        pd.to_numeric(tf[tf_to_p_flux_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        if tf_to_p_flux_col in tf.columns
        else pd.Series([0.0] * len(tf), index=tf.index, dtype=float)
    )
    tf["to_p_flux"] = tf_to_p_flux_series

    species_to_p_flux = {}
    species_to_p_flux_class = {}
    if {"from_smiles_norm", "to_p_flux"}.issubset(set(tf.columns)):
        for from_state, flux_value, source_type in zip(
            tf["from_smiles_norm"],
            tf["to_p_flux"],
            tf["source_type"],
        ):
            from_norm = normalize_smiles_text(from_state)
            if not from_norm:
                continue
            src_text = clean_text(source_type).upper()
            row_class = "R->P" if src_text == "R" else ("I->P" if src_text == "I" else "")
            for part in split_smiles(from_norm):
                species_to_p_flux[part] = float(species_to_p_flux.get(part, 0.0)) + float(flux_value)
                if part not in species_to_p_flux_class and row_class:
                    species_to_p_flux_class[part] = row_class

    reagent_smiles = get_meta_smiles(meta, "start")
    product_smiles = get_meta_smiles(meta, "end")
    has_direct_r_to_p_in_yaml, has_reverse_p_to_r_in_yaml = has_direct_r_to_p(
        yaml_reactions,
        reagent_smiles=reagent_smiles,
        product_smiles=product_smiles,
    )

    path_records = meta.get("path_flux_records", []) or []
    terminal_product_states = list(meta.get("terminal_product_states", []) or [])
    if not terminal_product_states:
        terminal_product_states = collect_terminal_product_states(path_records)
    if not terminal_product_states and product_smiles:
        terminal_product_states = [product_smiles]
    path_intermediate_states = list(meta.get("path_intermediate_states", []) or [])
    if not path_intermediate_states:
        path_intermediate_states = collect_on_target_intermediate_states(
            path_records,
            reagent_smiles,
            terminal_product_states,
        )
    direct_path_intermediate_states = list(meta.get("direct_path_intermediate_states", []) or [])
    if not direct_path_intermediate_states:
        direct_path_intermediate_states = collect_direct_on_target_intermediate_states(
            path_records,
            reagent_smiles,
            terminal_product_states,
        )
    state_direction_lookup = {}
    for inter_state in path_intermediate_states:
        has_r_to_i, has_i_to_r, has_i_to_p = direction_flags_for_state(
            yaml_reactions,
            reagent_state=reagent_smiles,
            terminal_states=terminal_product_states,
            current_state=inter_state,
        )
        state_direction_lookup[inter_state] = {
            "has_r_to_i": bool(has_r_to_i),
            "has_i_to_r": bool(has_i_to_r),
            "has_i_to_p": bool(has_i_to_p),
        }

    on_target_intermediate_states = sorted(
        state
        for state, flags in state_direction_lookup.items()
        if bool(flags.get("has_r_to_i")) and bool(flags.get("has_i_to_p"))
    )
    if not on_target_intermediate_states:
        on_target_intermediate_states = list(meta.get("on_target_intermediate_states", []) or [])
    direct_on_target_intermediate_states = sorted(
        set(on_target_intermediate_states) & set(direct_path_intermediate_states)
    )
    if not direct_on_target_intermediate_states:
        direct_on_target_intermediate_states = list(meta.get("direct_on_target_intermediate_states", []) or [])
    if not direct_on_target_intermediate_states:
        direct_on_target_intermediate_states = list(on_target_intermediate_states)
    on_target_intermediates = collect_species_from_states(on_target_intermediate_states)
    direct_on_target_intermediates = collect_species_from_states(direct_on_target_intermediate_states)
    if not on_target_intermediates:
        on_target_intermediates = collect_on_target_intermediates(
            path_records,
            reagent_smiles,
            product_smiles,
        )
    if not direct_on_target_intermediates:
        direct_on_target_intermediates = collect_direct_on_target_intermediates(
            path_records,
            reagent_smiles,
            product_smiles,
        )
    intermediates = set(collect_all_intermediates(rxns, reagent_smiles, product_smiles))
    intermediates.update(on_target_intermediates)
    if "source_type" in tf.columns and "from_smiles_norm" in tf.columns:
        mask = tf["source_type"].astype(str) == "I"
        for state_smi in tf.loc[mask, "from_smiles_norm"].dropna().tolist():
            intermediates.update([part for part in split_smiles(state_smi) if part])
    intermediates = sorted({s for s in intermediates if s and s != reagent_smiles and s != product_smiles})
    coproduct_species = set()
    for state in terminal_product_states:
        parts = split_smiles(normalize_smiles_text(state))
        if product_smiles and product_smiles in parts:
            for part in parts:
                if part and part not in {reagent_smiles, product_smiles}:
                    coproduct_species.add(part)
    coproduct_species = sorted(coproduct_species)
    terminal_state_set = {normalize_smiles_text(s) for s in terminal_product_states if normalize_smiles_text(s)}
    nonterminal_species = set()
    for row in yaml_reactions:
        reactant_state = normalize_smiles_text(row.get("reactant_state"))
        product_state = normalize_smiles_text(row.get("product_state"))
        reactant_parts = set(row.get("reactant_parts", set()) or set())
        product_parts = set(row.get("product_parts", set()) or set())
        if product_state in terminal_state_set:
            nonterminal_species.update(reactant_parts)
        else:
            nonterminal_species.update(reactant_parts)
            nonterminal_species.update(product_parts)
        if reactant_state in terminal_state_set:
            nonterminal_species.update(product_parts)
    intermediates = sorted(
        s for s in intermediates
        if s not in set(coproduct_species) or s in nonterminal_species
    )
    on_target_set = set(on_target_intermediates)
    direct_on_target_set = set(direct_on_target_intermediates)
    on_target_state_set = set(on_target_intermediate_states)
    direct_on_target_state_set = set(direct_on_target_intermediate_states)
    intermediate_states_by_species = {}
    for state in sorted(path_intermediate_states):
        norm_state = normalize_smiles_text(state)
        if not norm_state:
            continue
        for part in split_smiles(norm_state):
            intermediate_states_by_species.setdefault(part, set()).add(norm_state)
    direct_states_by_species = {}
    for state in sorted(direct_on_target_intermediate_states):
        norm_state = normalize_smiles_text(state)
        if not norm_state:
            continue
        for part in split_smiles(norm_state):
            direct_states_by_species.setdefault(part, set()).add(norm_state)
    direct_missing_reverse_count = 0
    for inter_state in direct_on_target_intermediate_states:
        _, has_i_to_r, _ = direction_flags_for_state(
            yaml_reactions,
            reagent_state=reagent_smiles,
            terminal_states=terminal_product_states,
            current_state=inter_state,
        )
        if not has_i_to_r:
            direct_missing_reverse_count += 1

    intermediate_set = set(intermediates)
    coproduct_set = set(coproduct_species)

    def roles_for_species(smi):
        if smi == reagent_smiles:
            return ["reagent"]
        if smi == product_smiles:
            return ["product"]
        roles = []
        if smi in intermediate_set:
            roles.append("intermediate")
        if smi in coproduct_set:
            roles.append("co_product")
        return roles or ["intermediate"]

    total_flux = float(tf["to_p_flux"].sum()) if "to_p_flux" in tf.columns else 0.0
    conc_by_smiles_raw = meta.get("final_species_concentration_by_smiles", {}) or {}
    conc_by_smiles = {normalize_smiles_text(k): v for k, v in conc_by_smiles_raw.items() if normalize_smiles_text(k)}
    cantera_run_meta = meta.get("cantera_run", {}) or {}
    terminal_completion_mode = str(cantera_run_meta.get("terminal_completion_mode", "auto"))
    completion_terminal_species = [
        normalize_smiles_text(s) or str(s).strip()
        for s in (cantera_run_meta.get("completion_terminal_species", []) or [])
    ]
    completion_terminal_species = [s for s in dict.fromkeys(s for s in completion_terminal_species if s)]
    if not completion_terminal_species and product_smiles:
        completion_terminal_species = [product_smiles]
    completion_terminal_concentration = float(
        sum(to_float(conc_by_smiles.get(s, 0.0), default=0.0) for s in completion_terminal_species)
    )
    completion_target_value = to_float(cantera_run_meta.get("completion_target_value", math.nan), default=math.nan)
    completion_ratio = (
        completion_terminal_concentration / completion_target_value
        if completion_target_value and math.isfinite(completion_target_value)
        else math.nan
    )
    adaptive_extension = (cantera_run_meta.get("adaptive_extension", {}) or {})
    adaptive_runs = list(adaptive_extension.get("runs", []) or [])
    completion_reached_any = any(bool((row or {}).get("completion_reached", False)) for row in adaptive_runs)
    completion_reached_all = bool(adaptive_runs) and all(
        bool((row or {}).get("completion_reached", False)) for row in adaptive_runs
    )
    hit_max_time_any = any(bool((row or {}).get("hit_max_time", False)) for row in adaptive_runs)
    hit_max_time_all = bool(adaptive_runs) and all(
        bool((row or {}).get("hit_max_time", False)) for row in adaptive_runs
    )
    final_sim_times = [to_float((row or {}).get("final_sim_time_s", math.nan), default=math.nan) for row in adaptive_runs]
    final_sim_times = [v for v in final_sim_times if math.isfinite(v)]
    extension_counts = [to_float((row or {}).get("extension_count", math.nan), default=math.nan) for row in adaptive_runs]
    extension_counts = [v for v in extension_counts if math.isfinite(v)]
    kinetic_trap_flag = bool(
        math.isfinite(completion_target_value)
        and completion_terminal_concentration < completion_target_value
        and hit_max_time_any
    )
    path_records_json = safe_json_dumps(path_records)
    run_conditions = {
        "cantera_temperature_K": (
            cantera_temperature_k
            if math.isfinite(cantera_temperature_k)
            else to_float(cantera_run_meta.get("temperature_K", math.nan), default=math.nan)
        ),
        "cantera_pressure_atm": (
            cantera_pressure_atm
            if math.isfinite(cantera_pressure_atm)
            else to_float(cantera_run_meta.get("pressure_atm", math.nan), default=math.nan)
        ),
        "cantera_rule": str(cantera_run_meta.get("rule", "")),
        "cantera_time_sim_s": to_float(cantera_run_meta.get("time_sim", math.nan), default=math.nan),
        "cantera_time_step_s": to_float(cantera_run_meta.get("time_step", math.nan), default=math.nan),
        "cantera_uncertainty_enabled": bool(cantera_run_meta.get("uncertainty", False)),
        "cantera_uncertainty_cycles": int(to_float(cantera_run_meta.get("uncertainty_cycles", 0), default=0)),
        "cantera_uncertainty_scale": to_float(cantera_run_meta.get("scale", math.nan), default=math.nan),
        "cantera_fraction_basis": str(cantera_run_meta.get("fraction_basis", "")),
        "completion_tol": to_float(cantera_run_meta.get("completion_tol", math.nan), default=math.nan),
        "completion_target_value": completion_target_value,
        "completion_hold_steps": int(to_float(cantera_run_meta.get("completion_hold_steps", 0), default=0)),
        "completion_dxdt_tol": to_float(cantera_run_meta.get("completion_dxdt_tol", math.nan), default=math.nan),
        "min_completion_time_s": to_float(cantera_run_meta.get("min_completion_time", math.nan), default=math.nan),
        "completion_reached_any": bool(completion_reached_any),
        "completion_reached_all": bool(completion_reached_all),
        "hit_max_time_any": bool(hit_max_time_any),
        "hit_max_time_all": bool(hit_max_time_all),
        "final_sim_time_min_s": min(final_sim_times) if final_sim_times else math.nan,
        "final_sim_time_mean_s": (sum(final_sim_times) / len(final_sim_times)) if final_sim_times else math.nan,
        "final_sim_time_max_s": max(final_sim_times) if final_sim_times else math.nan,
        "extension_count_max": max(extension_counts) if extension_counts else math.nan,
        "extension_count_mean": (sum(extension_counts) / len(extension_counts)) if extension_counts else math.nan,
        "completion_ratio": completion_ratio,
        "kinetic_trap_flag": bool(kinetic_trap_flag),
        "adaptive_extension_runs_json": safe_json_dumps(adaptive_runs),
        "path_record_count": int(len(path_records)),
        "path_flux_records_json": path_records_json,
    }
    reaction_part_maps = build_reaction_part_maps(rxns)
    reaction_state_map = build_reaction_state_map(rxns)
    pair_lookup, reaction_identity_lookup = build_reaction_identity_lookup(rxns)
    flux_to_final_by_key = {
        str(k): to_float(v, default=0.0)
        for k, v in (meta.get("reaction_flux_to_final_by_key", {}) or {}).items()
    }
    flux_to_final_std_by_key = {
        str(k): to_float(v, default=0.0)
        for k, v in (meta.get("reaction_flux_to_final_std_by_key", {}) or {}).items()
    }
    total_flux_to_final_species = float(sum(max(v, 0.0) for v in flux_to_final_by_key.values()))
    species_flux_df = species_flux_timeseries_from_flux_rows(flux_ts, reaction_part_maps)
    hash_lookup = species_hash_lookup(rxns)
    product_id = str((meta.get("end") or {}).get("hash", ""))

    network_path = str(Path(args.network_pickle).resolve())
    network_id = Path(args.network_pickle).stem
    subnetwork_id = Path(args.subnetwork_pickle).stem
    run_id = os.environ.get("JOB_ID", "")
    now = datetime.now(timezone.utc).isoformat()

    print(
        f"Reaction coverage: yaml_included={len(included_keys)} | "
        f"subnetwork_total={len(rxns)} | intermediate_rows={len(intermediates)} | "
        f"on_target_states={len(on_target_intermediate_states)} | terminal_states={len(terminal_product_states)}"
    )

    if kinetic_trap_flag:
        status_flag_value = "kinetic_trap"
    elif math.isfinite(completion_target_value) and completion_terminal_concentration < completion_target_value:
        status_flag_value = "below_completion_target"
    else:
        status_flag_value = "ok"

    state_hashes = state_hash_lookup(rxns)
    reaction_map_by_name = reaction_map if isinstance(reaction_map, dict) else {}
    rxn_id_by_key = {
        clean_text(k): clean_text(v.get("rxn_id", ""))
        for k, v in reaction_identity_lookup.items()
    }
    rxn_hash_by_key = {
        clean_text(k): clean_text(v.get("rxn_hash", ""))
        for k, v in reaction_identity_lookup.items()
    }

    reaction_rows_work = tf.copy()
    required_cols = {
        "reaction_index": pd.NA,
        "reaction_name": pd.NA,
        "equation": pd.NA,
        "orig_key": "",
        "rxn_id": pd.NA,
        "rxn_hash": pd.NA,
        "source_type": pd.NA,
        "from_smiles": pd.NA,
        "to_smiles": pd.NA,
        "flux_to_final_species": 0.0,
        "flux_to_final_species_std": 0.0,
        "cumulative_abs_flux": 0.0,
        "cumulative_abs_flux_std": 0.0,
        "final_rate_of_progress": 0.0,
        "final_rate_of_progress_std": 0.0,
    }
    for col, default_val in required_cols.items():
        if col not in reaction_rows_work.columns:
            reaction_rows_work[col] = default_val

    reaction_rows_work["reaction_name"] = reaction_rows_work["reaction_name"].map(clean_text).replace("", pd.NA)
    reaction_rows_work["orig_key"] = reaction_rows_work["orig_key"].map(clean_text)
    mapped_orig = reaction_rows_work["reaction_name"].map(
        lambda name: clean_text((reaction_map_by_name.get(clean_text(name), {}) or {}).get("orig_key", ""))
    )
    reaction_rows_work["orig_key"] = [
        first_nonempty_text(orig, map_orig)
        for orig, map_orig in zip(reaction_rows_work["orig_key"], mapped_orig)
    ]

    reaction_rows_work["from_smiles"] = reaction_rows_work["from_smiles"].map(normalize_smiles_text)
    reaction_rows_work["to_smiles"] = reaction_rows_work["to_smiles"].map(normalize_smiles_text)
    reaction_rows_work["from_smiles"] = reaction_rows_work["from_smiles"].fillna(
        reaction_rows_work["orig_key"].map(
            lambda k: (reaction_state_map.get(clean_text(k), {}) or {}).get("from_smiles", pd.NA)
        )
    )
    reaction_rows_work["to_smiles"] = reaction_rows_work["to_smiles"].fillna(
        reaction_rows_work["orig_key"].map(
            lambda k: (reaction_state_map.get(clean_text(k), {}) or {}).get("to_smiles", pd.NA)
        )
    )
    reaction_rows_work["from_smiles"] = reaction_rows_work["from_smiles"].map(normalize_smiles_text)
    reaction_rows_work["to_smiles"] = reaction_rows_work["to_smiles"].map(normalize_smiles_text)

    if "equation" in reaction_rows_work.columns:
        parsed_states = reaction_rows_work["equation"].map(equation_states)
        eq_from = parsed_states.map(lambda pair: pair[0])
        eq_to = parsed_states.map(lambda pair: pair[1])
        reaction_rows_work["from_smiles"] = reaction_rows_work["from_smiles"].fillna(eq_from)
        reaction_rows_work["to_smiles"] = reaction_rows_work["to_smiles"].fillna(eq_to)
        reaction_rows_work["from_smiles"] = reaction_rows_work["from_smiles"].map(normalize_smiles_text)
        reaction_rows_work["to_smiles"] = reaction_rows_work["to_smiles"].map(normalize_smiles_text)

    reaction_rows_work["source_type"] = reaction_rows_work["source_type"].map(clean_text).str.upper()
    reaction_rows_work["source_type"] = reaction_rows_work["source_type"].replace("", pd.NA)
    reaction_rows_work["source_type"] = reaction_rows_work["source_type"].fillna(
        reaction_rows_work["from_smiles"].map(
            lambda smi: (
                "R"
                if reagent_smiles and reagent_smiles in split_smiles(normalize_smiles_text(smi))
                else "I"
            )
        )
    )
    reaction_rows_work = reaction_rows_work[
        reaction_rows_work["source_type"].isin({"R", "I"})
    ].copy()

    needs_backfill = reaction_rows_work["orig_key"].map(
        lambda k: (not clean_text(k)) or clean_text(k).startswith("reaction_")
    )
    if needs_backfill.any():
        backfilled = []
        view = reaction_rows_work.loc[needs_backfill, ["orig_key", "reaction_name", "from_smiles", "to_smiles"]]
        for _, row in view.iterrows():
            current_orig = clean_text(row.get("orig_key", ""))
            current_name = clean_text(row.get("reaction_name", ""))
            from_state = normalize_smiles_text(row.get("from_smiles"))
            to_state = normalize_smiles_text(row.get("to_smiles"))
            pair = (from_state, to_state)
            map_orig = clean_text((reaction_map_by_name.get(current_name, {}) or {}).get("orig_key", ""))
            candidates = pair_lookup.get(pair, [])
            chosen = None
            if map_orig and any(key == map_orig for key, _ in candidates):
                chosen = next((item for item in candidates if item[0] == map_orig), None)
            elif current_orig and any(key == current_orig for key, _ in candidates):
                chosen = next((item for item in candidates if item[0] == current_orig), None)
            elif candidates:
                chosen = sorted(candidates, key=lambda item: item[0])[0]
            if chosen is None and map_orig:
                chosen = (map_orig, pd.NA)
            if chosen is None and current_orig and not current_orig.startswith("reaction_"):
                chosen = (current_orig, pd.NA)
            resolved_orig = clean_text(chosen[0]) if chosen else current_orig
            backfilled.append(resolved_orig)
        reaction_rows_work.loc[needs_backfill, "orig_key"] = backfilled

    reaction_rows_work["orig_key"] = reaction_rows_work["orig_key"].map(clean_text)
    reaction_rows_work["reaction_index"] = pd.to_numeric(
        reaction_rows_work["reaction_index"], errors="coerce"
    )
    reaction_rows_work["reaction_index"] = reaction_rows_work["reaction_index"].fillna(
        reaction_rows_work["reaction_name"].map(
            lambda name: (
                parse_reaction_index(clean_text(name)) + 1
                if clean_text(name).startswith("reaction_")
                else math.nan
            )
        )
    )
    reaction_rows_work["reaction_index"] = reaction_rows_work["reaction_index"].astype("Int64")

    for metric_col in [
        "flux_to_final_species",
        "flux_to_final_species_std",
        "cumulative_abs_flux",
        "cumulative_abs_flux_std",
        "final_rate_of_progress",
        "final_rate_of_progress_std",
    ]:
        reaction_rows_work[metric_col] = pd.to_numeric(
            reaction_rows_work[metric_col], errors="coerce"
        ).fillna(0.0)

    mapped_rxn_id = reaction_rows_work["reaction_name"].map(
        lambda name: clean_text((reaction_map_by_name.get(clean_text(name), {}) or {}).get("rxn_id", ""))
    )
    mapped_rxn_hash = reaction_rows_work["reaction_name"].map(
        lambda name: clean_text((reaction_map_by_name.get(clean_text(name), {}) or {}).get("rxn_hash", ""))
    )
    key_rxn_id = reaction_rows_work["orig_key"].map(lambda k: clean_text(rxn_id_by_key.get(clean_text(k), "")))
    key_rxn_hash = reaction_rows_work["orig_key"].map(lambda k: clean_text(rxn_hash_by_key.get(clean_text(k), "")))
    reaction_rows_work["rxn_id"] = [
        first_nonempty_text(a, b, c)
        for a, b, c in zip(
            reaction_rows_work["rxn_id"],
            key_rxn_id,
            mapped_rxn_id,
        )
    ]
    reaction_rows_work["rxn_hash"] = [
        first_nonempty_text(a, b, c)
        for a, b, c in zip(
            reaction_rows_work["rxn_hash"],
            key_rxn_hash,
            mapped_rxn_hash,
        )
    ]
    reaction_rows_work["rxn_id"] = reaction_rows_work["rxn_id"].replace("", pd.NA)
    reaction_rows_work["rxn_hash"] = reaction_rows_work["rxn_hash"].replace("", pd.NA)

    reaction_rows_work["reaction_obj"] = reaction_rows_work["orig_key"].map(
        lambda k: get_reaction_by_key(rxns, clean_text(k))
    )
    reaction_rows_work["reaction_forward_barrier"] = reaction_rows_work["reaction_obj"].map(
        lambda rxn: forward_barrier(rxn, barrier_label) if rxn is not None else pd.NA
    )
    reaction_rows_work["reaction_reverse_barrier"] = reaction_rows_work["reaction_obj"].map(
        lambda rxn: reverse_barrier(rxn, barrier_label) if rxn is not None else pd.NA
    )

    def state_concentration(state_smiles):
        state_norm = normalize_smiles_text(state_smiles)
        parts = split_smiles(state_norm)
        if not parts:
            return math.nan
        return float(sum(to_float(conc_by_smiles.get(part, 0.0), default=0.0) for part in parts))

    def p_to_i_included(current_state):
        current = normalize_smiles_text(current_state)
        if not current:
            return False
        for row in (yaml_reactions or []):
            reactant_state = normalize_smiles_text(row.get("reactant_state"))
            product_state = normalize_smiles_text(row.get("product_state"))
            reverse_enabled = bool(row.get("reverse_enabled", False))
            if reactant_state == current and product_state in terminal_state_set and reverse_enabled:
                return True
            if reactant_state in terminal_state_set and product_state == current:
                return True
        return False

    rows = []
    total_reaction_to_p_flux = float(
        reaction_rows_work["flux_to_final_species"].clip(lower=0.0).sum()
    )
    path_summary_by_terminal = {}
    weighted_paths = []
    total_weighted_path_flux = 0.0
    for rec in (path_records or []):
        terminal_key = clean_text(rec.get("terminal_rxn_key", ""))
        path_flux = max(to_float(rec.get("terminal_flux_to_final_species", 0.0), default=0.0), 0.0)
        multiplicity = max(1, int(to_float(rec.get("multiplicity", 1), default=1)))
        weighted_flux = float(path_flux * multiplicity)
        path_index = int(to_float(rec.get("path_index", 0), default=0))
        path_smiles = clean_text(rec.get("smiles_path", ""))
        steps = sorted((rec.get("steps") or []), key=lambda row: int(row.get("step", 0)))
        path_class = "R->P" if len(steps) <= 1 else "R->I->P"

        total_weighted_path_flux += weighted_flux
        weighted_paths.append(
            {
                "terminal_key": terminal_key,
                "path_index": path_index,
                "path_smiles": path_smiles,
                "path_class": path_class,
                "weighted_flux": weighted_flux,
            }
        )
        summary = path_summary_by_terminal.setdefault(
            terminal_key,
            {
                "indices": [],
                "smiles_paths": [],
                "classes": set(),
                "weighted_flux_total": 0.0,
            },
        )
        summary["indices"].append(path_index)
        summary["smiles_paths"].append(path_smiles)
        summary["classes"].add(path_class)
        summary["weighted_flux_total"] += weighted_flux

    dominant_path_record = (
        max(weighted_paths, key=lambda row: float(row.get("weighted_flux", 0.0)))
        if weighted_paths
        else None
    )
    dominant_path_class_overall = (
        dominant_path_record.get("path_class", pd.NA)
        if dominant_path_record is not None
        else pd.NA
    )
    dominant_path_smiles_overall = (
        dominant_path_record.get("path_smiles", pd.NA)
        if dominant_path_record is not None
        else pd.NA
    )
    dominant_path_terminal_key = (
        dominant_path_record.get("terminal_key", pd.NA)
        if dominant_path_record is not None
        else pd.NA
    )
    dominant_path_weighted_flux = (
        float(dominant_path_record.get("weighted_flux", 0.0))
        if dominant_path_record is not None
        else 0.0
    )
    dominant_terminal_rxn_flux = (
        float(reaction_rows_work["flux_to_final_species"].max())
        if not reaction_rows_work.empty
        else 0.0
    )
    dominant_terminal_path_flux = (
        max((float(v.get("weighted_flux_total", 0.0)) for v in path_summary_by_terminal.values()), default=0.0)
    )

    for _, rxn_row in reaction_rows_work.iterrows():
        source_type = clean_text(rxn_row.get("source_type", "")).upper()
        from_state = normalize_smiles_text(rxn_row.get("from_smiles"))
        to_state = normalize_smiles_text(rxn_row.get("to_smiles"))
        orig_key = clean_text(rxn_row.get("orig_key", ""))
        reverse_enabled_for_row = bool(reverse_enabled_by_key.get(orig_key, False))
        reaction_obj = rxn_row.get("reaction_obj")

        from_hash_candidates = hash_candidates_for_smiles(
            from_state,
            state_hashes=state_hashes,
            species_hashes=hash_lookup,
        )
        to_hash_candidates = hash_candidates_for_smiles(
            to_state,
            state_hashes=state_hashes,
            species_hashes=hash_lookup,
        )

        r_to_i_forward = pd.NA
        r_to_i_reverse = pd.NA
        i_to_p_forward = pd.NA
        i_to_p_reverse = pd.NA
        r_to_p_forward = pd.NA
        r_to_p_reverse = pd.NA
        included_forward_r_to_i = pd.NA
        included_reverse_i_to_r = pd.NA
        included_forward_i_to_p = pd.NA
        included_reverse_p_to_i = pd.NA
        included_forward_r_to_p = bool(source_type == "R")
        included_reverse_p_to_r = bool(source_type == "R" and reverse_enabled_for_row)
        if source_type == "I" and from_state:
            _, r_to_i_rxn = pick_reaction(rxns, reagent_smiles, from_state, preferred_keys=included_keys)
            _, i_to_r_rxn = pick_reaction(rxns, from_state, reagent_smiles, preferred_keys=included_keys)
            r_to_i_forward = forward_barrier(r_to_i_rxn, barrier_label) if r_to_i_rxn is not None else pd.NA
            r_to_i_reverse = reverse_barrier(r_to_i_rxn, barrier_label) if r_to_i_rxn is not None else pd.NA
            i_to_p_forward = forward_barrier(reaction_obj, barrier_label) if reaction_obj is not None else pd.NA
            i_to_p_reverse = reverse_barrier(reaction_obj, barrier_label) if reaction_obj is not None else pd.NA
            state_r_to_i, state_i_to_r, state_i_to_p = direction_flags_for_state(
                yaml_reactions,
                reagent_state=reagent_smiles,
                terminal_states=terminal_product_states,
                current_state=from_state,
            )
            included_forward_r_to_i = bool(state_r_to_i)
            included_reverse_i_to_r = bool(state_i_to_r)
            included_forward_i_to_p = bool(state_i_to_p)
            included_reverse_p_to_i = bool(p_to_i_included(from_state))
        elif source_type == "R":
            r_to_p_forward = forward_barrier(reaction_obj, barrier_label) if reaction_obj is not None else pd.NA
            r_to_p_reverse = reverse_barrier(reaction_obj, barrier_label) if reaction_obj is not None else pd.NA

        row_flux = max(to_float(rxn_row.get("flux_to_final_species", 0.0), default=0.0), 0.0)
        row_fraction = (
            row_flux / total_reaction_to_p_flux
            if total_reaction_to_p_flux > 0.0
            else 0.0
        )
        path_summary = path_summary_by_terminal.get(orig_key, {})
        path_indices = sorted(set(path_summary.get("indices", [])))
        path_smiles_rows = list(
            dict.fromkeys([s for s in path_summary.get("smiles_paths", []) if clean_text(s)])
        )
        terminal_matching_paths = [row for row in weighted_paths if clean_text(row.get("terminal_key")) == orig_key]
        dominant_terminal_path = (
            max(terminal_matching_paths, key=lambda row: float(row.get("weighted_flux", 0.0)))
            if terminal_matching_paths
            else None
        )
        path_classes = sorted(path_summary.get("classes", set()) or [])
        weighted_terminal_flux = float(path_summary.get("weighted_flux_total", 0.0))
        weighted_terminal_fraction = (
            weighted_terminal_flux / total_weighted_path_flux
            if total_weighted_path_flux > 0.0
            else 0.0
        )
        inferred_path_class = (
            "|".join(path_classes)
            if path_classes
            else ("R->P" if source_type == "R" else "R->I->P")
        )
        rows.append(
            {
                "network_id": network_id,
                "subnetwork_id": subnetwork_id,
                "product_id": product_id,
                "source_network_path": network_path,
                "reagent_smiles": reagent_smiles,
                "product_smiles": product_smiles,
                "reaction_index": rxn_row.get("reaction_index", pd.NA),
                "reaction_name": rxn_row.get("reaction_name", pd.NA),
                "reaction_label": f"{from_state or '?'} -> {to_state or '?'}",
                "source_type": source_type,
                "to_p_flux_class_for_row": "R->P" if source_type == "R" else "I->P",
                "orig_key": orig_key or pd.NA,
                "rxn_id": clean_text(rxn_row.get("rxn_id", "")) or pd.NA,
                "rxn_hash": clean_text(rxn_row.get("rxn_hash", "")) or pd.NA,
                "from_smiles": from_state,
                "to_smiles": to_state,
                "from_yarpecule_hash_primary": from_hash_candidates[0] if from_hash_candidates else pd.NA,
                "from_yarpecule_hash_candidates": ";".join(from_hash_candidates) if from_hash_candidates else "",
                "to_yarpecule_hash_primary": to_hash_candidates[0] if to_hash_candidates else pd.NA,
                "to_yarpecule_hash_candidates": ";".join(to_hash_candidates) if to_hash_candidates else "",
                "r_to_i_forward_barrier": r_to_i_forward,
                "r_to_i_reverse_barrier": r_to_i_reverse,
                "i_to_p_forward_barrier": i_to_p_forward,
                "i_to_p_reverse_barrier": i_to_p_reverse,
                "r_to_p_forward_barrier": r_to_p_forward,
                "r_to_p_reverse_barrier": r_to_p_reverse,
                "included_forward_r_to_i_in_yaml": included_forward_r_to_i,
                "included_reverse_i_to_r_in_yaml": included_reverse_i_to_r,
                "included_forward_i_to_p_in_yaml": included_forward_i_to_p,
                "included_reverse_p_to_i_in_yaml": included_reverse_p_to_i,
                "included_forward_r_to_p_in_yaml": included_forward_r_to_p,
                "included_reverse_p_to_r_in_yaml": included_reverse_p_to_r,
                "network_has_any_r_to_p_in_yaml": bool(has_direct_r_to_p_in_yaml),
                "flux_to_final_species_for_row": float(rxn_row.get("flux_to_final_species", 0.0)),
                "flux_to_final_species_std_for_row": float(rxn_row.get("flux_to_final_species_std", 0.0)),
                "cumulative_abs_flux_for_row": float(rxn_row.get("cumulative_abs_flux", 0.0)),
                "cumulative_abs_flux_std_for_row": float(rxn_row.get("cumulative_abs_flux_std", 0.0)),
                "final_rate_of_progress_for_row": float(rxn_row.get("final_rate_of_progress", 0.0)),
                "final_rate_of_progress_std_for_row": float(rxn_row.get("final_rate_of_progress_std", 0.0)),
                "cumulative_flux_into_p_for_row": float(rxn_row.get("flux_to_final_species", 0.0)),
                "to_p_cumulative_flux_for_row": float(rxn_row.get("flux_to_final_species", 0.0)),
                "fraction_of_total_flux_into_p": row_fraction,
                "fraction_of_total_flux_to_final_species_for_row": row_fraction,
                "to_p_fraction_of_subnetwork_flux_for_row": row_fraction,
                "fraction_flux_into_p_label": row_fraction,
                "reaction_path_class": inferred_path_class,
                "terminal_reaction_path_count": int(len(path_indices)),
                "terminal_reaction_unique_path_count": int(len(path_smiles_rows)),
                "terminal_reaction_path_indices": ";".join(str(v) for v in path_indices),
                "terminal_reaction_paths_smiles": " || ".join(path_smiles_rows),
                "dominant_terminal_path_smiles": (
                    dominant_terminal_path.get("path_smiles", pd.NA)
                    if dominant_terminal_path is not None
                    else pd.NA
                ),
                "dominant_terminal_path_class": (
                    dominant_terminal_path.get("path_class", pd.NA)
                    if dominant_terminal_path is not None
                    else pd.NA
                ),
                "terminal_reaction_weighted_path_flux": weighted_terminal_flux,
                "terminal_reaction_weighted_path_fraction": weighted_terminal_fraction,
                "is_dominant_terminal_reaction_by_flux": bool(row_flux >= (dominant_terminal_rxn_flux - 1.0e-15)),
                "is_dominant_terminal_reaction_by_path_flux": bool(
                    weighted_terminal_flux >= (dominant_terminal_path_flux - 1.0e-15)
                ),
                "dominant_path_class_overall": dominant_path_class_overall,
                "dominant_path_smiles_overall": dominant_path_smiles_overall,
                "dominant_path_terminal_rxn_key": dominant_path_terminal_key,
                "dominant_path_weighted_flux_overall": dominant_path_weighted_flux,
                "intermediate_final_concentration": (
                    state_concentration(from_state) if source_type == "I" else math.nan
                ),
                "terminal_completion_concentration": float(completion_terminal_concentration),
                "completion_ratio": (
                    float(completion_ratio) if math.isfinite(to_float(completion_ratio, default=math.nan)) else math.nan
                ),
                "kinetic_trap_flag": bool(kinetic_trap_flag),
                "status_flag": status_flag_value,
                "cantera_temperature_K": run_conditions.get("cantera_temperature_K", math.nan),
                "cantera_pressure_atm": run_conditions.get("cantera_pressure_atm", math.nan),
                "cantera_time_sim_s": run_conditions.get("cantera_time_sim_s", math.nan),
                "cantera_uncertainty_enabled": run_conditions.get("cantera_uncertainty_enabled", pd.NA),
                "cantera_uncertainty_cycles": run_conditions.get("cantera_uncertainty_cycles", pd.NA),
                "min_completion_time_s": run_conditions.get("min_completion_time_s", math.nan),
                "final_sim_time_mean_s": run_conditions.get("final_sim_time_mean_s", math.nan),
                "final_sim_time_max_s": run_conditions.get("final_sim_time_max_s", math.nan),
                "path_record_count": int(len(path_records)),
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df["_class_order"] = out_df["source_type"].map({"I": 0, "R": 1}).fillna(2)
        out_df = out_df.sort_values(
            ["_class_order", "fraction_of_total_flux_into_p", "reaction_index"],
            ascending=[True, False, True],
            kind="stable",
        ).drop(columns=["_class_order"]).reset_index(drop=True)
        keep_cols = [
            "network_id",
            "subnetwork_id",
            "product_id",
            "reagent_smiles",
            "product_smiles",
            "reaction_index",
            "reaction_name",
            "reaction_label",
            "source_type",
            "to_p_flux_class_for_row",
            "orig_key",
            "rxn_id",
            "rxn_hash",
            "from_smiles",
            "to_smiles",
            "from_yarpecule_hash_primary",
            "to_yarpecule_hash_primary",
            "r_to_i_forward_barrier",
            "r_to_i_reverse_barrier",
            "i_to_p_forward_barrier",
            "i_to_p_reverse_barrier",
            "r_to_p_forward_barrier",
            "r_to_p_reverse_barrier",
            "included_forward_r_to_i_in_yaml",
            "included_reverse_i_to_r_in_yaml",
            "included_forward_i_to_p_in_yaml",
            "included_reverse_p_to_i_in_yaml",
            "included_forward_r_to_p_in_yaml",
            "included_reverse_p_to_r_in_yaml",
            "network_has_any_r_to_p_in_yaml",
            "flux_to_final_species_for_row",
            "flux_to_final_species_std_for_row",
            "final_rate_of_progress_for_row",
            "final_rate_of_progress_std_for_row",
            "fraction_flux_into_p_label",
            "reaction_path_class",
            "terminal_reaction_path_count",
            "terminal_reaction_unique_path_count",
            "terminal_reaction_path_indices",
            "terminal_reaction_paths_smiles",
            "dominant_terminal_path_smiles",
            "dominant_terminal_path_class",
            "terminal_reaction_weighted_path_flux",
            "terminal_reaction_weighted_path_fraction",
            "is_dominant_terminal_reaction_by_flux",
            "is_dominant_terminal_reaction_by_path_flux",
            "dominant_path_class_overall",
            "dominant_path_smiles_overall",
            "dominant_path_terminal_rxn_key",
            "intermediate_final_concentration",
            "terminal_completion_concentration",
            "completion_ratio",
            "kinetic_trap_flag",
            "status_flag",
            "cantera_temperature_K",
            "cantera_pressure_atm",
            "cantera_time_sim_s",
            "cantera_uncertainty_cycles",
            "min_completion_time_s",
            "final_sim_time_mean_s",
            "final_sim_time_max_s",
            "path_record_count",
        ]
        out_df = out_df[[c for c in keep_cols if c in out_df.columns]]
    written = write_table(out_df, out)
    print(f"Wrote product datatable: {written}")
    print(out_df.head(preview_rows).to_string(index=False))

    if args.flux_output:
        tf_cols = [c for c in ["orig_key", "source_type", "from_smiles_norm", "to_smiles_norm"] if c in tf.columns]
        tf_small = tf[tf_cols].copy() if tf_cols else pd.DataFrame()
        flux_ts_work = flux_ts.copy()
        reaction_df = flux_ts_work.copy()
        reaction_df["reaction_name"] = (
            reaction_df["reaction_name"].astype(str)
            if "reaction_name" in reaction_df.columns
            else pd.Series([pd.NA] * len(reaction_df), index=reaction_df.index)
        )
        reaction_df["orig_key"] = (
            reaction_df["orig_key"].map(clean_text)
            if "orig_key" in reaction_df.columns
            else pd.Series([""] * len(reaction_df), index=reaction_df.index)
        )
        reaction_df["source_type"] = pd.NA
        reaction_df["reaction_direction"] = pd.NA

        reaction_map_by_name = reaction_map if isinstance(reaction_map, dict) else {}
        rxn_id_by_key = {
            clean_text(k): clean_text(v.get("rxn_id", ""))
            for k, v in reaction_identity_lookup.items()
        }
        rxn_hash_by_key = {
            clean_text(k): clean_text(v.get("rxn_hash", ""))
            for k, v in reaction_identity_lookup.items()
        }
        mapped_orig = reaction_df["reaction_name"].map(
            lambda name: clean_text((reaction_map_by_name.get(clean_text(name), {}) or {}).get("orig_key", ""))
        )
        reaction_df["orig_key"] = [
            first_nonempty_text(orig, map_orig)
            for orig, map_orig in zip(reaction_df["orig_key"], mapped_orig)
        ]

        reaction_df["from_smiles"] = reaction_df["orig_key"].map(
            lambda k: (reaction_state_map.get(clean_text(k), {}) or {}).get("from_smiles", pd.NA)
        )
        reaction_df["to_smiles"] = reaction_df["orig_key"].map(
            lambda k: (reaction_state_map.get(clean_text(k), {}) or {}).get("to_smiles", pd.NA)
        )
        if "equation" in reaction_df.columns:
            parsed_states = reaction_df["equation"].map(equation_states)
            reaction_df["from_smiles_eq"] = parsed_states.map(lambda pair: pair[0])
            reaction_df["to_smiles_eq"] = parsed_states.map(lambda pair: pair[1])
            reaction_df["from_smiles"] = reaction_df["from_smiles"].fillna(reaction_df["from_smiles_eq"])
            reaction_df["to_smiles"] = reaction_df["to_smiles"].fillna(reaction_df["to_smiles_eq"])
            reaction_df = reaction_df.drop(columns=["from_smiles_eq", "to_smiles_eq"])

        reaction_df["from_smiles"] = reaction_df["from_smiles"].map(normalize_smiles_text)
        reaction_df["to_smiles"] = reaction_df["to_smiles"].map(normalize_smiles_text)
        needs_backfill = reaction_df["orig_key"].map(
            lambda k: (not clean_text(k)) or clean_text(k).startswith("reaction_")
        )
        if needs_backfill.any():
            backfilled = []
            backfilled_direction = []
            view = reaction_df.loc[needs_backfill, ["orig_key", "reaction_name", "from_smiles", "to_smiles"]]
            for _, row in view.iterrows():
                current_orig = clean_text(row.get("orig_key", ""))
                current_name = clean_text(row.get("reaction_name", ""))
                from_state = normalize_smiles_text(row.get("from_smiles"))
                to_state = normalize_smiles_text(row.get("to_smiles"))
                pair = (from_state, to_state)
                map_orig = clean_text((reaction_map_by_name.get(current_name, {}) or {}).get("orig_key", ""))
                candidates = pair_lookup.get(pair, [])
                chosen = None
                if map_orig and any(key == map_orig for key, _ in candidates):
                    chosen = next((item for item in candidates if item[0] == map_orig), None)
                elif current_orig and any(key == current_orig for key, _ in candidates):
                    chosen = next((item for item in candidates if item[0] == current_orig), None)
                elif candidates:
                    chosen = sorted(candidates, key=lambda item: item[0])[0]
                if chosen is None and map_orig:
                    chosen = (map_orig, pd.NA)
                if chosen is None and current_orig and not current_orig.startswith("reaction_"):
                    chosen = (current_orig, pd.NA)
                resolved_orig = clean_text(chosen[0]) if chosen else current_orig
                resolved_dir = chosen[1] if chosen else pd.NA
                backfilled.append(resolved_orig)
                backfilled_direction.append(resolved_dir)
            reaction_df.loc[needs_backfill, "orig_key"] = backfilled
            reaction_df.loc[needs_backfill, "reaction_direction"] = backfilled_direction

        reaction_df["orig_key"] = reaction_df["orig_key"].map(clean_text)
        reaction_df["from_smiles"] = reaction_df["from_smiles"].fillna(
            reaction_df["orig_key"].map(
                lambda k: (reaction_state_map.get(clean_text(k), {}) or {}).get("from_smiles", pd.NA)
            )
        )
        reaction_df["to_smiles"] = reaction_df["to_smiles"].fillna(
            reaction_df["orig_key"].map(
                lambda k: (reaction_state_map.get(clean_text(k), {}) or {}).get("to_smiles", pd.NA)
            )
        )
        reaction_df["from_smiles"] = reaction_df["from_smiles"].map(normalize_smiles_text)
        reaction_df["to_smiles"] = reaction_df["to_smiles"].map(normalize_smiles_text)

        for col in ("rxn_id", "rxn_hash"):
            if col not in reaction_df.columns:
                reaction_df[col] = pd.NA
            reaction_df[col] = reaction_df[col].map(clean_text)

        mapped_rxn_id = reaction_df["reaction_name"].map(
            lambda name: clean_text((reaction_map_by_name.get(clean_text(name), {}) or {}).get("rxn_id", ""))
        )
        mapped_rxn_hash = reaction_df["reaction_name"].map(
            lambda name: clean_text((reaction_map_by_name.get(clean_text(name), {}) or {}).get("rxn_hash", ""))
        )
        mapped_merged_rxn_id = reaction_df["reaction_name"].map(
            lambda name: first_nonempty_text(
                *((reaction_map_by_name.get(clean_text(name), {}) or {}).get("merged_rxn_ids", []) or [])
            )
        )
        mapped_merged_rxn_hash = reaction_df["reaction_name"].map(
            lambda name: first_nonempty_text(
                *((reaction_map_by_name.get(clean_text(name), {}) or {}).get("merged_rxn_hashes", []) or [])
            )
        )
        key_rxn_id = reaction_df["orig_key"].map(lambda k: clean_text(rxn_id_by_key.get(clean_text(k), "")))
        key_rxn_hash = reaction_df["orig_key"].map(lambda k: clean_text(rxn_hash_by_key.get(clean_text(k), "")))
        reaction_df["rxn_id"] = [
            first_nonempty_text(a, b, c, d)
            for a, b, c, d in zip(
                reaction_df["rxn_id"],
                key_rxn_id,
                mapped_rxn_id,
                mapped_merged_rxn_id,
            )
        ]
        reaction_df["rxn_hash"] = [
            first_nonempty_text(a, b, c, d)
            for a, b, c, d in zip(
                reaction_df["rxn_hash"],
                key_rxn_hash,
                mapped_rxn_hash,
                mapped_merged_rxn_hash,
            )
        ]
        reaction_df["rxn_id"] = reaction_df["rxn_id"].replace("", pd.NA)
        reaction_df["rxn_hash"] = reaction_df["rxn_hash"].replace("", pd.NA)
        missing_orig_count = int(reaction_df["orig_key"].map(clean_text).eq("").sum())
        missing_rxn_id_count = int(reaction_df["rxn_id"].isna().sum())
        missing_rxn_hash_count = int(reaction_df["rxn_hash"].isna().sum())
        print(
            "Random flux reaction identity coverage:",
            f"rows={len(reaction_df)}",
            f"missing_orig_key={missing_orig_count}",
            f"missing_rxn_id={missing_rxn_id_count}",
            f"missing_rxn_hash={missing_rxn_hash_count}",
        )

        def infer_reaction_direction(row):
            existing = clean_text(row.get("reaction_direction", ""))
            if existing:
                return existing
            key = clean_text(row.get("orig_key", ""))
            info = reaction_state_map.get(key, {}) or {}
            key_from = normalize_smiles_text(info.get("from_smiles"))
            key_to = normalize_smiles_text(info.get("to_smiles"))
            row_from = normalize_smiles_text(row.get("from_smiles"))
            row_to = normalize_smiles_text(row.get("to_smiles"))
            if key and key_from and key_to and row_from and row_to:
                if row_from == key_from and row_to == key_to:
                    return "forward"
                if row_from == key_to and row_to == key_from:
                    return "reverse"
            return pd.NA

        reaction_df["reaction_direction"] = reaction_df.apply(infer_reaction_direction, axis=1)

        if not tf_small.empty and "orig_key" in reaction_df.columns:
            tf_small = tf_small.rename(
                columns={
                    "from_smiles_norm": "from_smiles_norm_tf",
                    "to_smiles_norm": "to_smiles_norm_tf",
                }
            )
            tf_small["orig_key"] = tf_small["orig_key"].map(clean_text)
            source_lookup = tf_small[tf_small["orig_key"] != ""]
            if not source_lookup.empty and "source_type" in source_lookup.columns:
                source_lookup = source_lookup.drop_duplicates(subset=["orig_key"], keep="last")
                source_lookup = source_lookup.set_index("orig_key")["source_type"].to_dict()
                reaction_df["source_type"] = reaction_df["orig_key"].map(source_lookup)
        reaction_df["source_type"] = reaction_df["source_type"].map(clean_text).replace("", pd.NA)
        reaction_df["source_type"] = reaction_df["source_type"].fillna(
            reaction_df["from_smiles"].map(
                lambda smi: (
                    "R"
                    if reagent_smiles and reagent_smiles in split_smiles(normalize_smiles_text(smi))
                    else "I"
                )
            )
        )

        def reaction_intermediate_state(row):
            from_state = normalize_smiles_text(row.get("from_smiles"))
            to_state = normalize_smiles_text(row.get("to_smiles"))
            if from_state and from_state in on_target_state_set:
                return from_state
            if to_state and to_state in on_target_state_set:
                return to_state
            if (
                clean_text(row.get("source_type", "")) == "I"
                and from_state
                and from_state != reagent_smiles
                and from_state not in terminal_state_set
            ):
                return from_state
            if to_state and to_state != reagent_smiles and to_state not in terminal_state_set:
                return to_state
            return pd.NA

        reaction_df["row_kind"] = "reaction_flux"
        reaction_df["species_smiles"] = pd.NA
        reaction_df["cumulative_in_flux"] = pd.NA
        reaction_df["cumulative_out_flux"] = pd.NA
        reaction_df["cumulative_in_flux_std"] = pd.NA
        reaction_df["cumulative_out_flux_std"] = pd.NA
        reaction_df["is_on_target"] = pd.NA
        reaction_df["is_direct_on_target"] = pd.NA
        reaction_df["row_role"] = pd.NA
        reaction_df["reagent_smiles"] = reagent_smiles
        reaction_df["intermediate_smiles"] = reaction_df["from_smiles"].where(reaction_df["source_type"] == "I", pd.NA)
        reaction_df["intermediate_state_smiles"] = reaction_df.apply(reaction_intermediate_state, axis=1)
        reaction_df["intermediate_state_smiles_all"] = reaction_df["intermediate_state_smiles"]
        reaction_df["direct_intermediate_state_smiles_all"] = reaction_df["intermediate_state_smiles"].map(
            lambda s: s if normalize_smiles_text(s) in direct_on_target_state_set else pd.NA
        )
        reaction_df["product_smiles"] = product_smiles
        reaction_df["is_target_product"] = False
        reaction_df["included_direct_r_to_p_in_yaml"] = bool(has_direct_r_to_p_in_yaml)
        reaction_df["included_reverse_p_to_r_in_yaml"] = bool(has_reverse_p_to_r_in_yaml)
        reaction_df["network_id"] = network_id
        reaction_df["subnetwork_id"] = subnetwork_id
        reaction_df["run_id"] = run_id
        reaction_df["created_at_utc"] = now
        reaction_df["product_id"] = product_id
        reaction_df["reaction_label"] = (
            reaction_df["from_smiles"].fillna("?")
            + " -> "
            + reaction_df["to_smiles"].fillna(product_smiles or "?")
        )
        reaction_df["source_network_path"] = network_path
        reaction_df["terminal_product_states"] = ";".join(sorted(terminal_product_states))
        reaction_df["terminal_completion_mode"] = terminal_completion_mode
        reaction_df["completion_terminal_species"] = ";".join(completion_terminal_species)
        reaction_df["completion_terminal_species_count"] = int(len(completion_terminal_species))
        reaction_df["completion_terminal_concentration"] = completion_terminal_concentration
        reaction_df["direct_on_target_intermediate_count"] = int(len(direct_on_target_set))
        reaction_df["on_target_intermediate_state_count"] = int(len(on_target_state_set))
        reaction_df["direct_on_target_intermediate_state_count"] = int(len(direct_on_target_state_set))
        reaction_df["missing_reverse_into_reagent_on_direct_count"] = int(direct_missing_reverse_count)
        reaction_df["flux_to_final_species"] = reaction_df["orig_key"].map(
            lambda k: to_float(flux_to_final_by_key.get(clean_text(k), 0.0), default=0.0)
        )
        reaction_df["flux_to_final_species_std"] = reaction_df["orig_key"].map(
            lambda k: to_float(flux_to_final_std_by_key.get(clean_text(k), 0.0), default=0.0)
        )
        reaction_df["fraction_of_total_flux_to_final_species"] = reaction_df["flux_to_final_species"].map(
            lambda v: (float(v) / total_flux_to_final_species) if total_flux_to_final_species > 0.0 else 0.0
        )
        reaction_df["to_p_flux_class_for_row"] = reaction_df["source_type"].map(
            lambda s: "R->P" if clean_text(s).upper() == "R" else ("I->P" if clean_text(s).upper() == "I" else "")
        )
        reaction_df["to_p_cumulative_flux_for_row"] = reaction_df["flux_to_final_species"]
        reaction_df["to_p_fraction_of_subnetwork_flux_for_row"] = reaction_df["to_p_cumulative_flux_for_row"].map(
            lambda v: (float(v) / total_flux) if total_flux > 0.0 else 0.0
        )
        reaction_df["status_flag"] = status_flag_value

        def species_row_to_p_flux(smi, role_name):
            role_text = clean_text(role_name)
            if role_text in {"reagent", "product"}:
                return float(total_flux)
            return float(species_to_p_flux.get(clean_text(smi), 0.0))

        def species_row_to_p_class(smi, role_name):
            role_text = clean_text(role_name)
            if role_text == "reagent":
                return "R->P"
            if role_text == "intermediate":
                return "I->P"
            if role_text == "co_product":
                return species_to_p_flux_class.get(clean_text(smi), "I->P")
            return species_to_p_flux_class.get(clean_text(smi), "")

        species_random_df = species_flux_df.copy()
        if not species_random_df.empty:
            species_random_df["row_kind"] = "species_flux"
            species_random_df["orig_key"] = pd.NA
            species_random_df["source_type"] = pd.NA
            species_random_df["from_smiles"] = pd.NA
            species_random_df["to_smiles"] = pd.NA
            species_random_df["reaction_index"] = pd.NA
            species_random_df["reaction_name"] = pd.NA
            species_random_df["equation"] = pd.NA
            species_random_df["rxn_id"] = pd.NA
            species_random_df["rxn_hash"] = pd.NA
            species_random_df["reaction_direction"] = pd.NA
            species_random_df["cumulative_abs_flux"] = pd.NA
            species_random_df["cumulative_abs_flux_std"] = pd.NA
            species_random_df["reaction_label"] = pd.NA
            species_random_df["reagent_smiles"] = reagent_smiles
            species_random_df["intermediate_smiles"] = pd.NA
            species_random_df["product_smiles"] = product_smiles
            species_random_df["is_target_product"] = species_random_df["species_smiles"].map(lambda s: bool(s == product_smiles))
            species_random_df["included_direct_r_to_p_in_yaml"] = bool(has_direct_r_to_p_in_yaml)
            species_random_df["included_reverse_p_to_r_in_yaml"] = bool(has_reverse_p_to_r_in_yaml)
            species_random_df["network_id"] = network_id
            species_random_df["subnetwork_id"] = subnetwork_id
            species_random_df["run_id"] = run_id
            species_random_df["created_at_utc"] = now
            species_random_df["product_id"] = product_id
            species_random_df["source_network_path"] = network_path
            species_random_df["terminal_product_states"] = ";".join(sorted(terminal_product_states))
            species_random_df["terminal_completion_mode"] = terminal_completion_mode
            species_random_df["completion_terminal_species"] = ";".join(completion_terminal_species)
            species_random_df["completion_terminal_species_count"] = int(len(completion_terminal_species))
            species_random_df["completion_terminal_concentration"] = completion_terminal_concentration
            species_random_df["direct_on_target_intermediate_count"] = int(len(direct_on_target_set))
            species_random_df["on_target_intermediate_state_count"] = int(len(on_target_state_set))
            species_random_df["direct_on_target_intermediate_state_count"] = int(len(direct_on_target_state_set))
            species_random_df["missing_reverse_into_reagent_on_direct_count"] = int(direct_missing_reverse_count)
            species_random_df["is_on_target"] = species_random_df["species_smiles"].map(
                lambda s: bool(s in on_target_set or s == reagent_smiles or s == product_smiles)
            )
            species_random_df["is_direct_on_target"] = species_random_df["species_smiles"].map(
                lambda s: bool(s in direct_on_target_set)
            )
            species_random_df["row_role_list"] = species_random_df["species_smiles"].map(roles_for_species)
            species_random_df = species_random_df.explode("row_role_list", ignore_index=True)
            species_random_df["row_role"] = species_random_df["row_role_list"]
            species_random_df = species_random_df.drop(columns=["row_role_list"])
            species_random_df["intermediate_state_smiles"] = species_random_df["species_smiles"].map(
                lambda s: next(iter(sorted(intermediate_states_by_species.get(s, set()))), pd.NA)
            )
            species_random_df["intermediate_state_smiles_all"] = species_random_df["species_smiles"].map(
                lambda s: ";".join(sorted(intermediate_states_by_species.get(s, set())))
            )
            species_random_df["direct_intermediate_state_smiles_all"] = species_random_df["species_smiles"].map(
                lambda s: ";".join(sorted(direct_states_by_species.get(s, set())))
            )
            species_random_df["flux_to_final_species"] = pd.NA
            species_random_df["flux_to_final_species_std"] = pd.NA
            species_random_df["fraction_of_total_flux_to_final_species"] = pd.NA
            species_random_df["to_p_cumulative_flux_for_row"] = [
                species_row_to_p_flux(smi, role)
                for smi, role in zip(species_random_df["species_smiles"], species_random_df["row_role"])
            ]
            species_random_df["to_p_flux_class_for_row"] = [
                species_row_to_p_class(smi, role)
                for smi, role in zip(species_random_df["species_smiles"], species_random_df["row_role"])
            ]
            species_random_df["to_p_fraction_of_subnetwork_flux_for_row"] = species_random_df["to_p_cumulative_flux_for_row"].map(
                lambda v: (float(v) / total_flux) if total_flux > 0.0 else 0.0
            )
            species_random_df["status_flag"] = status_flag_value

        concentration_rows = []
        ts_grid = list(meta.get("species_time_grid_s", []) or [])
        ts_mean = dict(meta.get("species_concentration_timeseries_by_smiles", {}) or {})
        ts_std = dict(meta.get("species_concentration_timeseries_std_by_smiles", {}) or {})
        for species_name, series in ts_mean.items():
            smi = normalize_smiles_text(species_name) or str(species_name)
            values = list(series or [])
            std_values = list(ts_std.get(species_name, []) or [])
            for i, t in enumerate(ts_grid):
                if i >= len(values):
                    break
                for role_name in roles_for_species(smi):
                    concentration_rows.append(
                        {
                            "row_kind": "species_concentration",
                            "time_s": float(t),
                            "species_smiles": smi,
                            "concentration_x": float(values[i]),
                            "concentration_x_std": float(std_values[i]) if i < len(std_values) else 0.0,
                            "is_on_target": bool(smi in on_target_set or smi == reagent_smiles or smi == product_smiles or smi in coproduct_set),
                            "is_direct_on_target": bool(smi in direct_on_target_set),
                            "row_role": role_name,
                            "reagent_smiles": reagent_smiles,
                            "intermediate_smiles": pd.NA,
                            "intermediate_state_smiles": next(iter(sorted(intermediate_states_by_species.get(smi, set()))), pd.NA),
                            "intermediate_state_smiles_all": ";".join(sorted(intermediate_states_by_species.get(smi, set()))),
                            "direct_intermediate_state_smiles_all": ";".join(sorted(direct_states_by_species.get(smi, set()))),
                            "product_smiles": product_smiles,
                            "is_target_product": bool(smi == product_smiles),
                            "network_id": network_id,
                            "subnetwork_id": subnetwork_id,
                            "run_id": run_id,
                            "created_at_utc": now,
                            "product_id": product_id,
                            "source_network_path": network_path,
                            "terminal_product_states": ";".join(sorted(terminal_product_states)),
                            "terminal_completion_mode": terminal_completion_mode,
                            "completion_terminal_species": ";".join(completion_terminal_species),
                            "completion_terminal_species_count": int(len(completion_terminal_species)),
                            "completion_terminal_concentration": completion_terminal_concentration,
                            "included_direct_r_to_p_in_yaml": bool(has_direct_r_to_p_in_yaml),
                            "included_reverse_p_to_r_in_yaml": bool(has_reverse_p_to_r_in_yaml),
                            "orig_key": pd.NA,
                            "source_type": pd.NA,
                            "from_smiles": pd.NA,
                            "to_smiles": pd.NA,
                            "reaction_index": pd.NA,
                            "reaction_name": pd.NA,
                            "equation": pd.NA,
                            "rxn_id": pd.NA,
                            "rxn_hash": pd.NA,
                            "reaction_direction": pd.NA,
                            "cumulative_abs_flux": pd.NA,
                            "cumulative_abs_flux_std": pd.NA,
                            "cumulative_in_flux": pd.NA,
                            "cumulative_out_flux": pd.NA,
                            "cumulative_in_flux_std": pd.NA,
                            "cumulative_out_flux_std": pd.NA,
                            "reaction_label": pd.NA,
                            "flux_to_final_species": pd.NA,
                            "flux_to_final_species_std": pd.NA,
                            "fraction_of_total_flux_to_final_species": pd.NA,
                            "to_p_cumulative_flux_for_row": species_row_to_p_flux(smi, role_name),
                            "to_p_flux_class_for_row": species_row_to_p_class(smi, role_name),
                            "to_p_fraction_of_subnetwork_flux_for_row": (
                                (species_row_to_p_flux(smi, role_name) / total_flux) if total_flux > 0.0 else 0.0
                            ),
                            "status_flag": status_flag_value,
                            "direct_on_target_intermediate_count": int(len(direct_on_target_set)),
                            "on_target_intermediate_state_count": int(len(on_target_state_set)),
                            "direct_on_target_intermediate_state_count": int(len(direct_on_target_state_set)),
                            "missing_reverse_into_reagent_on_direct_count": int(direct_missing_reverse_count),
                        }
                    )
        concentration_df = pd.DataFrame(concentration_rows)

        path_rows = []
        path_flux_total = float(
            sum(
                max(to_float(rec.get("terminal_flux_to_final_species", 0.0), default=0.0), 0.0)
                * max(1, int(to_float(rec.get("multiplicity", 1), default=1)))
                for rec in path_records
            )
        )
        for rec in path_records:
            terminal_rxn_key = clean_text(rec.get("terminal_rxn_key", ""))
            terminal_flux = to_float(rec.get("terminal_flux_to_final_species", 0.0), default=0.0)
            multiplicity = max(1, int(to_float(rec.get("multiplicity", 1), default=1)))
            weighted_flux = terminal_flux * multiplicity
            path_rows.append(
                {
                    "row_kind": "path_flux",
                    "time_s": pd.NA,
                    "species_smiles": pd.NA,
                    "row_role": "path",
                    "path_index": int(to_float(rec.get("path_index", 0), default=0)),
                    "path_multiplicity": int(multiplicity),
                    "path_smiles": rec.get("smiles_path", ""),
                    "path_terminal_rxn_key": terminal_rxn_key or pd.NA,
                    "path_steps_json": safe_json_dumps(rec.get("steps", [])),
                    "path_terminal_flux_to_final_species": float(terminal_flux),
                    "path_weighted_flux_to_final_species": float(weighted_flux),
                    "fraction_of_total_flux_to_final_species": (
                        float(weighted_flux) / path_flux_total if path_flux_total > 0.0 else 0.0
                    ),
                    "orig_key": terminal_rxn_key or pd.NA,
                    "rxn_id": clean_text(rxn_id_by_key.get(terminal_rxn_key, "")) or pd.NA,
                    "rxn_hash": clean_text(rxn_hash_by_key.get(terminal_rxn_key, "")) or pd.NA,
                    "reaction_name": pd.NA,
                    "reaction_index": pd.NA,
                    "equation": pd.NA,
                    "reaction_direction": pd.NA,
                    "source_type": pd.NA,
                    "from_smiles": pd.NA,
                    "to_smiles": pd.NA,
                    "reagent_smiles": reagent_smiles,
                    "intermediate_smiles": pd.NA,
                    "intermediate_state_smiles": pd.NA,
                    "intermediate_state_smiles_all": pd.NA,
                    "direct_intermediate_state_smiles_all": pd.NA,
                    "product_smiles": product_smiles,
                    "is_target_product": False,
                    "network_id": network_id,
                    "subnetwork_id": subnetwork_id,
                    "run_id": run_id,
                    "created_at_utc": now,
                    "product_id": product_id,
                    "source_network_path": network_path,
                    "is_on_target": True,
                    "is_direct_on_target": True,
                    "reaction_label": rec.get("smiles_path", ""),
                    "cumulative_abs_flux": pd.NA,
                    "cumulative_abs_flux_std": pd.NA,
                    "cumulative_in_flux": pd.NA,
                    "cumulative_out_flux": pd.NA,
                    "cumulative_in_flux_std": pd.NA,
                    "cumulative_out_flux_std": pd.NA,
                    "flux_to_final_species": float(terminal_flux),
                    "flux_to_final_species_std": pd.NA,
                    "to_p_cumulative_flux_for_row": float(weighted_flux),
                    "to_p_flux_class_for_row": "I->P",
                    "to_p_fraction_of_subnetwork_flux_for_row": (
                        float(weighted_flux) / total_flux if total_flux > 0.0 else 0.0
                    ),
                    "included_direct_r_to_p_in_yaml": bool(has_direct_r_to_p_in_yaml),
                    "included_reverse_p_to_r_in_yaml": bool(has_reverse_p_to_r_in_yaml),
                    "terminal_product_states": ";".join(sorted(terminal_product_states)),
                    "terminal_completion_mode": terminal_completion_mode,
                    "completion_terminal_species": ";".join(completion_terminal_species),
                    "completion_terminal_species_count": int(len(completion_terminal_species)),
                    "completion_terminal_concentration": completion_terminal_concentration,
                    "status_flag": status_flag_value,
                    "direct_on_target_intermediate_count": int(len(direct_on_target_set)),
                    "on_target_intermediate_state_count": int(len(on_target_state_set)),
                    "direct_on_target_intermediate_state_count": int(len(direct_on_target_state_set)),
                    "missing_reverse_into_reagent_on_direct_count": int(direct_missing_reverse_count),
                }
            )
        path_df = pd.DataFrame(path_rows)

        run_conditions_random = dict(run_conditions)
        run_conditions_random.pop("path_flux_records_json", None)
        for col, value in run_conditions_random.items():
            reaction_df[col] = value
            if not species_random_df.empty:
                species_random_df[col] = value
            if not concentration_df.empty:
                concentration_df[col] = value
            if not path_df.empty:
                path_df[col] = value

        if apply_retained_downsample:
            before_counts = (len(reaction_df), len(species_random_df), len(concentration_df), len(path_df))
            reaction_df = downsample_timeseries_df(
                reaction_df,
                retained_downsample_seconds,
                group_cols=["reaction_label"],
            )
            species_random_df = downsample_timeseries_df(
                species_random_df,
                retained_downsample_seconds,
                group_cols=["species_smiles", "row_role"],
            )
            concentration_df = downsample_timeseries_df(
                concentration_df,
                retained_downsample_seconds,
                group_cols=["species_smiles", "row_role"],
            )
            after_counts = (len(reaction_df), len(species_random_df), len(concentration_df), len(path_df))
            print(
                "Applied retained-output downsampling:",
                f"dt={retained_downsample_seconds}s",
                f"reaction_flux={before_counts[0]}->{after_counts[0]}",
                f"species_flux={before_counts[1]}->{after_counts[1]}",
                f"species_concentration={before_counts[2]}->{after_counts[2]}",
                f"path_flux={before_counts[3]}->{after_counts[3]}",
            )

        pieces = [df for df in [reaction_df, species_random_df, concentration_df, path_df] if not df.empty]
        random_df = pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame()
        Path(args.flux_output).parent.mkdir(parents=True, exist_ok=True)
        written_random = write_table(random_df, args.flux_output)
        print(f"Wrote flux timeseries profile: {written_random}")
        print(random_df.head(preview_rows).to_string(index=False))


if __name__ == "__main__":
    main()
