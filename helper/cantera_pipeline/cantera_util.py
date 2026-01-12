from __future__ import annotations
import io
from pathlib import Path
from typing import Dict
from decimal import Decimal, ROUND_HALF_UP
from yarp.reaction import *
from collections import Counter
from rdkit import Chem
from rdkit import RDLogger
import pickle as pkl
RDLogger.DisableLog("rdApp.*")




def elements_from_smiles(smiles):
    """Return elemental composition from a SMILES using RDKit."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"Bad SMILES: {smiles}")
    m = Chem.AddHs(m)
    counts: Dict[str, int] = {}
    for a in m.GetAtoms():
        sym = a.GetSymbol()
        counts[sym] = counts.get(sym, 0) + 1
    return counts

def quote(s):
    """Single-quote a string for YAML, doubling any embedded single quotes."""
    return "'" + s.replace("'", "''") + "'"

def eq_quote(s):
    """Single-quote the whole equation; double any embedded single quotes for YAML"""
    return "'" + s.replace("'", "''") + "'"

def split_species(state):
    """Accept a SMILES string or sequence and return a list of species strings."""
    if state is None:
        return []
    if isinstance(state, str):
        # Allow dot-delimited lists
        return [p.strip() for p in state.split(".") if p.strip()]
    return [str(p).strip() for p in state if str(p).strip()]


def canonicalize_smiles(smiles, kekule=True):
    """Canonicalize a single SMILES string via RDKit; returns input on failure."""
    if not smiles:
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=kekule)
    except Exception:
        return smiles


def canonicalize_state_string(state):
    """Canonicalize a dot-delimited state string."""
    species = split_species(state)
    if not species:
        return state
    canonical = []
    for smi in species:
        canon_smi = canonicalize_smiles(smi)
        canonical.append(canon_smi if canon_smi else smi)
    return ".".join(canonical)


def normalize_composition(all_species, overrides):
    """Build a normalized composition dict from all species and any overrides."""
    comp = {s: 0.0 for s in all_species}
    for k, v in (overrides or {}).items():
        comp[k] = float(v)
    total = sum(comp.values())
    if total > 0:
        return {k: v / total for k, v in comp.items()}
    # nothing provided or all zeros: give first species a mole fraction of 1.0
    if comp:
        first = next(iter(comp))
        comp[first] = 1.0
    return comp

def convert_energy_to_kcal(value, units):
    """Convert an energy value to kcal/mol from the given units."""
    u = (units or "").lower()
    if u in ("kcal/mol", "kcal"):
        return float(value)
    if u in ("kj/mol", "kj"):
        return float(value) * 0.239005736  # kJ -> kcal
    raise ValueError(f"Unsupported energy units '{units}'. Provide kcal/mol or kJ/mol.")

def fmt(x, places=3):
    """Round a number to the specified number of decimal places."""
    return str(Decimal(x).quantize(Decimal(f"1.{'0'*places}"), rounding=ROUND_HALF_UP))

def state_to_smiles(state_obj):
        """
        Normalize a reaction state to a Cantera-friendly SMILES representation.
        - If the state has multiple species, return a dot-delimited string in canonical order.
        - If the state exposes a single canonical SMILES, return that.
        """
        if hasattr(state_obj, "species"):
            smi_list = []
            for sp in getattr(state_obj, "species", []):
                smi = getattr(sp, "canon_smi", None)
                if smi:
                    smi_list.append(canonicalize_smiles(smi))
            return ".".join(filter(None, smi_list))
        # Fallback to state-level canon_smi if present
        raw = getattr(state_obj, "canon_smi", None)
        return canonicalize_state_string(raw)
    
def load_yarp_pickle(payload):
    """
    Accepts a pickle path, raw pickle bytes/bytearray, or an already loaded object.
    Returns the unpickled YARP reaction object.
    """
    if isinstance(payload, (bytes, bytearray)):
        return pkl.loads(payload)
    if isinstance(payload, (str, Path)):
        with open(payload, "rb") as fh:
            return pkl.load(fh)
    return payload

def state_composition(smiles_state):
    """
    Aggregate elemental composition for a dot-delimited SMILES state.
    """
    totals = Counter()
    for smi in split_species(smiles_state):
        if not smi:
            continue
        comp = elements_from_smiles(smi)
        totals.update(comp)
    return totals

def append_hydrogen(species_list, count):
    """
    Append hydrogen species to the provided list.
    Prefer H2 for even counts, fall back to atomic H otherwise.
    """
    while count >= 2:
        species_list.append("[H][H]")
        count -= 2
    while count > 0:
        species_list.append("[H]")
        count -= 1
        
        

def balance_hydrogen(entry):
    """
    Ensure each reaction entry is hydrogen balanced by adding H2/H
    to the deficient side when only hydrogen imbalance is detected.
    """
    reactants = split_species(entry.get("reactant_smi"))
    products = split_species(entry.get("product_smi"))

    react_comp = state_composition(entry.get("reactant_smi"))
    prod_comp = state_composition(entry.get("product_smi"))

    diff = Counter(prod_comp)
    diff.subtract(react_comp)

    non_h_diff = {el: val for el, val in diff.items() if el.upper() != "H" and val != 0}
    if non_h_diff:
        return entry  # leave untouched if other elements are imbalanced

    h_diff = diff.get("H", 0)
    if h_diff == 0:
        return entry

    if h_diff > 0:
        append_hydrogen(reactants, h_diff)
    else:
        append_hydrogen(products, -h_diff)

    entry = dict(entry)
    entry["reactant_smi"] = ".".join(reactants)
    entry["product_smi"] = ".".join(products)
    return entry

def normalized_state_tuple(state_str):
    """
    Return a deterministic tuple of species for a state string.
    Sorting makes A.B equivalent to B.A so we can identify duplicates reliably.
    """
    species = split_species(state_str)
    return tuple(sorted(species))


def quote_yaml_scalar(value):
    """Wrap a scalar in single quotes for YAML while escaping existing quotes."""
    text = "" if value is None else str(value)
    return "'" + text.replace("'", "''") + "'"


def canonical_thermo_label(state_name):
    """Map wrapper state labels to Cantera thermo keywords."""
    default = "ideal-gas"
    if not state_name:
        return default
    token = str(state_name).strip()
    key = token.lower().replace("_", "").replace("-", "")
    mapping = {
        "idealgas": "ideal-gas",
    }
    return mapping.get(key, token)


def format_species_block(cantera_job):
    """Emit the species list as a block sequence, quoting SMILES as needed."""
    lines = ["  species:"]
    for sp in getattr(cantera_job, "all_species", []):
        lines.append(f"    - {quote_yaml_scalar(sp)}")
    return lines


def format_state_block(cantera_job):
    """Emit the state section with nested mappings."""
    lines = [
        "  state:",
        f"    T: {getattr(cantera_job, 'Temperature', '')}",
        f"    P: {getattr(cantera_job, 'Pressure', '')}",
    ]
    key = "Y" if str(getattr(cantera_job, "rule", "")).lower() == "final_mass" else "X"
    lines.append(f"    {key}:")
    comp_entries = []
    seen = set()
    for entry in getattr(cantera_job, "initial_species", []):
        if not entry:
            continue
        smi = entry[0]
        frac = entry[1] if len(entry) > 1 else 1.0
        comp_entries.append((smi, frac))
        seen.add(smi)
    for sp in getattr(cantera_job, "all_species", []):
        if sp not in seen:
            comp_entries.append((sp, 0))
    for smi, value in comp_entries:
        lines.append(f"      {quote_yaml_scalar(smi)}: {value}")
    return lines


def sanitize_wrapper_yaml(cantera_job):
    """
    Rewrite sections of the wrapper YAML to avoid flow-style sequences that break SMILES.
    """
    raw_lines = cantera_job.f.getvalue().splitlines()
    sanitized_lines = []
    thermo_inserted = False
    for line in raw_lines:
        if line.startswith("  elements:"):
            derived = derive_element_list(cantera_job)
            if derived:
                sanitized_lines.append(f"  elements: [{', '.join(derived)}]")
            else:
                sanitized_lines.append(line)
            if not thermo_inserted:
                thermo = canonical_thermo_label(getattr(cantera_job, "state", None))
                sanitized_lines.append(f"  thermo: {thermo}")
                thermo_inserted = True
            continue
        if line.startswith("  species: ["):
            sanitized_lines.extend(format_species_block(cantera_job))
            continue
        if line.startswith("  state: {"):
            sanitized_lines.extend(format_state_block(cantera_job))
            sanitized_lines.append("")
            continue
        sanitized_lines.append(line)
    return "\n".join(sanitized_lines).rstrip() + "\n"


def derive_element_list(cantera_job):
    """
    Build a deterministic list of element symbols from the wrapper's species.
    """
    species_entries = getattr(cantera_job, "all_species", [])
    ordered = []
    seen = set()
    for entry in species_entries:
        for smi in split_species(entry):
            if not smi:
                continue
            try:
                comp = elements_from_smiles(smi)
            except Exception:
                continue
            for elem in comp.keys():
                symbol = elem[:1].upper() + elem[1:].lower()
                if symbol not in seen:
                    seen.add(symbol)
                    ordered.append(symbol)
    return ordered

