from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import pickle as pkl  # at the top


from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from yarp.reaction import *
from rdkit import Chem
from rdkit.Chem import MolFromSmiles as smi2mol, MolToSmiles as mol2smi
from typing import Dict, List

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
    return "'" + s.replace("'", "''") + "'"

def eq_quote(s):
    # single-quote the whole equation; double any embedded single quotes for YAML
    return "'" + s.replace("'", "''") + "'"



def split_species(state):
    """Accept a SMILES string or sequence and return a list of species strings."""
    if state is None:
        return []
    if isinstance(state, str):
        # Allow dot-delimited lists
        return [p.strip() for p in state.split(".") if p.strip()]
    return [str(p).strip() for p in state if str(p).strip()]


def normalize_composition(all_species, overrides):
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
    u = (units or "").lower()
    if u in ("kcal/mol", "kcal"):
        return float(value)
    if u in ("kj/mol", "kj"):
        return float(value) * 0.239005736  # kJ -> kcal
    raise ValueError(f"Unsupported energy units '{units}'. Provide kcal/mol or kJ/mol.")

def fmt(x, places=3):
    return str(Decimal(x).quantize(Decimal(f"1.{'0'*places}"), rounding=ROUND_HALF_UP))

def state_to_smiles(state_obj):
        """
        Normalize a reaction state to a Cantera-friendly SMILES representation.
        - If the state has multiple species, return a dot-delimited string in canonical order.
        - If the state exposes a single canonical SMILES, return that.
        """
        if hasattr(state_obj, "species"):
            # state.species is a list of yarpecule objects
            smi_list = [sp.canon_smi for sp in state_obj.species]
            return ".".join(smi_list)
        # Fallback to state-level canon_smi if present
        return getattr(state_obj, "canon_smi", None)

def extract_barrier(energy):
        """
        Barriers in YARP objects are often dicts keyed by level of theory (e.g., 'DFT').
        Accept either a float or dict; prefer DFT if present, else first value.
        """
        if energy is None:
            return None
        if isinstance(energy, dict):
            if "DFT" in energy:
                return energy["DFT"]
            # grab the first available entry
            for _, v in energy.items():
                return v
            return None
        return energy
    
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