#!/usr/bin/env python3
"""
Two lightweight, standalone modules in one file for convenience.
Split into two files in your repo as:
  - yaks_rxn.py
  - to_pickle.py

They are intentionally simple and mirror the YARP `reaction` style.
No nested classes, no hidden magic. Just attributes.
"""

# =============================
# yaks_rxn.py
# =============================
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable, Any


@dataclass
class Species:
    """Minimal species record used by YAKS_Rxn states.
    Use canonical SMILES in `canon_smi`. `inchi` is optional.
    """
    canon_smi: str
    inchi: Optional[str] = None


@dataclass
class State:
    """Minimal YARP-like `state` with just a `species` list."""
    species: List[Species] = field(default_factory=list)

    @classmethod
    def from_smiles(cls, smiles_list: Iterable[str]) -> "State":
        return cls([Species(s) for s in smiles_list])


@dataclass
class RateModel:
    """Arrhenius parameters in conventional units.
    k(T) = A * T^b * exp(-Ea/RT) with Ea in kcal/mol.
    """
    A: float
    b: float
    Ea_kcal_per_mol: float


class YAKS_Rxn:
    """
    Simple reaction container mirroring YARP's `reaction` layout.

    Attributes (mirroring YARP):
    ----------------------------
    reactant : State
    product  : State
    ts : dict                       # transition state info (by level of theory)
    barrier : dict                  # dG‡ (kcal/mol) forward, keyed by level
    reverse_barrier : dict          # dG‡ (kcal/mol) reverse, keyed by level
    heat_of_rxn : dict              # dH (kcal/mol) keyed by level
    max_flux : float                # for downstream microkinetics
    id : str                        # human-readable reaction label
    hash : int                      # unique-ish identifier

    Cantera add-ons (kept minimal):
    -------------------------------
    rate : Dict[str, RateModel]     # keyed by a label (e.g., "B3LYP")
    """

    def __init__(self, reactant, product, rxn_id = None):
        self.reactant: State = reactant
        self.product: State = product

        self.ts: Dict[str, Any] = {}
        self.barrier: Dict[str, float] = {}
        self.reverse_barrier: Dict[str, float] = {}
        self.heat_of_rxn: Dict[str, float] = {}

        self.max_flux: float = 0.0

        # ID: prefer InChI if present; else first SMILES on each side
        r_inchi = self._first_attr(self.reactant, "inchi")
        p_inchi = self._first_attr(self.product, "inchi")
        if rxn_id is not None:
            self.id = rxn_id
        elif r_inchi and p_inchi:
            self.id = f"{r_inchi}_to_{p_inchi}"
        else:
            r0 = self._first_attr(self.reactant, "canon_smi") or "R"
            p0 = self._first_attr(self.product, "canon_smi") or "P"
            self.id = f"{r0}_to_{p0}"
        self.hash = hash((self.id,))

        # Cantera rates by label
        self.rate: Dict[str, RateModel] = {}

    # ---------- helpers ----------
    @staticmethod
    def _first_attr(state, attr):
        for sp in state.species:
            v = getattr(sp, attr, None)
            if v:
                return v
        return None

    def species_smiles(self):
        r = [sp.canon_smi for sp in self.reactant.species]
        p = [sp.canon_smi for sp in self.product.species]
        return sorted(set(r) | set(p))

    def set_arrhenius(self, label, A, b, Ea_kcal_per_mol):
        self.rate[label] = RateModel(A=A, b=b, Ea_kcal_per_mol=Ea_kcal_per_mol)

    @classmethod
    def from_smiles_lists(cls, reactant_smiles, product_smiles, rxn_id):
        return cls(State.from_smiles(list(reactant_smiles)), State.from_smiles(list(product_smiles)), rxn_id=rxn_id)