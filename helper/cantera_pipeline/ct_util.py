#Imports
from typing import Dict
from rdkit import Chem
import argparse, csv, pickle
from typing import Tuple


# Constants
R_KCAL = 1.98720425864083e-3 # kcal/mol/K


def _elements_from_smiles(smiles):
    """Return elemental composition from a SMILES using RDKit.
    Unspecified/implicit hydrogens are made explicit by AddHs.
    """
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"Bad SMILES: {smiles}")
    m = Chem.AddHs(m)
    counts: Dict[str, int] = {}
    for a in m.GetAtoms():
        sym = a.GetSymbol()
        counts[sym] = counts.get(sym, 0) + 1
    return counts

def _quote(s):
    return '"' + s.replace('"', '\\"') + '"'

def extract_smiles_from_state(state_obj):
    smi: List[str] = []
    for sp in getattr(state_obj, "species", []) or []:
        val = getattr(sp, "canon_smi", None)
    if val:
        smi.append(val)
    return smi

def species_smiles_for_rxn(rxn):
    if hasattr(rxn, "species_smiles") and callable(rxn.species_smiles):
        return list(rxn.species_smiles())
    r = extract_smiles_from_state(getattr(rxn, "reactant", None))
    p = extract_smiles_from_state(getattr(rxn, "product", None))
    return sorted(set(r) | set(p))

#generate InChI from SMILES and truncate to non-stereochemical layer (before first dash)
def smi_to_inchi(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        return Chem.MolToInchi(mol).split('-')[0]
    else:
        return None


# =============================
# to_pickle.py
# =============================
"""
Create a pickle containing a list of YAKS_Rxn from a simple CSV or inline args.

CSV format (headers):
    id, reactant_smiles, product_smiles, rate_label, A, b, Ea_kcal
Where `reactant_smiles` and `product_smiles` are dot-separated lists, e.g.:
    C=C.[H], C[C], default, 1e12, 0.0, 10.0

Usage examples:
  # From CSV
  python to_pickle.py --csv reactions.csv --out reactions.pkl

  # From inline args (repeatable)
  python to_pickle.py --out reactions.pkl \
      --rxn "id=rxn1;R=C=C.[H];P=C[C];label=def;A=1e12;b=0;Ea=10" \
      --rxn "id=rxn2;R=O.O;P=O=O;label=def;A=5e11;b=0.5;Ea=15"
"""



def _parse_dot_list(s):
    return [t.strip() for t in s.split('.') if t.strip()]


def _parse_inline(s):
    # format: id=..;R=..;P=..;label=..;A=..;b=..;Ea=..
    parts = dict(
        (kv.split('=')[0].strip(), '='.join(kv.split('=')[1:]).strip())
        for kv in s.split(';') if '=' in kv
    )
    rxn_id = parts.get('id')
    R = _parse_dot_list(parts['R'])
    P = _parse_dot_list(parts['P'])
    label = parts.get('label', 'default')
    A = float(parts['A'])
    b = float(parts.get('b', '0'))
    Ea = float(parts['Ea'])

    rxn = YAKS_Rxn.from_smiles_lists(R, P, rxn_id)
    rxn.set_arrhenius(label, A, b, Ea)
    return rxn, label


def main_to_pickle(csv, out_path, rxn_specs):
    rxns: List[YAKS_Rxn] = []

    if args.csv:
        with open(args.csv, newline='') as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                rxn_id = row.get('id') or None
                R = _parse_dot_list(row['reactant_smiles'])
                P = _parse_dot_list(row['product_smiles'])
                label = row.get('rate_label', 'default') or 'default'
                A = float(row['A'])
                b = float(row.get('b', 0.0))
                Ea = float(row['Ea_kcal'])
                rxn = YAKS_Rxn.from_smiles_lists(R, P, rxn_id)
                rxn.set_arrhenius(label, A, b, Ea)
                rxns.append(rxn)

    if args.rxn:
        for spec in args.rxn:
            rxn, _ = _parse_inline(spec)
            rxns.append(rxn)

    if not rxns:
        raise SystemExit("No reactions supplied. Use --csv or --rxn.")

    with open(args.out, 'wb') as f:
        pickle.dump(rxns, f)
    print(f"Wrote {args.out} with {len(rxns)} reactions")


if __name__ == '__main__':
    # If you save this file as two modules, keep only their own main blocks.
    main_to_pickle()
