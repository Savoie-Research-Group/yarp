#!/usr/bin/env python3
# yaml_from_csv_min.py
"""
List of Reaction Objects to Cantera YAML (minimal)

Input Objects: YARP Reactions (dataframe List)

- Each reaction object will have the following (useful) fields:
    - id: string identifier
    - reactant: "state" object with species attribute, which has a canon_smi property (self.canon_smi)
    - product: "state" object with species attribute, which has a canon_smi property
    - 
    
Outputs:
  <csv_basename>.yaml
"""

#OLD CODE:
"""
import sys, os, io
import pandas as pd
from network_utils import *
from rdkit.Chem import MolFromSmiles as smi2mol, MolToSmiles as mol2smi

# ---- constants (edit if needed) ----
DEFAULT_T_K = 1000      # for Eyring A-factor
DEFAULT_P_ATM = 1.0
DG_UNITS = "kcal/mol"     # Activation_PRED units assumed
WRITE_SKIP_CSV = True
AUTO_BALANCE_H = True
HYDROGEN_SPECIES = ("[H]", "[H][H]")  # H atom and H2


#species with nonzero starting mole fractions
INITIAL_COMPOSITION = {            # override zeros here
    "C=CC=C": 0.5,
    "C=C": 0.5,
}

def main():
    if not (2 <= len(sys.argv) <= 3):
        print("Usage: python yaml_from_csv_min.py reactions.csv [out.yaml]", file=sys.stderr)
        sys.exit(2)

    csv_path = sys.argv[1]
    out_yaml = "reactions.yaml"

    df = pd.read_csv(csv_path)
    need = {"ID","Rsmiles","Psmiles","Activation_PRED"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"CSV missing columns: {miss}")

    # gather species
    species = sorted({
        s for col in ("Rsmiles","Psmiles")
        for entry in df[col].fillna("")
        for s in split_smi_list(entry)
    })
    if AUTO_BALANCE_H:
        species = sorted(set(species) | set(HYDROGEN_SPECIES))
    elems = sorted({k for smi in species for k in elements(smi)})

    skipped = []
    buf = io.StringIO()
    buf.write("units: {length: cm, time: s, quantity: mol, activation-energy: kcal/mol, pressure: atm, energy: kcal}\n\n")
    buf.write("phases:\n")
    buf.write("- name: gas\n  thermo: ideal-gas\n")
    buf.write(f"  elements: [{', '.join(elems)}]\n")
    buf.write("  kinetics: gas\n  reactions: all\n")
    # Initial X all zeros (set your own later if desired)
    quoted_species = ", ".join(yq(s) for s in species)
    # Build composition dict, with user overrides
    comp_dict = {s: 0.0 for s in species}
    for k, v in INITIAL_COMPOSITION.items():
        if k not in comp_dict:
            print(f"⚠️  Warning: initial species '{k}' not in mechanism species list.")
        comp_dict[k] = v

    # Normalize (optional) so sum(X) = 1.0 if any nonzero given
    total = sum(comp_dict.values())
    if total > 0:
        comp_dict = {k: v / total for k, v in comp_dict.items()}

    zero_comp = "{ " + ", ".join([f"'{s}': {comp_dict[s]}" for s in species]) + " }"

    buf.write(f"  species: [{quoted_species}]\n")
    buf.write(f"  state: {{T: {DEFAULT_T_K}, P: {DEFAULT_P_ATM} atm, X: {zero_comp}}}\n\n")

    buf.write("species:\n")
    for s in species:
        buf.write(f"- name: '{s}'\n")
        buf.write(f"  composition: {elements(s)}\n")
        buf.write("  thermo:\n    model: constant-cp\n    cp0: 30.0\n    h0: 0.0\n    s0: 0.0\n")
        buf.write("  equation-of-state: ideal-gas\n\n")

    buf.write("reactions:\n")
    for i, row in df.iterrows():
        rid = str(row["ID"]).strip()
        R_list = split_smi_list(row["Rsmiles"])
        P_list = split_smi_list(row["Psmiles"])
        ok, why = balanced(R_list, P_list)
        if not ok and AUTO_BALANCE_H:
            patched_R, patched_P = patch_hydrogen_imbalance(R_list, P_list)
            if patched_R is not None:
                R_list, P_list = patched_R, patched_P
                ok, why = balanced(R_list, P_list)  # recheck
        if not ok:
            skipped.append({"row": i, "ID": rid, "why": why, "R": row["Rsmiles"], "P": row["Psmiles"]})
            continue

        try:
            dG_kcal = convert_dG_to_kcal(float(row["Activation_PRED"]), DG_UNITS)
        except Exception:
            skipped.append({"row": i, "ID": rid, "why": f"bad Activation_PRED: {row['Activation_PRED']}", "R": row["Rsmiles"], "P": row["Psmiles"]})
            continue

        m = len(R_list)
        A, b = eyring_A(m, DEFAULT_T_K, DEFAULT_P_ATM)
        eq = " + ".join(R_list) + " => " + " + ".join(P_list)
        buf.write(f"- equation: {yq(eq)}\n  type: elementary\n")
        if rid and rid.lower()!="nan":
            buf.write(f"  id: {rid}\n")
        buf.write(f"  rate-constant: {{A: {A:.6g}, b: {b:.3f}, Ea: {dG_kcal:.6g} kcal/mol}}\n\n")

    if skipped:
        buf.write("# ---- skipped reactions ----\n")
        for s in skipped:
            buf.write(f"# row={s['row']} ID={s['ID']} | {s['why']} | R={s['R']} | P={s['P']}\n")

    with open(out_yaml, "w") as f:
        f.write(buf.getvalue())

    if skipped and WRITE_SKIP_CSV:
        pd.DataFrame(skipped).to_csv(os.path.splitext(out_yaml)[0]+".skipped.csv", index=False)

    print(f"✅ Wrote {out_yaml}. Skipped {len(skipped)} reaction(s).")

if __name__ == "__main__":
    main()
"""