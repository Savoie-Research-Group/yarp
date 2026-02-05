#!/usr/bin/env python3
"""
run_cantera_constants.py — minimal batch reactor runner (no CLI flags)

- Edit the CONSTANTS block below to set:
    YAML_PATH, T_END, DT, RULE, OUT_PREFIX
- Loads a Cantera YAML (phase 'gas')
- Runs isothermal, constant-pressure reactor
- Saves time-series and summaries to CSV and XLSX
- Carries reaction IDs (from YAML 'id:' fields) into outputs

RULE choices:
  - "cnf"        : cumulative net formation (integral of species prod. rates)
  - "css"        : final mole fractions
  - "final_mass" : final mass fractions
  - "cum_conc"   : cumulative concentrations (integral of conc.)
"""

from __future__ import annotations

# ---------------------------
# CONSTANTS (edit these)
# ---------------------------
YAML_PATH  = "reactions.yaml"          # Path to your Cantera YAML (phase: 'gas')
T_END      = 3600                # [s]
DT         = 0.1                  # [s]
RULE       = "css"                # one of: {"cnf","css","final_mass","cum_conc"}
OUT_PREFIX = "cantera_results"    # output prefix for CSV/XLSX

# ---------------------------
# Imports
# ---------------------------
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import cantera as ct


# ---------------------------
# Mechanics & Simulation
# ---------------------------

def load_mech(yaml_path):
    """Load mechanism and return (gas, species_names, reaction_equations, reaction_ids)."""
    gas = ct.Solution(yaml_path, "gas")
    rxns = gas.reactions()
    rxn_eqns = [rxn.equation for rxn in rxns]
    # Prefer explicit YAML ids; fall back to a deterministic placeholder
    rxn_ids  = [getattr(rxn, "id", "") or f"rxn_{i:04d}" for i, rxn in enumerate(rxns)]
    return gas, gas.species_names, rxn_eqns, rxn_ids


def run_reactor(gas, t_end, dt):
    """Run isothermal constant-P reactor; return collected arrays."""
    r = ct.IdealGasConstPressureReactor(gas, energy="off")
    sim = ct.ReactorNet([r])

    times, X, Y, C, R, S = [], [], [], [], [], []
    times.append(sim.time); X.append(gas.X.copy()); Y.append(gas.Y.copy())
    C.append(gas.concentrations.copy()); R.append(gas.net_rates_of_progress.copy())
    S.append(gas.net_production_rates.copy())

    while sim.time < t_end:
        sim.advance(sim.time + dt)
        times.append(sim.time); X.append(gas.X.copy()); Y.append(gas.Y.copy())
        C.append(gas.concentrations.copy()); R.append(gas.net_rates_of_progress.copy())
        S.append(gas.net_production_rates.copy())

    return {
        "t": np.asarray(times),
        "X": np.vstack(X),
        "Y": np.vstack(Y),
        "C": np.vstack(C),
        "R": np.vstack(R),  # (N_time, N_rxns) net rates of progress
        "S": np.vstack(S),  # (N_time, N_species) net production rates
    }


# ---------------------------
# Summaries
# ---------------------------

def build_flux_summary(t, R, rxn_eqns, rxn_ids):
    """
    Per-reaction flux metrics from net rates of progress R(t).

    Columns:
    - reaction_index
    - reaction_id
    - equation
    - flux_signed_integral  (∫ r_i dt; net progress, signed)
    - flux_abs_integral     (∫ |r_i| dt; total traffic)
    - final_rate            (r_i at t_end)
    - max_abs_rate
    - time_of_max_abs_rate
    - mean_rate
    - std_rate
    """
    # Use np.trapezoid for wide compatibility (works with NumPy 1.x and 2.x)
    flux_signed = np.trapezoid(R, x=t, axis=0)
    flux_abs    = np.trapezoid(np.abs(R), x=t, axis=0)

    final_rate = R[-1, :]
    abs_R = np.abs(R)
    max_idx = abs_R.argmax(axis=0)
    max_abs_rate = abs_R[max_idx, np.arange(R.shape[1])]
    time_of_max  = t[max_idx]
    mean_rate = R.mean(axis=0)
    std_rate  = R.std(axis=0, ddof=1 if R.shape[0] > 1 else 0)

    df = pd.DataFrame({
        "reaction_index": np.arange(R.shape[1], dtype=int),
        "reaction_id": rxn_ids,
        "equation": rxn_eqns,
        "flux_signed_integral": flux_signed,
        "flux_abs_integral": flux_abs,
        "final_rate": final_rate,
        "max_abs_rate": max_abs_rate,
        "time_of_max_abs_rate": time_of_max,
        "mean_rate": mean_rate,
        "std_rate": std_rate,
    }).sort_values("flux_abs_integral", ascending=False).reset_index(drop=True)
    return df


def summarize(rule, arrs, species, dt):
    """Return summary DataFrame for chosen rule or None."""
    if rule == "cnf":
        vals = np.trapezoid(arrs["S"][1:], dx=dt, axis=0)         # integrate species prod. rates
        return pd.DataFrame([vals], columns=species)
    if rule == "css":
        return pd.DataFrame([arrs["X"][-1]], columns=species) # final mole fraction
    if rule == "final_mass":
        return pd.DataFrame([arrs["Y"][-1]], columns=species) # final mass fraction
    if rule == "cum_conc":
        vals = np.trapezoid(arrs["C"], dx=dt, axis=0)             # integrate concentrations
        return pd.DataFrame([vals], columns=species)
    return None


# ---------------------------
# I/O
# ---------------------------

def save_outputs(prefix,arrs,species,rxn_eqns,rxn_ids,summary,rule):
    """Write tidy CSVs and an XLSX workbook (includes reaction_fluxes and id map)."""
    # Time-series tables
    df_time = pd.DataFrame({"t": arrs["t"]})
    df_X    = pd.DataFrame(arrs["X"], columns=species)
    df_Y    = pd.DataFrame(arrs["Y"], columns=species)
    df_C    = pd.DataFrame(arrs["C"], columns=species)
    # Use reaction IDs as column headers for time-series rates
    df_R    = pd.DataFrame(arrs["R"], columns=rxn_ids)
    df_S    = pd.DataFrame(arrs["S"], columns=species)

    # Flux summary per reaction
    df_flux = build_flux_summary(arrs["t"], arrs["R"], rxn_eqns, rxn_ids)

    # (Optional but handy) mapping of reaction_index → (reaction_id, equation)
    df_map = pd.DataFrame({
        "reaction_index": np.arange(len(rxn_ids), dtype=int),
        "reaction_id": rxn_ids,
        "equation": rxn_eqns
    })

    # CSVs
    df_time.to_csv(f"{prefix}_time.csv", index=False)
    df_X.to_csv(f"{prefix}_mole_fractions.csv", index=False)
    #df_Y.to_csv(f"{prefix}_mass_fractions.csv", index=False)
    #df_C.to_csv(f"{prefix}_concentrations.csv", index=False)
    df_R.to_csv(f"{prefix}_rxn_rates.csv", index=False)               # columns = reaction_id
    #df_S.to_csv(f"{prefix}_species_prod_rates.csv", index=False)
    df_flux.to_csv(f"{prefix}_reaction_fluxes.csv", index=False)      # single per-reaction flux CSV
    df_map.to_csv(f"{prefix}_reaction_index_map.csv", index=False)    # for joins / debugging
    if summary is not None:
        summary.to_csv(f"{prefix}_{rule}_summary.csv", index=False)

    # Excel (same sheets)
    #with pd.ExcelWriter(f"{prefix}.xlsx") as w:
        #df_time.to_excel(w, sheet_name="time", index=False)
        #df_X.to_excel(w, sheet_name="mole_fractions", index=False)
        #df_Y.to_excel(w, sheet_name="mass_fractions", index=False)
    #    df_C.to_excel(w, sheet_name="concentrations", index=False)
        #df_R.to_excel(w, sheet_name="rxn_rates", index=False)
        #df_S.to_excel(w, sheet_name="species_prod_rates", index=False)
    #    df_flux.to_excel(w, sheet_name="reaction_fluxes", index=False)
        #df_map.to_excel(w, sheet_name="reaction_index_map", index=False)
        #if summary is not None:
        #    summary.to_excel(w, sheet_name=f"{rule}_summary", index=False)


# ---------------------------
# Runner
# ---------------------------

def main():
    # Basic validation
    allowed = {"cnf", "css", "final_mass", "cum_conc"}
    if RULE not in allowed:
        raise ValueError(f"RULE must be one of {allowed}, got '{RULE}'")
    if DT <= 0 or T_END <= 0:
        raise ValueError("DT and T_END must be positive.")

    gas, species, rxn_eqns, rxn_ids = load_mech(YAML_PATH)
    arrs = run_reactor(gas, T_END, DT)
    summary = summarize(RULE, arrs, species, DT)
    save_outputs(OUT_PREFIX, arrs, species, rxn_eqns, rxn_ids, summary, RULE)

    #print(f"✅ Wrote CSVs with prefix '{OUT_PREFIX}_*.csv' and Excel '{OUT_PREFIX}.xlsx'")
    print(f"   ➤ Flux summary: {OUT_PREFIX}_reaction_fluxes.csv")
    #print(f"   ➤ Reaction map: {OUT_PREFIX}_reaction_index_map.csv (index ↔ id ↔ equation)")


if __name__ == "__main__":
    main()
