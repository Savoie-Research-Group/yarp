#!/usr/bin/env python3
import argparse
import pathlib
import cantera as ct
import pandas as pd
import numpy as np


def run_reactor(yaml_path, t_end, time_step, out_prefix):
    """Run an isothermal, isobaric reactor using the given Cantera YAML file."""
    yaml_path = pathlib.Path(yaml_path)
    out_prefix = pathlib.Path(out_prefix or yaml_path.with_suffix("").name)  # base name of the YAML
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    gas = ct.Solution(yaml_path)
    r = ct.IdealGasConstPressureReactor(contents=gas, energy="off", name="isothermal_reactor")
    sim = ct.ReactorNet([r])
    states = ct.SolutionArray(gas, extra=["t"])

    # march forward
    states.append(r.thermo.state, t=sim.time)
    while sim.time < t_end:
        sim.advance(sim.time + time_step)
        states.append(r.thermo.state, t=sim.time)

    # per-species flux stats over the trajectory
    net_prod = states.net_production_rates          # (nt, nspecies)
    stats = {
        "max":   np.max(net_prod, axis=0),
        "min":   np.min(net_prod, axis=0),
        "mean":  np.mean(net_prod, axis=0),
        "final": net_prod[-1],
    }

    # species flux summary
    summary_df = pd.DataFrame(stats, index=states.species_names)
    summary_df.index.name = "species"
    summary_df.to_csv(f"{out_prefix}_species_flux_summary.csv")

    # reaction stats over time
    rxn_rates = states.net_rates_of_progress  # shape: (nt, n_rxn)
    rxns = gas.reactions()
    rxn_ids = [getattr(rx, "ID", f"rxn_{i}") or f"rxn_{i}" for i, rx in enumerate(rxns)]

    rxn_stats = {
        "max":   np.max(rxn_rates, axis=0),
        "min":   np.min(rxn_rates, axis=0),
        "mean":  np.mean(rxn_rates, axis=0),
        "final": rxn_rates[-1],
    }
    rxn_summary = pd.DataFrame(rxn_stats, index=rxn_ids)
    rxn_summary.index.name = "reaction_id"
    rxn_summary.to_csv(f"{out_prefix}_reaction_flux_summary.csv")

    # keep final mole fractions
    spe_x = states.X[-1]
    conc_df = pd.DataFrame([spe_x], columns=states.species_names)
    conc_df.to_csv(f"{out_prefix}_final_mole_fractions.csv", index=False)
    
    # Keep final concentrations
    conc = states.concentrations[-1]
    conc_df = pd.DataFrame([conc], columns=states.species_names)
    conc_df.to_csv(f"{out_prefix}_final_concentrations.csv", index=False)

    return True
