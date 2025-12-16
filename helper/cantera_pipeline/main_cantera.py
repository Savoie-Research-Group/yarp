#!/usr/bin/env python3
# yaml_from_csv_min.py
"""
Intakes a YARP reaction pickle, processes each reaction object to perform Cantera simulations,
and returns an updated YARP reaction pickle with the simulation results embedded.

In a loop, go through each reaction object in the yarp pickle and pull the necessary data to 
a dictionary, which can be passed to the write_cantera_yaml.cantera_handle function. 

The attributes of interest from the reaction object are:
- reactant (state object) SMILES
- product (state object) SMILES
- barrier 
- reverse barrier (if available)
- id
- hash

The yaml will be validated with validate_cantera_yaml.validate_cantera_yaml after writing.

The run will then be executed with run_cantera_simulation.run_cantera_simulation, and the results
will be parsed with parse_cantera_results.parse_cantera_results and added back to the reaction object on the 
following attributes (from the rxn object):
- max_flux

All this data (yamls, cantera outputs, etc) will be stored in a hashed directory for each unique network.

This hash will be derived from the inchikey for the initial species, plus the temperature, pressure, 
and length of the simulation.

Author: Thomas Burton
Date: 08DEC2025
"""

from copy import deepcopy
import pickle as pkl  # at the top
from pathlib import Path
from io import StringIO
from yarp.reaction import *
from rdkit import Chem
from rdkit.Chem import MolFromSmiles as smi2mol, MolToSmiles as mol2smi
from typing import Dict, List
import cantera as ct
import pandas as pd
import yaml
from cantera_wrapper import CANTERA
from cantera_util import *


def network_hash(initial_species, temperature_k, pressure_atm, sim_length_s):
    """Generate a hash string for the reaction network based on initial species and conditions."""
    inchies = []
    for smi in initial_species:
        mol = smi2mol(smi)
        inchi_key = Chem.InchiToInchiKey(Chem.MolToInchi(mol))  # full InChIKey
        head = inchi_key.split('-')[0]  # connectivity block, if truncation desired
        inchies.append(head)
    inchies.sort()
    inch_str = "_".join(inchies)
    parts = [inch_str,fmt(temperature_k, 2),fmt(pressure_atm, 2),fmt(sim_length_s, 2)]
    print(f"Network hash parts: {parts}")
    return ("_".join(parts))


def pull_cantera_data_from_rxn_obj(rxn_obj, theory):
    """
    Given a YARP reaction object, extract the necessary data for Cantera processing.
    Returns a dictionary with the required fields.
    """
    cantera_data = {
        "id": rxn_obj.id,
        "reactant_smi": state_to_smiles(rxn_obj.reactant),
        "product_smi": state_to_smiles(rxn_obj.product),
        "barrier": extract_barrier(rxn_obj.barrier, theory, reverse=False),
        "reverse_barrier": extract_barrier(getattr(rxn_obj, "reverse_barrier", None), theory, reverse=True),
        "hash": getattr(rxn_obj, 'hash', None),
        "heat_of_rxn": getattr(rxn_obj, 'heat_of_rxn', None),
        "dG_rxn": getattr(rxn_obj, 'dG_rxn', None)
    }
    return cantera_data


def parse_cantera_results(reaction_flux_summary_path, final_conc_path):
    """
    Parse Cantera outputs:
    - reaction fluxes: reaction_id -> max
    - final concentrations: species -> final concentration
    """
    max_fluxes = pd.read_csv(reaction_flux_summary_path)
    fluxes = {row["reaction_id"]: row["max"] for _, row in max_fluxes.iterrows()}

    final_path = Path(final_conc_path)
    conc_df = pd.read_csv(final_path)
    last_row = conc_df.iloc[-1]
    concentrations = {species: float(last_row[species]) for species in last_row.index}
    print(f"Parsed final concentrations for {len(concentrations)} species from {final_path}.")
    print(f"Parsed {len(fluxes)} reactions' max fluxes from {reaction_flux_summary_path}.")
    return fluxes, concentrations


def update_rxn_obj_with_results(reactions, fluxes, concentrations=None):
    # max_flux update
    updated = 0
    for rxn in reactions:
        rid = getattr(rxn, "id", None)
        if rid and rid in fluxes:
            rxn.max_flux = fluxes[rid]
            updated += 1
    print(f"Updated {updated} reactions with fluxes.")

    # per-state concentrations
    updated = 0
    if concentrations is not None:
        for rxn in reactions:
            for sp in getattr(rxn.reactant, "species", []):
                smi = getattr(sp, "canon_smi", None)
                if smi and smi in concentrations:
                    rxn.reactant.conc[smi] = concentrations[smi]
                    updated += 1
                    #print(f"Set concentration for reactant species {smi} to {concentrations[smi]}")
            for sp in getattr(rxn.product, "species", []):
                smi = getattr(sp, "canon_smi", None)
                if smi and smi in concentrations:
                    rxn.product.conc[smi] = concentrations[smi]
                    updated += 1
                    #print(f"Set concentration for product species {smi} to {concentrations[smi]}")
        print(f"Filled state.conc for matching species with final concentrations, {updated} updates made.")
    return reactions


def build_cantera_summary(cantera_data_list):
    """
    Build the summary pickle structure expected by updates_from_AVM.CANTERA.
    """
    summary = {}
    skipped = 0
    for entry in cantera_data_list:
        rid = entry.get("id") or f"rxn_{len(summary)}"
        barrier = entry.get("barrier", None)
        if barrier is None:
            skipped += 1
            continue
        summary[str(rid)] = {
            "reactant_smiles": entry.get("reactant_smi", ""),
            "product_smiles": entry.get("product_smi", ""),
            "activation_energy": barrier,
            "dG": entry.get("dG_rxn", 0.0) or 0.0,
        }
    summary["interior_nodes"] = []
    if skipped:
        print(f"Skipped {skipped} reactions without activation barriers for AVM wrapper.")
    return summary


def patch_cantera_species_builder(cantera_job):
    """Ensure wrapper enumerates individual species instead of dot-delimited states."""

    def _add_state_species(state_str, collected):
        for smi in split_species(state_str):
            if smi and smi not in collected:
                collected.append(smi)

    def _patched_pull_all_species(self):  # bound below
        species = []

        for entry in getattr(self, "initial_species", []):
            if isinstance(entry, (list, tuple)) and entry:
                _add_state_species(entry[0], species)
            else:
                _add_state_species(entry, species)

        for key, val in getattr(self, "summary", {}).items():
            if key == "interior_nodes" or not isinstance(val, dict):
                continue
            _add_state_species(val.get("reactant_smiles", ""), species)
            _add_state_species(val.get("product_smiles", ""), species)

        self.all_species = species

    cantera_job.pull_all_species = _patched_pull_all_species.__get__(cantera_job, CANTERA)


def auto_balance_entry(entry):
    """Balance a single reaction entry, dropping it if non-H elements are imbalanced."""
    entry = balance_hydrogen(entry)
    reactants = split_species(entry.get("reactant_smi", ""))
    products = split_species(entry.get("product_smi", ""))
    react_comp = state_composition(entry.get("reactant_smi", ""))
    prod_comp = state_composition(entry.get("product_smi", ""))

    if react_comp == prod_comp:
        return entry, True

    # If non-hydrogen elements are off, we cannot fix this automatically.
    react_non_h = {k: v for k, v in react_comp.items() if k.upper() != "H"}
    prod_non_h = {k: v for k, v in prod_comp.items() if k.upper() != "H"}
    if react_non_h != prod_non_h:
        return entry, False

    delta_h = prod_comp.get("H", 0) - react_comp.get("H", 0)
    if delta_h > 0:
        append_hydrogen(reactants, delta_h)
    elif delta_h < 0:
        append_hydrogen(products, -delta_h)

    entry = dict(entry)
    entry["reactant_smi"] = ".".join(reactants)
    entry["product_smi"] = ".".join(products)

    react_comp = state_composition(entry.get("reactant_smi", ""))
    prod_comp = state_composition(entry.get("product_smi", ""))
    return entry, react_comp == prod_comp


def enforce_reaction_balance(entries):
    """Ensure every entry is elementally balanced; drop ones that cannot be fixed."""
    balanced = []
    dropped = 0
    for entry in entries:
        fixed, ok = auto_balance_entry(entry)
        if not ok:
            dropped += 1
            rid = entry.get("id")
            print(f"Skipping unbalanced reaction {rid} (cannot auto-balance heavy elements).")
            continue
        balanced.append(fixed)
    if dropped:
        print(f"Removed {dropped} reactions that could not be balanced.")
    return balanced


def quote_species_names_in_yaml(yaml_text):
    """Ensure `- name:` entries quote SMILES so YAML parses correctly."""
    lines = []
    for line in yaml_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("- name: "):
            indent = line[: len(line) - len(stripped)]
            name = stripped[len("- name: ") :].strip()
            if not (name.startswith("'") and name.endswith("'")):
                name = quote_yaml_scalar(name)
            line = f"{indent}- name: {name}"
        lines.append(line)
    return "\n".join(lines) + "\n"

def write_settings_yaml(settings_path, initial_species_pairs, temp, pressure, sim_length, sim_dt):
    """
    Emit the YAML settings file consumed by the AVM Cantera wrapper.
    """
    settings = {
        "initial_species": initial_species_pairs,
        "direction": "forward",
        "state": "IdealGas",
        "press": "const",
        "therm": "isotherm",
        "rule": "css",
        "reactions": "all",
        "kinetics": "gas",
        "model": "constant-cp",
        "EOS": "ideal-gas",
        "Temperature": temp,
        "Pressure": pressure,
        "time_sim": sim_length,
        "time_step": sim_dt,
        "dump": None,
        "uncertainty_cycles": None
    }
    with open(settings_path, "w") as fh:
        yaml.safe_dump(settings, fh)


def dedupe_cantera_data(cantera_data_list, prefer="lowest_barrier"):
    """
    Deduplicate cantera_data dicts by (reactant_smi, product_smi).
    Returns:
      unique_list: list of kept cantera_data dicts
      dup_id_to_kept_id: dict mapping duplicate reaction id -> kept reaction id

    prefer:
      - "lowest_barrier": keep entry with smallest barrier (None treated as +inf)
      - "first": keep first seen
    """
    def key(d):
        return (
            normalized_state_tuple(d.get("reactant_smi", "")),
            normalized_state_tuple(d.get("product_smi", "")),
        )

    def barrier(d):
        b = d.get("barrier", None)
        if b is None:
            return float("inf")
        try:
            return float(b)
        except Exception:
            return float("inf")

    kept_by_key = {}          # key -> dict
    dup_id_to_kept_id = {}    # dup_id -> kept_id

    for d in cantera_data_list:
        k = key(d)
        rid = d.get("id", None)

        if k not in kept_by_key:
            kept_by_key[k] = d
            continue

        keep = kept_by_key[k]

        if prefer == "first":
            # current d becomes a duplicate of keep
            if rid is not None and keep.get("id", None) is not None:
                dup_id_to_kept_id[rid] = keep["id"]
            continue

        # prefer == "lowest_barrier"
        if barrier(d) < barrier(keep):
            # old keeper becomes duplicate of new
            old_id = keep.get("id", None)
            new_id = rid
            if old_id is not None and new_id is not None:
                dup_id_to_kept_id[old_id] = new_id
            kept_by_key[k] = d
        else:
            # new becomes duplicate of old keeper
            keep_id = keep.get("id", None)
            if rid is not None and keep_id is not None:
                dup_id_to_kept_id[rid] = keep_id

    unique_list = list(kept_by_key.values())
    return unique_list, dup_id_to_kept_id


def extract_reactions(container):
    """
    Normalize the reaction collection from various pickle shapes.
    Supports:
    - objects with a `.reactions` attribute
    - dict mapping -> values
    - iterable list/tuple/set of reactions
    """
    if hasattr(container, "reactions"):
        return list(getattr(container, "reactions"))
    if isinstance(container, dict):
        return list(container.values())
    if isinstance(container, (list, tuple, set)):
        return list(container)
    raise TypeError("Unsupported reaction container type; expected .reactions, dict, or iterable.")


def extract_barrier(energy, theory, reverse=False):
        """
        Barriers in YARP objects are often dicts keyed by level of theory (e.g., 'DFT').
        Accept either a float or dict; prefer DFT if present, else first value.
        """
        if energy is None:
            return None
        if isinstance(energy, dict):
            if theory in energy:
                return energy[theory]
            else:
                #Raise error and exit if theory not found
                if reverse:
                    #print(f"Warning: Reverse Theory '{theory}' not found in energy dict. Not including reverse barrier.")
                    return None
                print(f"Warning: Theory '{theory}' not found in energy dict. Specify theory or use a float.")
                exit()


def main_cantera(
    pickle,
    output,
    temp = 500,
    pressure = 1,
    simulation_length_s = 1,
    sim_dt_s = 0.01,
    theory = "DFT",
    dg_units = "kcal/mol",
    initial_species_list = List,
    initial_species_mol_frac= List,
):
    """
    Main Cantera Pipeline Function
    1. Expose the reaction objects from the YARP pickle
    2. For each reaction object:
        - retrieve the reactant and product SMILES, barrier, reverse barrier (if available), id, and hash
        - create a Cantera input dictionary
    3. Use the AVM CANTERA wrapper to write/validate/run the Cantera simulation
    4. Parse the results using parse_cantera_results
    5. Update the reaction objects in the YARP pickle with the simulation results
    6. Return the updated YARP reaction pickle
    """
    print(f"\n===================================")
    print(f"=== Starting Cantera Pipeline ===")
    print(f"===================================")
    print("Cantera version:", ct.__version__)
    #0. Load YARP reaction pickle, define hash
    if len(initial_species_list) != len(initial_species_mol_frac):
        print("Number of mol fractions does not match initial species list. Correct and try again.")
        exit()

    initial_species_list = [
        canonicalize_state_string(s) if s else s for s in initial_species_list
    ]
        
    print(f"Initial Composition:")
    for i in range(len(initial_species_list)):
        print(f"Species:{initial_species_list[i]}; Fraction:{initial_species_mol_frac[i]}")
        
    rxn_pickle_obj = load_yarp_pickle(pickle)
    updated_yarp_rxn_pickle = deepcopy(rxn_pickle_obj)
    net_hash = network_hash(
        initial_species = initial_species_list,
        temperature_k = temp,
        pressure_atm = pressure,  # atm to Pa
        sim_length_s = simulation_length_s
    )
    print(f"Network hash for this simulation: {net_hash}")
    print(f"\n===================================")
    print(f"=== Generating Cantera YAML for network: {net_hash} ===")
    print(f"===================================")
    #1. Extract reaction objects
    output_dir = Path(net_hash)
    output_dir.mkdir(parents=True, exist_ok=True)
    reactions = extract_reactions(updated_yarp_rxn_pickle)
    print(f"Extracted {len(reactions)} reactions from the YARP pickle.")

    #2. preparing Cantera data
    cantera_data_list = []
    for rxn_obj in reactions:
        cantera_data = pull_cantera_data_from_rxn_obj(rxn_obj, theory)
        cantera_data_list.append(balance_hydrogen(cantera_data))
    print(f"Prepared Cantera data for {len(cantera_data_list)} reactions.")
    #for react in cantera_data_list:
        #print(f"Cantera data: {react}")
    
    #3 Remove Duplicate Reactions, Keep the lower barrier
    unique_cantera_data_list, dup_id_map = dedupe_cantera_data(cantera_data_list, prefer="lowest_barrier")
    print(f"Deduped Cantera data: {len(unique_cantera_data_list)} unique (removed {len(cantera_data_list) - len(unique_cantera_data_list)} dups).")
    unique_cantera_data_list = enforce_reaction_balance(unique_cantera_data_list)

    #4 Dump Cantera Data to hashed pickle (for YAKS)
    cantera_data_pickle_path = output_dir / f"cantera_data_{net_hash}.pkl"
    with open(cantera_data_pickle_path, "wb") as fh:
        pkl.dump(unique_cantera_data_list, fh)
    print(f"Cantera data pickle saved to: {unique_cantera_data_list}")

    #5. Prepare AVM wrapper settings + summary files
    summary = build_cantera_summary(unique_cantera_data_list)
    summary_path = output_dir / "network_summary.pkl"
    with open(summary_path, "wb") as fh:
        pkl.dump(summary, fh)
    print(f"Saved network summary to: {summary_path}")

    settings_path = output_dir / "network_setting.yaml"
    initial_pairs = [[s, initial_species_mol_frac[idx]] for idx, s in enumerate(initial_species_list)]
    write_settings_yaml(settings_path,initial_pairs,temp,pressure,simulation_length_s,sim_dt_s,)
    print(f"Wrote network settings to: {settings_path}")
    
    #6 Run the Simulation
    print(f"\n===================================")
    print(f"=== Running Cantera Simulation for network: {net_hash} ===")
    print(f"===================================")
    cantera_job = CANTERA(path_to_settings_file=str(settings_path),path_to_dicts=str(output_dir),direction="forward",Temperature=temp,Pressure=pressure,time_sim=simulation_length_s,time_step=sim_dt_s,)
    patch_cantera_species_builder(cantera_job)
    cantera_job.write_yaml()
    yaml_text = sanitize_wrapper_yaml(cantera_job)
    yaml_text = quote_species_names_in_yaml(yaml_text)
    yaml_out_path = output_dir / f"cantera_input_{net_hash}.yaml"
    yaml_out_path.write_text(yaml_text)
    print(f"Cantera yaml written by wrapper to: {yaml_out_path}")
    cantera_job.f = StringIO(yaml_text)
    cantera_job.build_and_run_reactor()
    print(f"Reactor run complete.")
    
    #7. Parse results
    print(f"\n===================================")
    print(f"=== Parsing Cantera Results for network: {net_hash} ===")
    print(f"===================================")
    flux_results, conc_results = parse_cantera_results(
        cantera_job.reaction_summary_path,
        cantera_job.final_concentration_path,
    )
    print(f"Results parsed")
    #8. Update reaction objects with results
    update_rxn_obj_with_results(reactions, flux_results, conc_results)


    try:
        setattr(updated_yarp_rxn_pickle, "final_concentrations", conc_results)
    except Exception:
        pass
    # 7. Save updated pickle
    updated_pickle_path = output_dir / f"{output}_cantera.pkl"
    with open(updated_pickle_path, "wb") as fh:
        pkl.dump(updated_yarp_rxn_pickle, fh)
    print(f"Updated YARP reaction pickle saved to: {updated_pickle_path}")

    #return updated pickle object and path
    print(f"\n=================================")
    print(f"=== Cantera Pipeline Completed for network: {net_hash} ===")
    print(f"=================================")
    print(f"You can find the updated YARP reaction pickle at: {updated_pickle_path}")
    print(f"You can find the Cantera YAML file at: {yaml_out_path}")
    print(f"You can find the Cantera simulation results at: {output_dir}")
    print(f"\n")
    return updated_yarp_rxn_pickle, updated_pickle_path

#pull in arguments with argparse when run as a script
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Cantera Pipeline for YARP Reaction Pickles")
    parser.add_argument("--pickle", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--temp", type=float, default=500)
    parser.add_argument("--pressure", type=float, default=1)
    parser.add_argument("--sim_l_s", type=float, default=1)
    parser.add_argument("--sim_dt_s", type=float, default=0.01)
    parser.add_argument("--theory", type=str, default="DFT")
    parser.add_argument("--dg_units", type=str, default="kcal/mol")

    parser.add_argument(
        "--initial_species_comp",
        type=json.loads,
        required=True,
        help='JSON list, e.g. ["O=CCCOO","C=O"]'
    )
    parser.add_argument(
        "--initial_species_mol_frac",
        type=json.loads,
        required=True,
        help='JSON list, e.g. [1.0, 0.0]'
    )

    args = parser.parse_args()

    main_cantera(
        pickle=args.pickle,
        output=args.output,
        temp=args.temp,
        pressure=args.pressure,
        simulation_length_s=args.sim_l_s,
        sim_dt_s=args.sim_dt_s,
        dg_units=args.dg_units,
        theory=args.theory,
        initial_species_list=args.initial_species_comp,
        initial_species_mol_frac=args.initial_species_mol_frac
    )
