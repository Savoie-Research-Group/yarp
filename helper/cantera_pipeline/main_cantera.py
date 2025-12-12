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

INI_COMP = ["O=C(CO)[C@@H](O)[C@@H](O)[C@@H](O)CO", "O"]
INIT_FRAC = [0.5,0.5]


from copy import deepcopy
import pickle as pkl  # at the top
from pathlib import Path
from yarp.reaction import *
from rdkit import Chem
from rdkit.Chem import MolFromSmiles as smi2mol, MolToSmiles as mol2smi
from typing import Dict, List

from  write_yaml import *
from validate_yaml import *
from run_reactor import *
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

def write_cantera_yaml(cantera_data_list, 
                    out_yaml, 
                    temp, 
                    pressure, 
                    initial_species_smi,
                    initial_species_frac,
                    auto_balance_h=True, 
                    write_skip_csv=True, 
                    dg_units="kcal/mol", 
                    hygrogen_species=("[H]", "[H][H]")
                    ):
    """
    Given a list of cantera data dictionaries, write the Cantera YAML file.
    This function would utilize the existing write_cantera_yaml functionality.
    """
    print(f"Writing Cantera YAML to {out_yaml}...")
    print(f"Initial species SMILES: {initial_species_smi}"
        f"\nwith mole fractions: {initial_species_frac}")
    write_yaml(
        cantera_data_list,
        out_yaml,
        temp,
        pressure,
        initial_species_smi=initial_species_smi,    # if you have lists; else leave None
        initial_species_frac=initial_species_frac,  # if you have lists; else leave None
        auto_balance_h=True,
        write_skip_csv=True,
        dg_units=dg_units,
        hygrogen_species=("[H]", "[H][H]"),
        )

    
    # Return true if yaml is written successfully by checking if a yaml file is created.
    if Path(out_yaml).exists():
        print(f"Cantera YAML file written successfully: {out_yaml}")
        return True
    else:
        raise IOError(f"Failed to write Cantera YAML file: {out_yaml}")


def validate_cantera_yaml(yaml_path):
    """
    Validate the Cantera YAML file using the existing validate_cantera_yaml functionality.
    """
    if validate_yaml(yaml_path):
        print(f"YAML validation successful for: {yaml_path}")
        return True


def run_cantera_simulation(yaml_path, sim_len_s, sim_dt_s, out_prefix):
    """
    Run the Cantera simulation using the existing run_cantera_simulation functionality.
    """
    print(f"Running Cantera simulation for {sim_len_s}s with dt={sim_dt_s}s...")
    if run_reactor(yaml_path, sim_len_s, sim_dt_s, out_prefix):
        print(f"Cantera simulation completed successfully. Outputs prefixed with: {out_prefix}")
        return True
    else:
        print(f"Cantera simulation failed for YAML: {yaml_path}")

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
                    print(f"Warning: Reverse Theory '{theory}' not found in energy dict. Not including reverse barrier.")
                    return None
                print(f"Warning: Theory '{theory}' not found in energy dict. Specify theory or use a float.")
                exit()

def main_cantera(
    pickle,
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
    3. Write the Cantera YAML file using write_cantera_yaml
    4. Validate the YAML file using validate_cantera_yaml
    5. Run the Cantera simulation using run_cantera_simulation
    6. Parse the results using parse_cantera_results
    7. Update the reaction objects in the YARP pickle with the simulation results
    8. Return the updated YARP reaction pickle
    """
    print(f"\n===================================")
    print(f"=== Starting Cantera Pipeline ===")
    print(f"===================================")
    #0. Load YARP reaction pickle, define hash
    
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
    output_dir = Path(net_hash)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_yaml_name = output_dir / f"cantera_input_{net_hash}.yaml"
    
    #1. Extract reaction objects
    reactions = extract_reactions(updated_yarp_rxn_pickle)
    print(f"Extracted {len(reactions)} reactions from the YARP pickle.")

    #2. preparing Cantera data
    cantera_data_list = []
    for rxn_obj in reactions:
        cantera_data = pull_cantera_data_from_rxn_obj(rxn_obj, theory)
        cantera_data_list.append(cantera_data)
    print(f"Prepared Cantera data for {len(cantera_data_list)} reactions.")
    #for react in cantera_data_list:
        #print(f"Cantera data: {react}")
    
    #2.5 Dump Cantera Data to hashed pickle (for YAKS)
    cantera_data_pickle_path = output_dir / f"cantera_data_{net_hash}.pkl"
    with open(cantera_data_pickle_path, "wb") as fh:
        pkl.dump(cantera_data_list, fh)
    print(f"Cantera data pickle saved to: {cantera_data_pickle_path}")
    
    #2. Write Cantera YAML
    write_cantera_yaml(
        cantera_data_list,
        out_yaml_name,
        temp,
        pressure,
        initial_species_smi=initial_species_list,
        initial_species_frac=initial_species_mol_frac,
        auto_balance_h=True,
        write_skip_csv=True,
        dg_units=dg_units,
        hygrogen_species=("[H]", "[H][H]"),
    )
    
    #3. Validate YAML
    validate_cantera_yaml(out_yaml_name)
    
    print(f"\n===================================")
    print(f"=== Running Cantera Simulation for network: {net_hash} ===")
    print(f"===================================")
    #4. Run Cantera simulation
    run_cantera_simulation(
        out_yaml_name,
        simulation_length_s,
        sim_dt_s,
        out_prefix = output_dir / f"cantera_results_{net_hash}"  # OUT_PREFIX
    )
    
    #5. Parse results
    print(f"\n===================================")
    print(f"=== Parsing Cantera Results for network: {net_hash} ===")
    print(f"===================================")
    flux_results, conc_results = parse_cantera_results(output_dir / f"cantera_results_{net_hash}_reaction_flux_summary.csv", 
                                        output_dir / f"cantera_results_{net_hash}_final_concentrations.csv")
    
    #6. Update reaction objects with results
    update_rxn_obj_with_results(reactions, flux_results, conc_results)


    try:
        setattr(updated_yarp_rxn_pickle, "final_concentrations", conc_results)
    except Exception:
        pass
    # 7. Save hashed updated pickle
    updated_pickle_path = output_dir / f"updated_yarp_reactions_{net_hash}.pkl"
    with open(updated_pickle_path, "wb") as fh:
        pkl.dump(updated_yarp_rxn_pickle, fh)
    print(f"Updated YARP reaction pickle saved to: {updated_pickle_path}")

    #return updated pickle object and path
    print(f"\n=================================")
    print(f"=== Cantera Pipeline Completed for network: {net_hash} ===")
    print(f"=================================")
    print(f"You can find the updated YARP reaction pickle at: {updated_pickle_path}")
    print(f"You can find the Cantera YAML file at: {out_yaml_name}")
    print(f"You can find the Cantera simulation results at: {output_dir}")
    print(f"\n")
    return updated_yarp_rxn_pickle, updated_pickle_path

#pull in arguments with argparse when run as a script
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cantera Pipeline for YARP Reaction Pickles")
    parser.add_argument("--pickle", type=str, help="Path to the YARP reaction pickle file")
    parser.add_argument("--temp", type=float, default=500, help="Temperature in Kelvin")
    parser.add_argument("--pressure", type=float, default=1, help="Pressure in atm")
    parser.add_argument("--sim_l_s", type=float, default=1, help="Simulation length in seconds")
    parser.add_argument("--sim_dt_s", type=float, default=0.01, help="Simulation time step in seconds")
    parser.add_argument("--theory", type=str, default="DFT", help="Level of theory for energy extraction (DFT, EGAT, etc.)")
    parser.add_argument("--dg_units", type=str, default="kcal/mol", help="Units for Gibbs free energy")
    parser.add_argument("--initial_species_comp", type=List, default=None, help="Initial species composition")  
    parser.add_argument("--initial_species_mol_frac", type=list, default=None, help="Initial species mole fractions")  
    
    args = parser.parse_args()
    
    main_cantera(
        pickle=args.pickle,
        temp=args.temp,
        pressure=args.pressure,
        simulation_length_s=args.sim_l_s,
        sim_dt_s=args.sim_dt_s,
        dg_units=args.dg_units,
        theory=args.theory,
        initial_species_list=INI_COMP,
        initial_species_mol_frac=INIT_FRAC
    )
