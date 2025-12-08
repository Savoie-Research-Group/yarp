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
import hashlib
import pickle as pkl  # at the top


from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from yarp.reaction import *
from rdkit import Chem
from rdkit.Chem import MolFromSmiles as smi2mol, MolToSmiles as mol2smi
from typing import Dict, List

# Import existing helper functions
from  write_yaml import *
from validate_yaml import *
from run_reactor import *

import sys
import importlib


def fmt(x, places=3):
    return str(Decimal(x).quantize(Decimal(f"1.{'0'*places}"), rounding=ROUND_HALF_UP))

def network_hash(initial_species, temperature_k, pressure_atm, sim_length_s):
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

def pull_cantera_data_from_rxn_obj(rxn_obj):
    """
    Given a YARP reaction object, extract the necessary data for Cantera processing.
    Returns a dictionary with the required fields.
    """

    cantera_data = {
        "id": rxn_obj.id,
        "reactant_smi": state_to_smiles(rxn_obj.reactant),
        "product_smi": state_to_smiles(rxn_obj.product),
        "barrier": extract_barrier(rxn_obj.barrier),
        "reverse_barrier": extract_barrier(getattr(rxn_obj, "reverse_barrier", None)),
        "hash": getattr(rxn_obj, 'hash', None),
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

def parse_cantera_results(reaction_flux_summary_path):
    """
    Parse the results from the Cantera simulation from the provided csv.
    return a dictionary mapping reaction IDs to their max flux.
    """
    max_fluxes = pd.read_csv(reaction_flux_summary_path)
    parsed = {}
    for _, row in max_fluxes.iterrows():
        rxn_id = row["reaction_id"]
        max_flux = row["max"]
        parsed[rxn_id] = max_flux
    print(f"Parsed {len(parsed)} reactions' max fluxes from {reaction_flux_summary_path}.")
    return parsed    

def update_rxn_obj_with_results(reactions, parsed_results):
    """
    Update the reaction object with the parsed results from the Cantera simulation.
    rxn_objs: list of reaction objects (from _extract_reactions)
    parsed_results: dict {reaction_id: max_flux}
    """
    updated = 0
    for rxn in reactions:
        rid = getattr(rxn, "id", None)
        if rid and rid in parsed_results:
            setattr(rxn, "max_flux",parsed_results[rid])
            updated += 1
            print(f"Reaction {rid} updated with max_flux: {parsed_results[rid]}")
    print(f"Updated {updated} reactions with fluxes.")
    return reactions

def _load_yarp_pickle(payload):
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

def _extract_reactions(container):
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


def main_cantera(
    pickle,
    temp: float = 500,
    pressure: float = 1,
    simulation_length_s: float = 1,
    sim_dt_s: float = 0.01,
    dg_units: str = "kcal/mol",
    final_conc: bool = False,
    initial_species_list = list,
    initial_species_mol_frac = list
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
    #0. Load YARP reaction pickle, define hash
    rxn_pickle_obj = _load_yarp_pickle(pickle)
    updated_yarp_rxn_pickle = deepcopy(rxn_pickle_obj)
    net_hash = network_hash(
        initial_species = initial_species_list,
        temperature_k = temp,
        pressure_atm = pressure,  # atm to Pa
        sim_length_s = simulation_length_s
    )
    print(f"Network hash for this simulation: {net_hash}")
    output_dir = Path(net_hash)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_yaml_name = output_dir / f"cantera_input_{net_hash}.yaml"
    #1. Extract reaction objects
    reactions = _extract_reactions(updated_yarp_rxn_pickle)
    print(f"Extracted {len(reactions)} reactions from the YARP pickle.")

    #2. preparing Cantera data
    cantera_data_list = []
    for rxn_obj in reactions:
        cantera_data = pull_cantera_data_from_rxn_obj(rxn_obj)
        cantera_data_list.append(cantera_data)
    print(f"Prepared Cantera data for {len(cantera_data_list)} reactions.")
    
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
    
    #4. Run Cantera simulation
    run_cantera_simulation(
        out_yaml_name,
        simulation_length_s,
        sim_dt_s,
        out_prefix = output_dir / f"cantera_results_{net_hash}"  # OUT_PREFIX
    )
    
    #5. Parse results
    parsed_results = parse_cantera_results(output_dir / f"cantera_results_{net_hash}_reaction_flux_summary.csv")
    
    #6. Update reaction objects with results
    update_rxn_obj_with_results(reactions, parsed_results)

    # 7. Save hashed updated pickle
    updated_pickle_path = output_dir / f"updated_yarp_reactions_{net_hash}.pkl"
    with open(updated_pickle_path, "wb") as fh:
        pkl.dump(updated_yarp_rxn_pickle, fh)
    print(f"Updated YARP reaction pickle saved to: {updated_pickle_path}")

    #return updated pickle object and path
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
    parser.add_argument("--dg_units", type=str, default="kcal/mol", help="Units for Gibbs free energy")
    parser.add_argument("--final_conc", action="store_true", help="Flag to return final concentrations")
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
        final_conc=args.final_conc,
        initial_species_list=INI_COMP,
        initial_species_mol_frac=INIT_FRAC
    )
