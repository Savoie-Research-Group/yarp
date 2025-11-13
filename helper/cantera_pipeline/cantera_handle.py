#!/usr/bin/env python3
# yaml_from_csv_min.py
"""
Intakes a List of Reaction Objects, Calls the following functions:
- rx_object_to_cantera_rxn (internal code to this file)
- write_cantera_yaml.py (seperate file)
- run_reactor.py (seperate file)
- compile_cantera_results.py (seperate file)
- cantera_rxn_to_rx_object.py (internal code to this file)

Input Objects: YARP Reactions (dataframe List)

- Each reaction object will have the following (useful) fields:
    - id: string identifier
    - reactant: "state" object with species attribute, which has a canon_smi property (self.canon_smi)
    - product: "state" object with species attribute, which has a canon_smi property
    - barrier: a dictionary of activation barriers (dG) at various levels of theory
    - reverse_barrier: a dictionary of activation barriers (dG) at various levels of theory

The output of the code will be a list of reaction objects, filling in the following fields (via self.<field>=<value>):
        - network_hash: unique hash for the reaction network associated with the kinetics
        - max_flux: maximum flux observed in the reactor simulation (float)
        - kinetic_end_mol_frac: final mole fraction of the product species (float)
        
    

    
"""

from copy import deepcopy
from yarp import *

def cantera_handle(Rxn_list, T_Kelvin, dt, t_end, Inital_Species_List, Initial_Mole_Fractions,
        P_atm=1.0, DG_Units="kcal/mol",
        Auto_Balance_H=True, Hydrogen_Species=("[H]", "[H][H]"),
        yaml_path="reactions.yaml", out_prefix="cantera_results",
        rule="css"):
    Rxn_List_Updated = Rxn_list.deepcopy()
    return Rxn_List_Updated

def Rxn_obj_to_Can_Obj(Rxn_obj):
    return Can_Obj