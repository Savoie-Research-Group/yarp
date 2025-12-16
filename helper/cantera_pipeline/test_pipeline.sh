#!/usr/bin/env bash
set -euo pipefail

# Run the Cantera YAML generation pipeline against a pickle.


python main_cantera.py \
    --pickle glucose_multi_path.pkl \
    --temp 1000 \
    --pressure 1 \
    --theory "DFT" \
    --sim_l_s 500 \
    --sim_dt_s 1 \
    --initial_species_comp "[""]" \
    --initial_species_mol_frac "[1.0]"