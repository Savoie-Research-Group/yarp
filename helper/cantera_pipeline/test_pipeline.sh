#!/usr/bin/env bash
set -euo pipefail

# Run the Cantera YAML generation pipeline against a pickle.


python main_cantera.py \
    --pickle glucose_multi_path.pkl \
    --temp 1000 \
    --pressure 1 \
    --theory "DFT" \
    --sim_l_s 5000 \
    --sim_dt_s 1 \