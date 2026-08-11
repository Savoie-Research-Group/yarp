#!/bin/bash
# Run EGAT self-test. Activate conda env first: conda activate egat-container
set -e
cd "$(dirname "$0")"
conda activate /groups/bsavoie2/bpiguave/egat_container_env
python /groups/bsavoie2/bpiguave/yarp-again/EGAT_container/egat_predict_reaction_csv.py --self-test \
  --input examples/sample_reactions.csv \
  --output examples/sample_out.csv
