#!/bin/bash
# Run EGAT prediction. Activate conda env first: conda activate egat-container
# Usage: ./run_predict.sh /path/to/input.csv /path/to/output.csv
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:$PYTHONPATH"
INPUT="${1:-examples/sample_reactions.csv}"
OUTPUT="${2:-examples/predictions.csv}"
python egat_predict_reaction_csv.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --reactions-col reaction_smiles
