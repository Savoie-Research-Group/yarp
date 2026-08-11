#!/bin/bash
# Run enthalpy prediction with a local conda/micromamba env.
# Usage: ./run_predict.sh <input.csv> <output.csv> [reaction_column]
set -e
cd "$(dirname "$0")"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src:$PYTHONPATH"
INPUT="${1:-examples/sample_reactions.csv}"
OUTPUT="${2:-examples/predictions.csv}"
COL="${3:-reaction_smiles}"
python predict_enthalpy_csv.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --reactions-col "$COL"
