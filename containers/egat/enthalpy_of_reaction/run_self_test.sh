#!/bin/bash
# Smoke test: predict on the bundled sample and verify finite outputs.
set -e
cd "$(dirname "$0")"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python predict_enthalpy_csv.py --self-test
