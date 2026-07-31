#!/bin/bash
# Run unit tests. Activate conda env first: conda activate egat-container
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:$PYTHONPATH"
pytest tests/ -v
