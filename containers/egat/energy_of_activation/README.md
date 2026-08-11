# EGAT_container (standalone)

Minimal, self-contained EGAT inference project. No external `yarp` dependency.

Reads a CSV of **reaction SMILES** and writes a CSV with:

- `reaction_smiles`
- `activation_barrier`
- `reaction_enthalpy`

---

## Folder layout

```
EGAT_container/
├── egat_predict_reaction_csv.py   # CLI entrypoint
├── run_self_test.sh               # Smoke test
├── run_predict.sh                 # Convenience wrapper
├── environment.yml               # Conda env
├── models/
│   ├── Activation_barrier.yaml
│   ├── Activation_barrier.pth
│   ├── Enthalpy.yaml
│   └── Enthalpy.pth
├── examples/
│   └── sample_reactions.csv
└── src/
    ├── egat/      # EGAT model, dataset, compile_model, predict_enthalpy
    ├── graph/     # adjacency, fragment
    ├── util/      # properties, misc
    └── lewis/      # be_mat (minimal)
```

---

## Setup

```bash
cd /groups/bsavoie2/bpiguave/yarp-again/EGAT_container
conda env create -f environment.yml
conda activate egat-container
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

---

## Input CSV format

Column **`reaction_smiles`** (or `reactions` / `reaction` / `AAM`):

```text
reaction_smiles
reactant_smiles>>product_smiles
...
```

Atom-mapped SMILES are expected.

---

## Run inference

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"

python egat_predict_reaction_csv.py \
  --input  in.csv \
  --output out.csv \
  --reactions-col reaction_smiles
```

Or use the wrapper:

```bash
./run_predict.sh in.csv out.csv
```

---

## Self-test

```bash
./run_self_test.sh
```

Or:

```bash
python egat_predict_reaction_csv.py --self-test \
  --input examples/sample_reactions.csv \
  --output examples/sample_out.csv
```

---

## Unit tests

```bash
conda activate egat-container  # or egat_container_env
export PYTHONPATH="$PWD/src:$PYTHONPATH"
pytest tests/ -v
```

Tests compare predictions against `tests/reference_predictions.csv` (tolerance 0.01 kcal/mol).

---

## Export conda env and Apptainer

**Export conda env** (from egat_container_env):

```bash
./export_env.sh
# or manually:
conda env export -n egat_container_env --no-builds > environment_exported.yml
```

**Build Apptainer** (minimal env from environment.yml):

```bash
apptainer build egat.sif egat.def
```

**Build from exported env** (matches egat_container_env exactly):

```bash
./export_env.sh   # create environment_exported.yml first
apptainer build egat.sif egat_from_exported.def
```

---

## Interactive CPU node (qrsh)

```bash
qrsh -q long -pe smp 12
cd /groups/bsavoie2/bpiguave/yarp-again/EGAT_container
conda activate egat-container
export PYTHONPATH="$PWD/src:$PYTHONPATH"
./run_self_test.sh
pytest tests/ -v
```

