# EGAT — Reaction Enthalpy (ΔH) Container

Self-contained Docker / Apptainer image that predicts the **heat of reaction
ΔH (kcal/mol)** from atom-mapped reaction SMILES, using the embedding-based
`EGATReactionNet` model trained on the RGD1-CHNO dataset.

Test performance (random 70/20/10 split): **test MAE 1.53 kcal/mol, R² 0.983**;
endothermic (+ΔH) and exothermic (−ΔH) predicted symmetrically (reverse
augmentation), so the sign is meaningful.

## Contents

```
enthalpy_of_reaction/
├── Dockerfile                 # Docker build
├── enthalpy.def               # Apptainer/Singularity build
├── environment.yml            # conda env (python 3.11, torch 2.4 cpu, dgl 2.4, rdkit)
├── predict_enthalpy_csv.py    # CSV entrypoint
├── run_predict.sh             # local run helper
├── run_self_test.sh           # local smoke test
├── models/egat_dh.pth         # trained checkpoint (weights + cfg + y_mean/y_std)
├── examples/sample_reactions.csv
├── tests/test_enthalpy.py
└── src/rxn_egat/              # self-contained inference package
    ├── egat_conv.py           # EGATConv (verbatim, as used at training)
    ├── _graph_core.py         # return_matrix / return_reactive (exact copies)
    ├── rings.py               # vendored ring detection (numpy only)
    ├── elements.py            # element -> atomic number
    ├── featurize.py           # reaction -> graph tensors
    ├── model.py               # EGATReactionNet
    └── predict.py             # load_model / predict_reactions
```

## Input format

CSV with **either**:
- a `reaction_smiles` column as `reactant>>product` (also accepts
  `reactions` / `reaction` / `AAM` / `rxn_smiles`), **or**
- separate `reactant` and `product` columns.

SMILES must be atom-mapped and reactant/product must share the same atom set
(same ordering by map number). Output: `reaction_smiles,reaction_enthalpy`.

## Build

Docker:
```bash
cd enthalpy_of_reaction
docker build -t egat-enthalpy .
```

Apptainer:
```bash
cd enthalpy_of_reaction
apptainer build egat-enthalpy.sif enthalpy.def
```

## Run

Docker (mount a working dir):
```bash
docker run --rm -v "$PWD:/work" egat-enthalpy \
  --input /work/in.csv --output /work/out.csv
```

Apptainer:
```bash
apptainer run egat-enthalpy.sif \
  --input in.csv --output out.csv
# or:  ./egat-enthalpy.sif --input in.csv --output out.csv
```

Self-test (built-in sample):
```bash
docker run --rm egat-enthalpy --self-test
apptainer run egat-enthalpy.sif --self-test
```

## Run without a container

```bash
conda env create -f environment.yml
conda activate egat-enthalpy
./run_self_test.sh
./run_predict.sh in.csv out.csv
pytest tests/ -v
```

## Notes

- CPU image (`cpuonly`). Uses GPU automatically if `torch.cuda.is_available()`.
- Featurization is byte-identical to training (verified: container predictions
  match the training pipeline exactly), so results are reproducible.
- The checkpoint carries the target standardization; predictions are returned
  denormalized in kcal/mol.
