# Initial Structure Tutorial

This tutorial demonstrates two ways to initialize a YARP job for a batch of reactions — using **reaction SMILES strings** or **paired XYZ geometries** — followed by EGAT barrier prediction.

## Contents

| File/Directory | Description |
|---|---|
| `input.yaml` | YARP configuration for SMILES-based input |
| `input_xyz.yaml` | YARP configuration for XYZ-based input |
| `batch_SMILES_rxn.txt` | Reaction SMILES strings (one per line) |
| `batch_xyz_rxn/` | Paired reactant/product XYZ files (one per reaction) |

## Input Formats

### SMILES (`input.yaml`)

Reactions are provided as a plain-text file of reaction SMILES strings. Each line encodes one reaction in the form `reactants>>products`.

```yaml
initial_structure:
  source: batch_SMILES_rxn.txt
  type: smiles
  mode: reaction
```

### XYZ (`input_xyz.yaml`)

Reactions are provided as a directory of `.xyz` files, where each file contains two concatenated geometry blocks — the reactant geometry followed by the product geometry.

```yaml
initial_structure:
  source: batch_xyz_rxn
  type: xyz
  mode: reaction
```

Both configurations use the same job manager and EGAT stage settings: SLURM/Apptainer, up to 4 active jobs, and the `egat_rgd1` model with 8 CPUs and 1 GB RAM per CPU.

## Reproducing This Tutorial

The steps below apply to either input file. Substitute `input_xyz.yaml` in place of `input.yaml` to run the XYZ variant.

### Step 1 — Initialize

From the directory containing the input file, run:

```bash
yarp-init input.yaml
# or
yarp-init input_xyz.yaml
```

After this step, two new files will appear:

- `rxns.pkl` — serialized reaction data
- `STATUS.json` — job status tracking file

### Step 2 — Check Progress (First Time)

```bash
yarp-progress .
```

You should see one job queued, using 8 CPUs.

### Step 3 — Wait for Job Completion

Allow the queued job to finish before proceeding.

### Step 4 — Check Progress (After Completion)

```bash
yarp-progress .
```

All reactions should now show EGAT barriers.

### Step 5 — Inspect Results

```bash
yarp-read -ia rxns.pkl
```
