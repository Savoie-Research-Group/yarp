# Initial Structure Tutorial

This tutorial demonstrates four ways to initialize a YARP job — using **reaction SMILES strings** or **paired XYZ geometries**, each in single-reaction and batch variants — followed by EGAT barrier prediction.

Each subdirectory is self-contained. `cd` into any of them and run `yarp-init input.yaml` without overwriting output from other examples.

## Subdirectories

| Directory | Input type | # Reactions | Source file/dir |
|---|---|---|---|
| `single_smi/` | Reaction SMILES (`.txt`) | 1 | `rxn.txt` |
| `batch_smi/` | Reaction SMILES (`.txt`) | 7 | `batch_SMILES_rxn.txt` |
| `single_xyz/` | Paired XYZ geometries (directory) | 1 | `rxn/` |
| `batch_xyz/` | Paired XYZ geometries (directory) | 5 | `batch_xyz_rxn/` |

## Input Formats

### SMILES

Reactions are provided as a plain-text file of reaction SMILES strings. Each line encodes one reaction in the form `reactants>>products`.

```yaml
initialize:
  initial_structure:
    source: rxn.txt          # or batch_SMILES_rxn.txt for batch
    type: smiles
    mode: reaction
```

### XYZ

Reactions are provided as a directory of `.xyz` files. Each file contains two concatenated geometry blocks — the reactant geometry followed by the product geometry.

```yaml
initialize:
  initial_structure:
    source: rxn              # or batch_xyz_rxn for batch
    type: xyz
    mode: reaction
```

## Reproducing This Tutorial

From any subdirectory, the steps are the same. For example, to run the batch SMILES variant:

```bash
cd batch_smi
yarp-init input.yaml
```

### Step 1 — Initialize

```bash
yarp-init input.yaml
```

After this step, two new files appear:

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
