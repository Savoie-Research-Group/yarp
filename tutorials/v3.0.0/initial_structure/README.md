# Initial Structure Tutorial

This tutorial demonstrates four ways to initialize a YARP job followed by EGAT barrier prediction:
1. `batch_xyz`: initialize from a directory of **paired XYZ geometries**
  - No product enumeration is required to initialize reaction objects
2. `batch_smi`: initialize from a file containing **reaction SMILES strings**
  - No product enumeration is required to initialize reaction objects
3. `single_xyz`: initialize from a single species XYZ file
  - Requires product enumeration to create reaction objects from a single species
4. `single_smi`: initialize from a single species SMILES string
  - Requires product enumeration to create reaction objects from a single species

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
    source: batch_SMILES_rxn.txt
    type: smiles
    mode: reaction
```

For a single species, you can directly write the SMILES in `source`
```yaml
initialize:
  initial_structure:
    source: OCC=O
    type: smiles
    mode: species
```

### XYZ

Reactions are provided as a directory of `.xyz` files. Each file contains two concatenated geometry blocks — the reactant geometry followed by the product geometry.

```yaml
initialize:
  initial_structure:
    source: batch_xyz_rxn           # or provide a single .xyz file to load in one reaction object
    type: xyz
    mode: reaction
```

For a single species, provide a path to an XYZ file with a single entry. If multiple are provided, YARP will read in the last entry
```yaml
initialize:
  initial_structure:
    source: mol.xyz
    type: xyz
    mode: species
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
