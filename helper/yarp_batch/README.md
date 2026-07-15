# yarp-batch

High-throughput wrappers around native `yarp-init` and `yarp-progress` for a CSV of starting SMILES.

`yarp-batch-i` reads `smiles.csv`, creates a numbered batch such as `runs/batch_001`, puts each SMILES in its own work directory, writes a normal YARP `input.yaml`, and runs `yarp-init` in that directory.

By default, each generated YARP pickle is named from the source molecule. The init script tries to build the first `yarpecule`, derive its InChIKey prefix, and write `<INCHIKEY>.pkl`. If that fails, it falls back to a file-safe SMILES stem plus a short hash.

`yarp-batch-p` scans initialized work directories for `STATUS.json`, runs `yarp-progress .` in each one, sleeps for the requested interval, and repeats until everything is quiescent or the duration expires.

## Batch Config

Edit `batch.yaml` to control batch size and locations:

```yaml
smiles_csv: smiles.csv
template_config: config.template.yaml
output_dir: runs
batch_size: 25
start_batch: 1
reaction_output_name: auto
progress_log_name: prog.out
```

`config.template.yaml` is the per-SMILES YARP template. The init script replaces:

```yaml
initialize:
  initial_structure:
    source: <SMILES>
    type: smiles
    mode: species
```

## Usage

Create and initialize the next batch, starting with `batch_001`:

```bash
cd helper/yarp_batch
yarp-batch-i batch.yaml
```

Create all batches from the CSV:

```bash
yarp-batch-i batch.yaml --all-batches
```

Create a specific batch or override size from the command line:

```bash
yarp-batch-i batch.yaml --batch-index 3 --batch-size 50
```

Dry-run a batch without writing or running YARP:

```bash
yarp-batch-i batch.yaml --dry-run
```

Progress every initialized run:

```bash
yarp-batch-p batch.yaml --interval 10 --duration 720
```

Progress one batch only:

```bash
yarp-batch-p batch.yaml --batch batch_001 --interval 10 --duration 720
```

Retry YARP-side failures up to two times:

```bash
yarp-batch-p batch.yaml --repair-failures --max-retries 2
```

Per-workdir logs:

- `yarp_init.out`
- `prog.out`, appended each cycle like `yarp-progress . >> prog.out 2>&1`
- `yarp_batch_retries.json` when retries are enabled

Each batch directory also gets `manifest.csv` and `yarp_batch.out`.
