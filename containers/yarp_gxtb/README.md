# yarp_gxtb container

YARP runs Pysisyphus tasks (pre-opt, GSM, R/P opt, TS opt, IRC) in a container
chosen by the task's level of theory:

- `lot: xtb` uses `erm42/yarp:pysis_xtb` (stock Pysisyphus + conda xtb).
- `lot: gxtb` uses `erm42/yarp_gxtb:v1`, built from this folder.

The mapping is in `yarp/reaction/external/pysis.py` (`pysis_image`,
`PYSIS_GXTB_IMAGE`).

## What the image contains

Built `FROM erm42/yarp:pysis_xtb`, adding:

| Piece | Where | Notes |
|---|---|---|
| g-xTB build of xtb 6.7.1 (`140526`) | `/opt/gxtb`, symlink `/usr/local/bin/xtb-gxtb` | statically linked; downloaded at build time and verified by SHA-256 (`GXTB_SHA256`); GPL-3.0, license at `/opt/gxtb/LICENSE` |
| Pysisyphus `e531f04c` + `pysisyphus-gxtb.patch` | conda env `yarp_pysisyphus` | the patch adds `XTB(gxtb=False)`; with `gxtb: True` the calculator passes `--gxtb` instead of `--gfn <n>` |
| `/opt/gxtb/pysisyphusrc`, found through `ENV PYSISRC` | `[xtb] cmd=/usr/local/bin/xtb-gxtb` | sets the executable only; the method comes from YARP's `gxtb: True`. Found through `PYSISRC`, not `~/.pysisyphusrc`, so Apptainer (which runs as your user, with your home) reads it too |

## Running with g-xTB

Set `lot: gxtb` (or `gsm_lot: gxtb` for `ts_guess`) on the tasks that should
use it. YARP pulls `erm42/yarp_gxtb:v1` the first time it is needed, with
`docker pull` or, on HPC, `apptainer pull` into `job_manager.sif_location`.
If the image is already present locally, YARP uses it and does not pull.

An example input that runs the whole characterization pipeline at g-xTB:

```yaml
# Characterize one reaction (from target_rxn.pkl) entirely at g-xTB.
# Lines marked "# LOT" choose the level of theory; set them to xtb for GFN2-xTB.
# Run from the folder holding this file:  yarp-init characterize.yaml, then yarp-loop
initialize:
  verbose: True
  output: gxtb_char.pkl
  status: gxtb_char_STATUS.json
  initial_structure:
    source: target_rxn.pkl
    type: yarp_pickle
    mode: reaction
  job_manager:
    scheduler: local
    container: docker
    max_active_jobs: 2

stages:
  - ll_path
  - ll_refine

# Pre-opt -> CREST conformers -> GSM TS guesses
ll_path:
  method: init_rxn_path
  pre_opt:
    software: pysisyphus
    lot: gxtb               # LOT
    charge: 0
    multiplicity: 1
    opt_type: lbfgs
    max_cycles: 300
    n_cpus: 4
  conf_gen:
    software: crest
    lot: gfn2               # CREST has no g-xTB option
    charge: 0
    n_unpaired_electrons: 0
    energy_window: 6.0
    seed: 42
    n_cpus: 4
  ts_guess:
    software: pysisyphus
    gsm_lot: gxtb           # LOT
    charge: 0
    multiplicity: 1
    n_conf: 3
    max_gsm_nodes: 30
    joint_opt: dual
    bias_lot: uff
    n_cpus: 4
    max_runtime: "04:00:00"

# R/P optimization, TS optimization, IRC validation
ll_refine:
  method: refine_rxn_path
  initial_geom:
    reactant:
      label: conf_gen
      lot: gfn2
      software: crest
    product:
      label: conf_gen
      lot: gfn2
      software: crest
    transition_state:
      label: ts_guess
      lot: gxtb             # LOT (must match ts_guess.gsm_lot)
      software: pysisyphus
  rp_opt:
    software: pysisyphus
    lot: gxtb               # LOT
    charge: 0
    multiplicity: 1
    opt_type: rfo
    hessian_recalc: 3
    max_cycles: 300
    n_cpus: 4
  ts_opt:
    software: pysisyphus
    lot: gxtb               # LOT
    charge: 0
    multiplicity: 1
    conv_thresh: gau
    hessian_recalc: 3
    max_cycles: 300
    n_cpus: 4
    max_runtime: "04:00:00"
  irc_val:
    software: pysisyphus
    lot: gxtb               # LOT
    charge: 0
    multiplicity: 1
    conv_thresh: gau
    max_cycles: 300
    divert_unintended: False
    n_cpus: 4
    max_runtime: "04:00:00"
```

If the pull fails, build the image yourself as below.

## Building the image

```bash
cd containers/yarp_gxtb
docker build --platform linux/amd64 -t erm42/yarp_gxtb:v1 .
```

Tagging the build with the name in `PYSIS_GXTB_IMAGE` is what lets YARP find
it locally. If the g-xTB download location changes, add
`--build-arg GXTB_URL=<url>`; the checksum still guards the file.

### Apptainer (HPC)

YARP looks for `<sif_location>/erm42_yarp_gxtb_v1.sif`. Export the Docker
build and convert it:

```bash
docker save -o yarp_gxtb_v1.tar erm42/yarp_gxtb:v1           # where Docker is
apptainer build erm42_yarp_gxtb_v1.sif docker-archive://yarp_gxtb_v1.tar   # on the cluster
```

Then move the `.sif` into `job_manager.sif_location`.

## Checking the image

Docker:

```bash
docker run --rm --platform linux/amd64 erm42/yarp_gxtb:v1 bash -c '
  sha256sum /opt/gxtb/bin/xtb      # 1b4e30b68ed4e88b4075f60d97f4756ee440fe92d3294cc20826d53f8121cd26
  python -c "
from pysisyphus.config import get_cmd
from pysisyphus.calculators.XTB import XTB
print(get_cmd(\"xtb\"))                  # /usr/local/bin/xtb-gxtb
print(XTB(gfn=2).prepare_add_args())    # ... --gfn 2
print(XTB(gxtb=True).prepare_add_args()) # ... --gxtb
"'
```

Apptainer (from the folder holding the `.sif`):

```bash
apptainer exec erm42_yarp_gxtb_v1.sif sha256sum /opt/gxtb/bin/xtb
apptainer exec erm42_yarp_gxtb_v1.sif python -c "from pysisyphus.config import get_cmd; print(get_cmd('xtb'))"
```

`get_cmd` must print `/usr/local/bin/xtb-gxtb` in both. Any other path means
Pysisyphus would run the conda xtb, which does not know `--gxtb`.

