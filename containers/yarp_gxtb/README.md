# yarp_gxtb container

Pysisyphus + g-xTB for YARP tasks run with `lot: gxtb` (pre-opt, GSM, R/P opt,
TS opt, IRC). Tasks with `lot: xtb` do **not** use this image; they keep
`erm42/yarp:pysis_xtb` (stock pysisyphus + conda xtb). The routing lives in
`yarp/reaction/external/calc_base.py` (`pysis_image`, `PYSIS_GXTB_IMAGE`).

## What the image contains

Built `FROM erm42/yarp:pysis_xtb`, adding:

| Piece | Where | Notes |
|---|---|---|
| g-xTB build of xtb 6.7.1 (`140526`) | `/opt/gxtb`, symlink `/usr/local/bin/xtb-gxtb` | statically linked; downloaded at build time and verified by SHA-256 (`GXTB_SHA256`); GPL-3.0, license at `/opt/gxtb/LICENSE` |
| pysisyphus `e531f04c` + `pysisyphus-gxtb.patch` | conda env `yarp_pysisyphus` | the patch adds `XTB(gxtb=False)`; with `gxtb: True` the calculator passes `--gxtb` instead of `--gfn <n>` |
| `/root/.pysisyphusrc` | `[xtb] cmd=/usr/local/bin/xtb-gxtb` | sets the executable only; the method comes from YARP's `gxtb: True` |

## Build

```bash
cd containers/yarp_gxtb
docker build --platform linux/amd64 -t erm42/yarp_gxtb:v1 .
```

If the g-xTB download location changes, pass `--build-arg GXTB_URL=<url>`; the
checksum still guards the file.

## Smoke tests

```bash
IMG=erm42/yarp_gxtb:v1
docker run --rm --platform linux/amd64 $IMG bash -c '
  sha256sum /opt/gxtb/bin/xtb      # 1b4e30b68ed4e88b4075f60d97f4756ee440fe92d3294cc20826d53f8121cd26
  cat /root/.pysisyphusrc          # [xtb] cmd=/usr/local/bin/xtb-gxtb
  python -c "
from pysisyphus.calculators.XTB import XTB
print(XTB(gfn=2).prepare_add_args())    # ... --gfn 2
print(XTB(gxtb=True).prepare_add_args()) # ... --gxtb
"'
```

## Publish

YARP pulls missing images from Docker Hub (`docker pull`, or
`apptainer pull docker://...` into `job_manager.sif_location` on HPC), so the
tag in `PYSIS_GXTB_IMAGE` must exist there.

```bash
docker push erm42/yarp_gxtb:v1
```

**Never re-push an existing tag with different contents.** YARP only pulls an
image (or `.sif`) that is missing locally, so users who already have `v1` would
keep the old one. Rebuild as `v2`, push, and update `PYSIS_GXTB_IMAGE`.

## Using a local build before it is published

```bash
docker tag <your local image> erm42/yarp_gxtb:v1
```

YARP finds the image locally and skips the pull. On Apptainer systems, build or
copy a `.sif` named `erm42_yarp_gxtb_v1.sif` into `job_manager.sif_location`.
