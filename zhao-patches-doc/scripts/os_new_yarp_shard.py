#!/usr/bin/env python3
"""os_new_yarp_shard.py <shard_list.txt> --out <csv_path> [--timeout SEC]

For each slim zip path in the shard:
  - Extract <stem>/finished_first.xyz (reactant) and <stem>/finished_last.xyz (product)
  - Parse charge from filename: <stem>_<charge>_<mult>.zip
  - Run new-patched YARP on each side (via PYTHONPATH preset) to get BEM
  - Compute OS = el_valence[el] - int(bem_diag[i]) for d-block atoms
  - Emit one CSV row per archive: zip_path, reactant_OS, product_OS
    (OS string format matches transition_metal_oxidation_states.csv:
     'El0:N;El1:M;...')

This is READ-ONLY: it never modifies the source zips and writes only into
the --out CSV path under Scripts/v2/os_test_new_yarp/results/.

Set NEW_YARP_PATH env var (or rely on the sbatch script's default) to point
at the patched-new-YARP checkout.
"""
from __future__ import annotations
import argparse, contextlib, csv, os, re, sys, tempfile, time, zipfile
from pathlib import Path

# Resolve new yarp via env var; sbatch sets it to the Zhao-YARP checkout.
NEW_YARP_PATH = os.environ.get(
    "NEW_YARP_PATH",
    "/home/li1724/061226-YARP-again/Zhao-YARP/classy-yarp")
sys.path.insert(0, NEW_YARP_PATH)

# Silence yarp's import-time prints (RDKit logger warnings etc.)
with open(os.devnull, "w") as _dn, \
     contextlib.redirect_stdout(_dn), contextlib.redirect_stderr(_dn):
    import yarp as yp

# Full d-block: atomic numbers 21-30, 39-48, 57, 72-80
TRANSITION_METALS = {
    'sc','ti','v','cr','mn','fe','co','ni','cu','zn',
    'y','zr','nb','mo','tc','ru','rh','pd','ag','cd',
    'la',
    'hf','ta','w','re','os','ir','pt','au','hg',
}

EL_VALENCE = {
    'sc':3,'ti':4,'v':5,'cr':6,'mn':7,'fe':8,'co':9,'ni':10,'cu':11,'zn':12,
    'y':3,'zr':4,'nb':5,'mo':6,'tc':7,'ru':8,'rh':9,'pd':10,'ag':11,'cd':12,
    'la':3,
    'hf':4,'ta':5,'w':6,'re':7,'os':8,'ir':9,'pt':10,'au':11,'hg':12,
}

CHARGE_RE = re.compile(r".*_(-?\d+)_(\d+)$")


def parse_charge(stem: str):
    m = CHARGE_RE.match(stem)
    return int(m.group(1)) if m else None


def rewrite_xyz_with_q(src_bytes: bytes, charge: int, tmp_path: Path):
    """Read xyz bytes, replace comment line with 'q <charge>', write to tmp_path."""
    lines = src_bytes.decode("utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        raise ValueError("xyz too short")
    lines[1] = f"q {charge}"
    tmp_path.write_text("\n".join(lines) + "\n")


def compute_os_string(xyz_path: Path):
    """Run new yarpecule on the xyz and return 'El0:OS;El1:OS;...' for TM atoms."""
    with open(os.devnull, "w") as dn, \
         contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
        y = yp.yarpecule(str(xyz_path), canon=False)
    parts = []
    bem = y.bond_mats[0]
    for i, el in enumerate(y.elements):
        el_lc = str(el).lower()
        if el_lc not in TRANSITION_METALS:
            continue
        v = EL_VALENCE.get(el_lc)
        if v is None:
            continue
        e = int(bem[i, i])
        os_val = v - e
        # Title-case element symbol to match existing CSV: 'Ti12'
        sym = el_lc[:1].upper() + el_lc[1:]
        parts.append(f"{sym}{i}:{os_val}")
    return ";".join(parts)


def process_one(zip_path_str: str, timeout_s: int):
    """Returns (status, wall, zip_path, reactant_str, product_str, msg)."""
    zip_path = Path(zip_path_str)
    t0 = time.time()
    if not zip_path.exists():
        return ("FAIL", time.time()-t0, zip_path_str, "", "", "zip not found")

    stem = zip_path.stem
    charge = parse_charge(stem)
    if charge is None:
        return ("FAIL", time.time()-t0, zip_path_str, "", "",
                f"cannot parse charge from stem={stem}")

    work = Path(tempfile.mkdtemp(prefix=f"osnew_{os.getpid()}_", dir="/tmp"))
    r_str = p_str = ""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            try:
                r_bytes = zf.read(f"{stem}/finished_first.xyz")
            except KeyError:
                r_bytes = None
            try:
                p_bytes = zf.read(f"{stem}/finished_last.xyz")
            except KeyError:
                p_bytes = None
        if r_bytes is None and p_bytes is None:
            return ("FAIL", time.time()-t0, zip_path_str, "", "",
                    "no finished_first.xyz nor finished_last.xyz")

        if r_bytes is not None:
            r_xyz = work / "reactant.xyz"
            try:
                rewrite_xyz_with_q(r_bytes, charge, r_xyz)
                r_str = compute_os_string(r_xyz)
            except SystemExit:
                r_str = "SYSEXIT"
            except Exception as e:
                r_str = f"ERR:{type(e).__name__}"
        if p_bytes is not None:
            p_xyz = work / "product.xyz"
            try:
                rewrite_xyz_with_q(p_bytes, charge, p_xyz)
                p_str = compute_os_string(p_xyz)
            except SystemExit:
                p_str = "SYSEXIT"
            except Exception as e:
                p_str = f"ERR:{type(e).__name__}"

        wall = time.time() - t0
        # If both empty AND have legit data, this archive simply has no TMs
        # (unexpected since this is the TM-only dedup list, but possible).
        return ("OK", wall, zip_path_str, r_str, p_str, "")
    except Exception as e:
        return ("FAIL", time.time()-t0, zip_path_str, r_str, p_str,
                f"{type(e).__name__}: {e}")
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard", type=Path)
    ap.add_argument("--out", type=Path, required=True,
                    help="Output CSV path (one row per archive).")
    ap.add_argument("--timeout", type=int, default=120,
                    help="Per-archive timeout (currently informational; "
                         "yarpecule is in-process so we can't cleanly enforce).")
    args = ap.parse_args()

    paths = [ln.strip() for ln in args.shard.read_text().splitlines() if ln.strip()]
    print(f"shard={args.shard.name} archives={len(paths)} "
          f"new_yarp={NEW_YARP_PATH} out={args.out}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_ok = n_fail = 0
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["zip_path",
                    "reactant_metal_oxidation_states",
                    "product_metal_oxidation_states"])
        for p in paths:
            st, wall, path, r_s, p_s, msg = process_one(p, args.timeout)
            w.writerow([path, r_s, p_s])
            if st == "OK":
                n_ok += 1
                if not (n_ok % 100):
                    print(f"  ok={n_ok}/{len(paths)} elapsed={time.time()-t0:.0f}s",
                          flush=True)
            else:
                n_fail += 1
                print(f"FAIL {wall:.2f}s {path} :: {msg}", flush=True)
    print(f"done: ok={n_ok} fail={n_fail} elapsed={time.time()-t0:.0f}s", flush=True)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
