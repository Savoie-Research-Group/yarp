#!/usr/bin/env python3
"""JSON runner for YARP's OpenBabel/xTB joint optimization boundary."""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from openbabel import openbabel as ob


def _safe_dirname(label):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("._")
    return safe or "joint_opt"


def _geometry(elements, geo):
    geometry = np.asarray(geo, dtype=float)
    if geometry.shape != (len(elements), 3):
        raise ValueError(f"Expected geometry shape {(len(elements), 3)}, got {geometry.shape}")
    return geometry


def _bond_matrix(elements, target_bem):
    matrix = np.asarray(target_bem, dtype=float)
    if matrix.shape != (len(elements), len(elements)):
        raise ValueError(f"Expected target_bem shape {(len(elements), len(elements))}, got {matrix.shape}")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("target_bem must be symmetric")
    return matrix


def _build_ob_mol(elements, geometry, target_bem, formal_charges, radical_atoms):
    molecule = ob.OBMol()
    molecule.BeginModify()
    try:
        for atom_index, (element, xyz) in enumerate(zip(elements, geometry)):
            atomic_number = ob.GetAtomicNum(str(element).capitalize())
            if atomic_number <= 0:
                raise ValueError(f"Unknown element for OpenBabel: {element}")
            atom = molecule.NewAtom()
            atom.SetAtomicNum(atomic_number)
            atom.SetVector(*map(float, xyz))
            atom.SetFormalCharge(int(formal_charges[atom_index]))
            if atom_index in radical_atoms:
                atom.SetSpinMultiplicity(2)

        for atom_i in range(len(elements) - 1):
            for atom_j in range(atom_i + 1, len(elements)):
                if target_bem[atom_i, atom_j] <= 0:
                    continue
                bond_order = max(1, int(target_bem[atom_i, atom_j]))
                molecule.AddBond(atom_i + 1, atom_j + 1, bond_order)
    finally:
        molecule.EndModify()
    return molecule


def _ob_optimize(elements, geometry, target_bem, formal_charges, radical_atoms, options):
    molecule = _build_ob_mol(elements, geometry, target_bem, formal_charges, radical_atoms)
    requested = str(options.get("ff_name", "uff"))
    force_field = ob.OBForceField.FindForceField(requested) or ob.OBForceField.FindForceField("uff")
    if force_field is None:
        raise RuntimeError(f"OpenBabel force field not found: {requested} or uff")
    if not force_field.Setup(molecule):
        force_field = ob.OBForceField.FindForceField("uff")
        if force_field is None or not force_field.Setup(molecule):
            raise RuntimeError("Failed to set up OpenBabel force field for joint optimization")

    force_field.ConjugateGradients(500)
    force_field.GetCoordinates(molecule)
    return [
        [molecule.GetAtom(index).GetX(), molecule.GetAtom(index).GetY(), molecule.GetAtom(index).GetZ()]
        for index in range(1, molecule.NumAtoms() + 1)
    ]


def _write_xyz(path, elements, geometry):
    lines = [str(len(elements)), ""]
    lines.extend(
        f"{str(element).upper()} {xyz[0]:>12.8f} {xyz[1]:>12.8f} {xyz[2]:>12.8f}"
        for element, xyz in zip(elements, geometry)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_xcontrol(path, constraints, force_constant):
    with path.open("w", encoding="utf-8") as handle:
        for constraint in constraints:
            atom_i = int(constraint["atom_i"])
            atom_j = int(constraint["atom_j"])
            distance = float(constraint["distance"])
            handle.write("$constrain\n")
            handle.write(f"force constant={float(force_constant)}\n")
            handle.write(f"distance: {atom_i}, {atom_j}, {distance:.4f}\n")
            handle.write("$\n\n")


def _xtb_lot_flags(lot):
    lot = str(lot).lower()
    if lot == "gfnff":
        return ["--gfnff"]
    if lot == "gfn2":
        return ["--gfn", "2"]
    if lot == "gfn1":
        return ["--gfn", "1"]
    raise ValueError(f"Unsupported xTB joint optimization level: {lot}")


def _read_xyz(path, expected_elements):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError(f"Empty xyz output: {path}")
    atom_count = int(lines[0].strip())
    if atom_count != len(expected_elements) or len(lines) < atom_count + 2:
        raise ValueError(f"Unexpected xyz output shape in {path}")
    geometry = []
    for line in lines[2:atom_count + 2]:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Malformed xyz line in {path}: {line!r}")
        geometry.append([float(fields[1]), float(fields[2]), float(fields[3])])
    return geometry


def _xtb_optimize(elements, geometry, constraints, options, job_dir):
    namespace = "joint_opt"
    input_xyz = job_dir / "input.xyz"
    xcontrol = job_dir / "joint_opt.xcontrol"
    stdout_path = job_dir / "xtb.stdout"
    stderr_path = job_dir / "xtb.stderr"
    _write_xyz(input_xyz, elements, geometry)
    _write_xcontrol(xcontrol, constraints, options.get("force_constant", 1.0))

    multiplicity = int(options.get("multiplicity", 1))
    command = [
        "xtb",
        input_xyz.name,
        "--iterations",
        str(int(options.get("scf_iters", 300))),
        "--chrg",
        str(int(options.get("charge", 0))),
        "--uhf",
        str(max(multiplicity - 1, 0)),
        "--namespace",
        namespace,
        "--opt",
        "--parallel",
        str(max(int(options.get("n_cpus", 1)), 1)),
        "--input",
        xcontrol.name,
    ]
    command.extend(_xtb_lot_flags(options.get("lot", "gfn2")))
    result = subprocess.run(command, cwd=job_dir, capture_output=True, text=True)
    stdout_path.write_text(result.stdout or "", encoding="utf-8")
    stderr_path.write_text(result.stderr or "", encoding="utf-8")

    output_xyz = next(
        (path for path in (job_dir / f"{namespace}.xtbopt.xyz", job_dir / "xtbopt.xyz") if path.exists()),
        None,
    )
    combined_output = "\n".join((result.stdout or "", result.stderr or ""))
    if result.returncode != 0 or "GEOMETRY OPTIMIZATION CONVERGED" not in combined_output or output_xyz is None:
        detail = f"xTB exited with code {result.returncode}"
        if output_xyz is None:
            detail += "; optimized geometry was not written"
        if "GEOMETRY OPTIMIZATION CONVERGED" not in combined_output:
            detail += "; optimization did not converge"
        raise RuntimeError(detail)
    return _read_xyz(output_xyz, elements)


def _run_job(job, jobs_dir):
    label = job.get("label")
    engine = job.get("engine")
    elements = list(job.get("elements", []))
    geometry = _geometry(elements, job.get("geo"))
    target_bem = _bond_matrix(elements, job.get("target_bem"))
    formal_charges = [int(charge) for charge in job.get("formal_charges", [0] * len(elements))]
    radical_atoms = {int(atom_index) for atom_index in job.get("radical_atoms", [])}
    if len(formal_charges) != len(elements):
        raise ValueError("formal_charges must contain one entry per element")
    options = dict(job.get("options") or {})
    job_dir = jobs_dir / _safe_dirname(label)
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        if engine == "ob":
            optimized = _ob_optimize(elements, geometry, target_bem, formal_charges, radical_atoms, options)
        elif engine == "xtb":
            optimized = _xtb_optimize(elements, geometry, job.get("constraints") or [], options, job_dir)
        else:
            raise ValueError(f"Unsupported joint optimization engine: {engine}")
        result = {"label": label, "success": True, "geo": optimized}
    except Exception as exc:
        result = {"label": label, "success": False, "error": f"{type(exc).__name__}: {exc}"}

    if not options.get("keep_files", False):
        shutil.rmtree(job_dir, ignore_errors=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != 1:
        raise ValueError("Unsupported joint optimization protocol version")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("Joint optimization input must contain a jobs list")

    jobs_dir = args.output.parent / "jobs"
    results = [_run_job(job, jobs_dir) for job in jobs]
    args.output.write_text(json.dumps({"protocol_version": 1, "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
