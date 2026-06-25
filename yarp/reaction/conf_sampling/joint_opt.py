import copy
import numpy as np
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple
from openbabel import openbabel as ob
from yarp.yarpecule.input_parsers import xyz_parse
from yarp.util.properties import el_radii
from yarp.util.write_files import mol_write_yp
from yarp.util.write_files import xyz_write


class XTBConstraint(NamedTuple):
    atom_i: int
    atom_j: int
    distance: float

def ob_joint_optimize(conformer, target_bem, ff_name="uff"):
    """
    Applies constraints based on the target BEM and runs an OpenBabel FF optimization.
    Returns a NEW conformer object with the biased geometry.
    """
    obMol = build_ob_mol(conformer.elements, conformer.geo, target_bem)
    ff = ob.OBForceField.FindForceField(ff_name) or ob.OBForceField.FindForceField("uff")
    if ff is None:
        raise RuntimeError(f"OpenBabel force field not found: {ff_name} or uff")

    if not ff.Setup(obMol):
        ff = ob.OBForceField.FindForceField("uff")
        if ff is None or not ff.Setup(obMol):
            raise RuntimeError("Failed to set up OpenBabel force field for joint optimization")

    ff.ConjugateGradients(500)
    ff.GetCoordinates(obMol)

    biased_conf = copy.deepcopy(conformer)
    biased_conf.geo = np.array([[obMol.GetAtom(i).GetX(), obMol.GetAtom(i).GetY(), obMol.GetAtom(i).GetZ()]
                                for i in range(1, obMol.NumAtoms() + 1)])
    biased_conf.type = f"biased_{conformer.type}"

    return biased_conf


def bem_to_distance_constraints(elements, target_bem):
    """
    Convert a target BEM into xTB distance constraints.

    Any positive off-diagonal BEM entry is treated as a target bond. Distances
    follow the legacy xTB joint-optimization behavior: summed covalent radii.
    Atom indices are converted to xTB's 1-indexed convention.
    """
    target_bem = np.asarray(target_bem)
    if target_bem.ndim != 2 or target_bem.shape[0] != target_bem.shape[1]:
        raise ValueError(f"target_bem must be a square matrix, got shape {target_bem.shape}")
    if len(elements) != target_bem.shape[0]:
        raise ValueError(
            f"elements length ({len(elements)}) does not match target_bem size ({target_bem.shape[0]})"
        )

    constraints = []
    for i in range(target_bem.shape[0] - 1):
        for j in range(i + 1, target_bem.shape[1]):
            if target_bem[i, j] <= 0:
                continue

            radius_i = el_radii.get(elements[i], el_radii.get(str(elements[i]).capitalize()))
            radius_j = el_radii.get(elements[j], el_radii.get(str(elements[j]).capitalize()))
            if radius_i is None or radius_j is None:
                raise KeyError(f"Missing covalent radius for constrained pair {elements[i]}-{elements[j]}")

            constraints.append(XTBConstraint(i + 1, j + 1, float(radius_i + radius_j)))

    return constraints


def write_xtb_xcontrol(path, constraints, force_constant=1.0):
    """
    Write xTB distance constraints using the legacy one-block-per-distance format.
    """
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for constraint in constraints:
            f.write("$constrain\n")
            f.write(f"force constant={force_constant}\n")
            f.write(f"distance: {constraint.atom_i}, {constraint.atom_j}, {constraint.distance:.4f}\n")
            f.write("$\n\n")


def _xtb_lot_flags(lot):
    lot = lot.lower()
    if lot == "gfnff":
        return ["--gfnff"]
    if lot == "gfn2":
        return ["--gfn", "2"]
    if lot == "gfn1":
        return ["--gfn", "1"]
    raise ValueError(f"Unsupported xTB joint optimization level: {lot}")


def _build_xtb_command(
    input_xyz,
    xcontrol,
    *,
    namespace,
    lot="gfn2",
    xtb_path="xtb",
    charge=0,
    multiplicity=1,
    n_cpus=1,
    scf_iters=300,
    solvent=None,
    solvation_model="alpb",
):
    cmd = [
        xtb_path,
        Path(input_xyz).name,
        "--iterations",
        str(scf_iters),
        "--chrg",
        str(charge),
        "--uhf",
        str(max(int(multiplicity) - 1, 0)),
        "--namespace",
        namespace,
        "--opt",
        "--parallel",
        str(n_cpus),
        "--input",
        Path(xcontrol).name,
    ]
    cmd.extend(_xtb_lot_flags(lot))

    if solvent:
        model = solvation_model.lower()
        if model not in {"alpb", "gbsa"}:
            raise ValueError(f"Unsupported xTB solvation model: {solvation_model}")
        cmd.extend([f"--{model}", str(solvent)])

    return cmd


def _xtb_converged(stdout, stderr, output_path=None):
    text = "\n".join([stdout or "", stderr or ""])
    if output_path is not None and Path(output_path).exists():
        text += "\n" + Path(output_path).read_text(encoding="utf-8", errors="replace")
    return "GEOMETRY OPTIMIZATION CONVERGED" in text


def _find_xtbopt_xyz(work_dir, namespace):
    work_dir = Path(work_dir)
    for candidate in [work_dir / f"{namespace}.xtbopt.xyz", work_dir / "xtbopt.xyz"]:
        if candidate.exists():
            return candidate
    return None


def xtb_joint_optimize(
    conformer,
    target_bem,
    scratch_dir,
    *,
    lot="gfn2",
    xtb_path="xtb",
    charge=0,
    multiplicity=1,
    n_cpus=1,
    force_constant=1.0,
    scf_iters=300,
    solvent=None,
    solvation_model="alpb",
    keep_files=False,
    runner=subprocess.run,
):
    """
    Applies target-BEM distance constraints and runs xTB geometry optimization.

    Returns a NEW conformer object with the biased geometry. If xTB fails or
    does not converge, returns None so pair selection can skip the candidate.
    """
    work_dir = Path(scratch_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    namespace = "joint_opt"
    input_xyz = work_dir / "input.xyz"
    xcontrol = work_dir / "joint_opt.xcontrol"
    stdout_path = work_dir / "xtb.stdout"
    stderr_path = work_dir / "xtb.stderr"

    constraints = bem_to_distance_constraints(conformer.elements, target_bem)
    xyz_write(str(input_xyz), conformer.elements, conformer.geo)
    write_xtb_xcontrol(xcontrol, constraints, force_constant=force_constant)

    cmd = _build_xtb_command(
        input_xyz,
        xcontrol,
        namespace=namespace,
        lot=lot,
        xtb_path=xtb_path,
        charge=charge,
        multiplicity=multiplicity,
        n_cpus=n_cpus,
        scf_iters=scf_iters,
        solvent=solvent,
        solvation_model=solvation_model,
    )

    result = runner(cmd, cwd=work_dir, capture_output=True, text=True)
    stdout_path.write_text(result.stdout or "", encoding="utf-8")
    stderr_path.write_text(result.stderr or "", encoding="utf-8")

    opt_xyz = _find_xtbopt_xyz(work_dir, namespace)
    if result.returncode != 0 or not _xtb_converged(result.stdout, result.stderr) or opt_xyz is None:
        if not keep_files:
            shutil.rmtree(work_dir, ignore_errors=True)
        return None

    _, opt_geo = xyz_parse(str(opt_xyz), multiple=False)
    biased_conf = copy.deepcopy(conformer)
    biased_conf.geo = opt_geo
    biased_conf.type = f"biased_xtb_{conformer.type}"
    biased_conf.lot = lot
    biased_conf.software = "xtb"

    if not keep_files:
        shutil.rmtree(work_dir, ignore_errors=True)

    return biased_conf


def bondmat_to_adjmat(bond_mat):
    adj_mat = copy.deepcopy(bond_mat)
    for count_i, i in enumerate(bond_mat):
        for count_j, j in enumerate(i):
            if j and count_i != count_j:
                adj_mat[count_i][count_j] = 1.0
            if count_i == count_j:
                adj_mat[count_i][count_i] = 0.0
    return adj_mat


def build_ob_mol(elements, coords, bond_mat):
    """
    Builds an OBMol object using explicit BEM bonding info.
    """
    adj_mat = bondmat_to_adjmat(bond_mat)
    obMol = ob.OBMol()
    conv = ob.OBConversion()
    conv.SetInFormat("mol")

    import os
    import tempfile

    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mol", delete=False) as tmp:
            tmp_file = tmp.name
        mol_write_yp(tmp_file, elements, coords, bond_mat, adj_mat)
        conv.ReadFile(obMol, tmp_file)
    finally:
        if tmp_file is not None:
            os.unlink(tmp_file)

    return obMol
