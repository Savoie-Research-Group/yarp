"""
Auto3D conformer generation, run inside the auto3d_conf container (Auto3D 3.2.2).

Reads a MOL file written by the YARP calculator (built from the state's
yarpecule via yarpecule_to_rdmol, so atom order and formal charges are YARP's),
then:
  1. embeds conformers with exactly the settings the rdkit_conf container
     uses (classy_yarp's ETKDG settings, YARP's n_conf / prune_rms_thresh);
  2. separates the fragments of a multi-species state (same placement as
     rdkit_conf);
  3. optimizes them with Auto3D's neural-network potential, through Auto3D's
     public `opt_geometry`, with the spin multiplicity YARP is configured with;
  4. ranks them with Auto3D's `ConformerRanker`, which drops unconverged and
     bond-changed structures and removes duplicates;
and writes the survivors sorted by energy. So the only difference from the
rdkit option is the optimizer and Auto3D's ranking.

Auto3D's `main` is not used: it re-embeds every input record with its own
settings, and for a multi-species state optimizes fragments that overlap,
which loses every conformer (measured with Auto3D 2.3.1 and 3.2.2).
Optimization and ranking use Auto3D's own pipeline defaults
(Auto3D.foundation.constants), so they behave as in `main`.

The connectivity check against the yarpecule graph is done afterwards on the
host, in the calculator's scrape_data, so this script stays free of YARP imports.

Outputs (in the working directory):
    auto3d_conformers.xyz  every kept conformer, lowest energy first
    auto3d.energies        one line per conformer: rank, energy (Hartree), E_rel (kcal/mol)
"""
import argparse
import os
import sys

# Before Auto3D (and through it aimnet / NVIDIA Warp) is imported: Warp compiles
# its kernels into a cache, by default under $HOME. Keep it in the job directory
# instead -- the image is read-only, and recompiling costs ~3 s per job
# (measured).
os.environ.setdefault("WARP_CACHE_PATH", os.path.abspath("warp_cache"))

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from Auto3D import opt_geometry
from Auto3D.domain.ranking import ConformerRanker
from Auto3D.foundation.constants import (
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_DUPLICATE_ENERGY_TOL,
    DEFAULT_OPT_STEPS,
    DEFAULT_PATIENCE,
    DEFAULT_RMSD_THRESHOLD,
)

TERMINATION_MSG = "Auto3D conformer generation terminated normally."

# YARP lot -> Auto3D model name. aimnet2-nse is the spin-aware AIMNet2, the
# only one valid for open-shell states.
ENGINES = {"aimnet2": "aimnet2", "aimnet2-nse": "aimnet2-nse", "ani2x": "ANI2x"}

# Same value and meaning as containers/rdkit_conf/run_conf_gen.py.
FRAGMENT_CONTACT = 3.0

# Auto3D names conformers <species>_<isomer>_<conformer> and groups by species.
SPECIES = "mol"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mol_file")
    parser.add_argument("--lot", choices=sorted(ENGINES), required=True)
    parser.add_argument("--n_conf", type=int, required=True)
    parser.add_argument("--prune_rms_thresh", type=float, required=True)
    parser.add_argument("--multiplicity", type=int, default=1,
                        help="spin multiplicity of the whole state, passed to Auto3D explicitly")
    parser.add_argument("--seed", type=int, default=-1, help="-1 lets RDKit pick a random seed")
    parser.add_argument("--n_threads", type=int, default=1)
    parser.add_argument("--frag_contact", type=float, default=FRAGMENT_CONTACT,
                        help="closest atom-atom distance (A) between fragments before optimization")
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    torch.set_num_threads(args.n_threads)

    mol = Chem.MolFromMolFile(args.mol_file, removeHs=False)
    if mol is None:
        sys.exit(f"RDKit could not read {args.mol_file}")

    # Embedding parameters are classy_yarp's conf_rdkit(), unchanged, as in rdkit_conf.
    conf_ids = list(AllChem.EmbedMultipleConfs(
        mol,
        numConfs=args.n_conf,
        maxAttempts=1000000,
        randomSeed=args.seed,
        useRandomCoords=True,
        pruneRmsThresh=args.prune_rms_thresh,
        useExpTorsionAnglePrefs=False,
        useBasicKnowledge=True,
        enforceChirality=False,
        numThreads=args.n_threads,
    ))
    if not conf_ids:
        sys.exit("EmbedMultipleConfs produced no conformers.")

    fragments = Chem.GetMolFrags(mol)
    if len(fragments) > 1:
        for conf_id in conf_ids:
            conf = mol.GetConformer(conf_id)
            placed = separate_fragments(conf.GetPositions(), fragments, args.frag_contact)
            for idx, (x, y, z) in enumerate(placed):
                conf.SetAtomPosition(idx, Point3D(float(x), float(y), float(z)))

    # Auto3D's stages exchange SDF files. An explicit multiplicity property
    # takes precedence over Auto3D's own guess from unpaired electrons, so the
    # conformers are made in the spin state the rest of the YARP run uses.
    with Chem.SDWriter("auto3d_embedded.sdf") as writer:
        for conf_id in conf_ids:
            mol.SetProp("_Name", f"{SPECIES}_0_{conf_id}")
            mol.SetIntProp("multiplicity", args.multiplicity)
            writer.write(mol, confId=conf_id)

    opt_geometry("auto3d_embedded.sdf", ENGINES[args.lot], use_gpu=False,
                 opt_tol=DEFAULT_CONVERGENCE_THRESHOLD, opt_steps=DEFAULT_OPT_STEPS,
                 patience=DEFAULT_PATIENCE, out_path="auto3d_optimized.sdf")
    # k = every embedded conformer: keep all unique survivors, not a top few.
    kept = ConformerRanker("auto3d_optimized.sdf", "auto3d_ranked.sdf", DEFAULT_RMSD_THRESHOLD,
                           k=len(conf_ids), duplicate_energy_tol=DEFAULT_DUPLICATE_ENERGY_TOL).run()
    if not kept:
        sys.exit("No conformer survived Auto3D's ranking (all unconverged or bond-changed).")

    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    with open("auto3d_conformers.xyz", "w") as xyz, open("auto3d.energies", "w") as ene:
        for rank, kept_mol in enumerate(kept):
            energy = float(kept_mol.GetProp("E_tot"))
            e_rel = float(kept_mol.GetProp("E_rel(kcal/mol)"))
            xyz.write(f"{len(symbols)}\n")
            xyz.write(f"rank {rank} energy {energy:.8f} Eh\n")
            for symbol, (x, y, z) in zip(symbols, kept_mol.GetConformer().GetPositions()):
                xyz.write(f"{symbol} {x:.8f} {y:.8f} {z:.8f}\n")
            ene.write(f"{rank} {energy:.8f} {e_rel:.6f}\n")

    print(f"Embedded {len(conf_ids)} conformers, optimized with {args.lot} "
          f"(multiplicity {args.multiplicity}); wrote {len(kept)} unique conformers lowest energy first.")
    print(TERMINATION_MSG)


# Copied verbatim from containers/rdkit_conf/run_conf_gen.py, so both options
# place fragments identically. test/containers/test_run_conf_gen.py checks the
# two copies stay the same.
def separate_fragments(positions, fragments, contact):
    """
    Rigidly translate fragments apart so each sits `contact` from the others.

    Fragment 0 stays put. Each later fragment, in order, is moved onto the
    centroid of the atoms already placed and pushed out along the line from that
    centroid to its own embedded centroid, until its closest atom to any placed
    atom is `contact` away. Only translations are applied, so every fragment's
    internal geometry, and its orientation from the embedding, is unchanged.
    No randomness: the result depends only on the input coordinates.

    positions : (N, 3) array; fragments : tuples of atom indices, as from
    Chem.GetMolFrags. Returns a new (N, 3) array.
    """
    positions = np.array(positions, dtype=float)
    placed = list(fragments[0])
    for frag in fragments[1:]:
        frag = list(frag)
        anchor = positions[placed].mean(axis=0)
        centroid = positions[frag].mean(axis=0)
        axis = centroid - anchor
        norm = np.linalg.norm(axis)
        axis = axis / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        start = positions[frag] - centroid + anchor

        def gap(shift):
            moved = start + shift * axis
            return np.linalg.norm(positions[placed][:, None] - moved[None], axis=-1).min()

        # At `far` the two atom sets cannot be within `contact` of each other.
        far = (np.linalg.norm(positions[placed] - anchor, axis=1).max()
               + np.linalg.norm(start - anchor, axis=1).max() + contact + 1.0)
        # Bisect to the gap == contact crossing. Any crossing will do: every
        # placed atom is then at least `contact` from this fragment, with the
        # closest one exactly at it. (If the fragment already clears `contact`
        # sitting on the anchor, this converges to a shift of 0 and leaves it there.)
        near = 0.0
        for _ in range(60):
            mid = 0.5 * (near + far)
            if gap(mid) < contact:
                near = mid
            else:
                far = mid
        positions[frag] = start + far * axis
        placed += frag
    return positions


if __name__ == "__main__":
    main()
