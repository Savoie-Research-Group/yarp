"""
RDKit conformer generation, run inside the rdkit_conf container.

Reads a MOL file written by RdkitConfCalculator (built from the state's
yarpecule via yarpecule_to_rdmol), embeds conformers with the ETKDG settings
classy_yarp used, force-field optimizes them, removes the duplicates that
optimization creates, and writes them out sorted by energy. The connectivity
check against the yarpecule graph is done afterwards on the host, in
RdkitConfCalculator.scrape_data, so this script stays free of YARP imports.

Outputs (in the working directory):
    rdkit_conformers.xyz  every unique optimized conformer, lowest energy first
    rdkit.energies        one line per conformer: rank, energy (kcal/mol), converged (1/0)
"""
import argparse
import sys

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

TERMINATION_MSG = "RDKit conformer generation terminated normally."

# classy_yarp's own force-field optimizer used 1000 steps; RDKit's default of
# 200 leaves many strained embeddings unconverged.
MAX_ITERS = 1000

# Two optimized conformers are duplicates when their energies match AND their
# symmetry-aware RMSD is small. Energy alone is not enough: mirror-image
# conformers have identical force-field energies but are distinct geometries.
# Measured on KHP (O=CCCOO) and its O=C=CCOO.[H][H] product under UFF, 96
# conformers: true duplicates sat at RMSD <= 0.003 A with dE < 1e-4 kcal/mol;
# the closest distinct pair was 0.434 A apart. Both cutoffs leave wide margins.
DEDUP_ENERGY_TOL = 1e-3   # kcal/mol
DEDUP_RMSD_TOL = 0.1      # Angstrom; same value as classy_yarp's pruneRmsThresh


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mol_file")
    parser.add_argument("--lot", choices=["uff", "mmff94"], required=True)
    parser.add_argument("--n_conf", type=int, required=True)
    parser.add_argument("--prune_rms_thresh", type=float, required=True)
    parser.add_argument("--seed", type=int, default=-1, help="-1 lets RDKit pick a random seed")
    parser.add_argument("--n_threads", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    mol = Chem.MolFromMolFile(args.mol_file, removeHs=False)
    if mol is None:
        sys.exit(f"RDKit could not read {args.mol_file}")

    # Embedding parameters are classy_yarp's conf_rdkit(), unchanged.
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

    # Interfragment interactions stay on, as in yarp.util.rdkit.rdkit_ff_opt:
    # with them off, the fragments of a multi-species state can drift into
    # each other, and the host-side connectivity check would then reject them.
    if args.lot == "uff":
        if not AllChem.UFFHasAllMoleculeParams(mol):
            sys.exit("UFF has no parameters for at least one atom in this molecule.")
        results = AllChem.UFFOptimizeMoleculeConfs(
            mol, numThreads=args.n_threads, maxIters=MAX_ITERS,
            ignoreInterfragInteractions=False,
        )
    else:
        if not AllChem.MMFFHasAllMoleculeParams(mol):
            sys.exit("MMFF94 has no parameters for at least one atom in this molecule; try lot: uff.")
        results = AllChem.MMFFOptimizeMoleculeConfs(
            mol, numThreads=args.n_threads, maxIters=MAX_ITERS,
            ignoreInterfragInteractions=False,
        )

    # results[i] is (not_converged, energy) for the i-th conformer on the mol,
    # which is the same order as conf_ids.
    ranked = sorted(
        ((energy, not_converged, conf_id) for (not_converged, energy), conf_id in zip(results, conf_ids)),
        key=lambda item: item[0],
    )
    ranked = remove_duplicates(mol, ranked)

    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    with open("rdkit_conformers.xyz", "w") as xyz, open("rdkit.energies", "w") as ene:
        for rank, (energy, not_converged, conf_id) in enumerate(ranked):
            positions = mol.GetConformer(conf_id).GetPositions()
            xyz.write(f"{len(symbols)}\n")
            xyz.write(f"rank {rank} energy {energy:.6f} kcal/mol\n")
            for symbol, (x, y, z) in zip(symbols, positions):
                xyz.write(f"{symbol} {x:.8f} {y:.8f} {z:.8f}\n")
            ene.write(f"{rank} {energy:.6f} {int(not_converged == 0)}\n")

    print(f"Embedded {len(conf_ids)} conformers, optimized with {args.lot}; "
          f"wrote {len(ranked)} unique conformers lowest energy first.")
    print(TERMINATION_MSG)


def remove_duplicates(mol, ranked):
    """
    Drop conformers that optimized into the same minimum as a lower-energy one.

    `ranked` is (energy, not_converged, conf_id) sorted by energy, so each
    conformer only needs comparing against those already kept, and the RMSD is
    only computed for pairs whose energies already match.
    """
    # GetBestRMS aligns the probe conformer in place; align a copy so the
    # coordinates written out are the optimizer's, untouched.
    probe = Chem.Mol(mol)
    kept = []
    for energy, not_converged, conf_id in ranked:
        duplicate = any(
            energy - kept_energy < DEDUP_ENERGY_TOL
            and rdMolAlign.GetBestRMS(probe, mol, prbId=conf_id, refId=kept_id) < DEDUP_RMSD_TOL
            for kept_energy, _, kept_id in kept
        )
        if not duplicate:
            kept.append((energy, not_converged, conf_id))
    return kept


if __name__ == "__main__":
    main()
