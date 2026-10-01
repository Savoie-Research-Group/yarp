# RDKit conformer generation: what happens when it fails

For discussion at the YARP meeting. Branch: `rdkit_conf`.

`software: rdkit` in a `conf_gen` block now runs `RdkitConfCalculator`
(`yarp/reaction/external/conf_gen.py`). It is a port of classy_yarp's
`conf_rdkit()`, with force-field optimization, energy ranking and
duplicate removal added. This note covers the cases where it produces no
usable conformers, because the current behavior differs from classy_yarp's.

## Current behavior

Every failure below removes the **whole reaction** from the network. It is
written to `failed_rxns.pkl` and `failed_status.json` in the work directory,
with the error and the scratch directory, and never reaches GSM.

| Where it fails | Cause | Recorded as |
|---|---|---|
| Before submission | No xTB pre-optimized geometry for this side (same check as CREST) | `Pre-flight check failed` |
| In the container | RDKit embeds zero conformers | `Output validation failed.` |
| In the container | `lot: mmff94` on a molecule MMFF94 has no parameters for (common for radicals) | `Output validation failed.` |
| On the host, after the job | **No conformer reproduces the state's bonding** (classy_yarp's connectivity filter) | `Data scraping failed.` |

The container's own reason (for example "MMFF94 has no parameters ... try
lot: uff") is printed from `rdkit_run.err` in the scratch directory.

If *some* conformers fail the connectivity check, they are dropped, a line
says how many, and the reaction continues with the rest.

## How classy_yarp differed

classy_yarp did not fail the reaction. When a side had no conformers left,
`rxn_conf_generation()` printed

    Warning: No conformers for product. Just use input geometry

and carried on with the input geometry as the only conformer
(`classy_yarp/reaction/wrappers/reaction.py`, lines 155-166).

The port currently fails instead, to match how a failed CREST job is handled.

## Inconsistency worth knowing about

CREST conformers are **not** connectivity-checked at all; RDKit conformers
are. So for the same molecule, RDKit can reject a reaction that CREST would
pass through.

## Questions for the meeting

1. When no RDKit conformer keeps the bonding, should the reaction:
   - fail, as now;
   - fall back to the xTB pre-optimized geometry as a single conformer (the
     closest match to classy_yarp's behavior); or
   - fall back to CREST for that species?
2. Should CREST output get the same connectivity filter, so both options apply
   the same standard?
3. For `lot: mmff94` on a molecule without MMFF94 parameters: fail with a
   message (as now), or switch to UFF automatically and say so?
4. Do the duplicate-removal cutoffs below look reasonable?

## Duplicate removal (for question 4)

Force-field optimization sends different starting embeddings into the same
minimum. Two optimized conformers count as duplicates when **both** their
energies match (ΔE < 1e-3 kcal/mol) **and** their symmetry-aware RMSD is
under 0.1 Å.

Energy alone is not enough: mirror-image conformers have identical
force-field energies but are different geometries.

Measured on KHP (`O=CCCOO`) and its `O=C=CCOO.[H][H]` product under UFF, 96
conformers in total:

- True duplicates had an RMSD of at most 0.003 Å, with ΔE < 1e-4 kcal/mol.
- The closest pair of genuinely distinct conformers was 0.434 Å apart.

The cutoffs sit well inside that gap. 0.1 Å is also the value classy_yarp
used to prune conformers before embedding.

Result on the same two runs:

- Reactant: 46 conformers were reduced to 32.
- Product: 50 conformers were reduced to 42.

Every pair left with matching energies is at least 0.434 Å apart. So only
true duplicates were removed, and the mirror-image pairs were kept.

The cutoffs are constants in `containers/rdkit_conf/run_conf_gen.py`
(`DEDUP_ENERGY_TOL`, `DEDUP_RMSD_TOL`), not YAML settings.
