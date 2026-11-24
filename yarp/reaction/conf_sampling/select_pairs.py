"""
Logic for joint optimization, ASE alignment, and ML-based conformer selection.
"""
import copy
import numpy as np
import pickle
from pathlib import Path

from ase import Atoms
from ase.build import minimize_rotation_and_translation

from yarp.reaction.conf_sampling.joint_opt import joint_optimize
from yarp.reaction.conf_sampling.indicator import return_indicator
from yarp.reaction.external.calc_base import CalculatorInputError
from yarp.yarpecule.graph.adjacency import compare_adjacency

# Hard cutoffs (Zhao, Hsu & Savoie, JCTC 2022, 18, 3006, Table 1): a reaction
# geometry with any indicator beyond its threshold is discarded before the RF
# model is applied. (indicator, "max" or "min", threshold in Angstrom); a value
# equal to the threshold passes. The hydrogen variants (max_Hdis,
# min_Hcross_dis) have no cutoff.
HARD_CUTOFFS = (
    ("RMSD", "max", 1.8),
    ("max_dis", "max", 4.8),
    ("min_cross_dis", "min", 0.03),
    ("path_cross", "min", 0.20),
)


def passes_hard_cutoffs(indicators):
    """True if a one-row indicator DataFrame (return_indicator) is within every HARD_CUTOFFS threshold."""
    row = indicators.iloc[0]
    return all(row[name] <= thr if kind == "max" else row[name] >= thr
               for name, kind, thr in HARD_CUTOFFS)


def select_gsm_pairs(rxn, config):
    """
    Orchestrates Collapse Check -> Biasing -> Alignment -> Hard Cutoffs -> ML Tournament -> QC -> Pairing.

    Raises `CalculatorInputError` if every reactant or every product conformer
    has collapsed onto the other side's graph, if every conformer pair fails
    the hard cutoffs (HARD_CUTOFFS), or, with config.pair_selection "intended",
    if no pair scores at least config.pair_threshold; each discards the reaction.
    """
    # Select the conformer-generation output explicitly, rather than taking
    # "everything that is not initial_geom". The states also carry the xTB
    # pre-optimization's structure (and, on a second refinement pass, the
    # rp_opt geometries), none of which are CREST conformers -- sweeping them
    # in here would quietly seed GSM with the wrong structures.
    r_confs = [conf for key, conf in rxn.reactant.conformers.items() if "conf_gen" in key]
    p_confs = [conf for key, conf in rxn.product.conformers.items() if "conf_gen" in key]

    # A conformer whose bonds perceive to exactly the OTHER side's graph has
    # collapsed -- e.g. a product that relaxed back onto the reactant during the
    # pre-optimization, which only warns on a graph change. Pairing it gives GSM
    # nothing to do. This is decided here, per reaction, because conformers are
    # pooled across every reaction sharing a species and "the other side"
    # differs between them.
    r_confs, r_collapsed = drop_collapsed_conformers(r_confs, rxn.product.graph.adj_mat)
    p_confs, p_collapsed = drop_collapsed_conformers(p_confs, rxn.reactant.graph.adj_mat)
    if r_collapsed or p_collapsed:
        print(f"     ! [{rxn.hash}] Dropped conformers that collapsed onto the other side's graph: "
              f"{r_collapsed} reactant, {p_collapsed} product.")

    # An empty side cannot be paired. Stop here rather than let it reach the
    # loops below, where an empty reactant list surfaces as a NameError.
    for side, kept, n_collapsed, other in (("reactant", r_confs, r_collapsed, "product"),
                                           ("product", p_confs, p_collapsed, "reactant")):
        if not kept:
            raise CalculatorInputError(
                f"No usable {side} conformers for GSM "
                f"({n_collapsed} collapsed onto the {other} graph)."
            )

    # --- STEP A: Apply Joint Optimization (Biasing) ---
    lot = config.bias_lot
    mode = config.joint_opt.lower()
    verbose = getattr(config, "verbose", False)

    biased_r = r_confs
    biased_p = p_confs
    # biased_p: product geometries guided by reactant geometries
    # biased_r: reactant geometries guided by product geometries
    #
    # joint_optimize returns None when neither optimizer can produce a
    # geometry consistent with the target BEM;
    # such conformers are dropped, keeping r_confs/p_confs
    # aligned with their biased counterparts.
    if mode in ['dual', 'r_only']:
        kept_r_confs, biased_p = [], []
        for c in r_confs:
            b = joint_optimize(c, rxn.reactant.paired_bem, lot)
            if b is None:
                if verbose:
                    print(f"  + SKIPPED! Unable to bias reactant conformer toward product BEM")
                continue
            kept_r_confs.append(c)
            biased_p.append(b)
        r_confs = kept_r_confs
    if mode in ['dual', 'p_only']:
        kept_p_confs, biased_r = [], []
        for c in p_confs:
            b = joint_optimize(c, rxn.product.paired_bem, lot)
            if b is None:
                if verbose:
                    print(f"  + SKIPPED! Unable to bias product conformer toward reactant BEM")
                continue
            kept_p_confs.append(c)
            biased_r.append(b)
        p_confs = kept_p_confs

    # Default to "conformation-poor" model, unless an excess of conformers is present
    n_conf = config.n_conf
    total_conf = len(r_confs) + len(p_confs)
    if verbose:
        print("Total number of reactant + product conformers generated = ", total_conf)

    module_dir = Path(__file__).parent.resolve()
    if total_conf/n_conf > 3.0:
        model_path = module_dir / 'rich_model.sav'
    else:
        model_path = module_dir / 'poor_model.sav'
    if verbose:
        print(f"{model_path.stem} is chosen to generate aligned reaction conformers")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # --- STEP B: Reactant-Product Pair Evaluation & Tournament ---
    scored_pairs = []   # (pair, indicators) for every tournament winner with p > 0
    dropped_pairs = 0
    n_cut = 0   # conformers whose aligned and unaligned geometries both fail the hard cutoffs

    for ind, r_c in enumerate(r_confs):
         
            # 1. Unaligned Evaluation
            ind_unaligned = return_indicator(E=r_c.elements, RG=r_c.geo, PG=biased_p[ind].geo)
            prob_unaligned = model.predict_proba(ind_unaligned) if passes_hard_cutoffs(ind_unaligned) else None

            # 2. Aligned Evaluation
            aligned_biased_p = align_conformers(r_c, biased_p[ind])
            ind_aligned = return_indicator(E=r_c.elements, RG=r_c.geo, PG=aligned_biased_p.geo)
            prob_aligned = model.predict_proba(ind_aligned) if passes_hard_cutoffs(ind_aligned) else None

            # 3. The Tournament (keep the higher probability setup), between the
            # geometries within the hard cutoffs. If neither is, the conformer is dropped.
            if prob_unaligned is None and prob_aligned is None:
                n_cut += 1
                continue
            if prob_unaligned is None or (prob_aligned is not None and prob_aligned[0][1] > prob_unaligned[0][1]):
                best_p = aligned_biased_p
                best_ind = ind_aligned
                best_prob = prob_aligned
            else:
                best_p = biased_p[ind]
                best_ind = ind_unaligned
                best_prob = prob_unaligned

            # 4. Quality Control (deduplication runs after the sort, below)
            if best_prob[0][1] > 0.0:
                scored_pairs.append(({
                    "r_conf": r_c,
                    "p_conf": best_p,
                    "score": best_prob[0][1]
                }, best_ind))
            else:
                dropped_pairs += 1
####################################################################
    for ind, p_c in enumerate(p_confs):

            # 1. Unaligned Evaluation
            ind_unaligned = return_indicator(E=p_c.elements, RG=p_c.geo, PG=biased_r[ind].geo)
            prob_unaligned = model.predict_proba(ind_unaligned) if passes_hard_cutoffs(ind_unaligned) else None

            # 2. Aligned Evaluation
            aligned_biased_r = align_conformers(p_c, biased_r[ind])
            ind_aligned = return_indicator(E=p_c.elements, RG=p_c.geo, PG=aligned_biased_r.geo)
            prob_aligned = model.predict_proba(ind_aligned) if passes_hard_cutoffs(ind_aligned) else None

            # 3. The Tournament (keep the higher probability setup), between the
            # geometries within the hard cutoffs. If neither is, the conformer is dropped.
            if prob_unaligned is None and prob_aligned is None:
                n_cut += 1
                continue
            if prob_unaligned is None or (prob_aligned is not None and prob_aligned[0][1] > prob_unaligned[0][1]):
                best_r = aligned_biased_r
                best_ind = ind_aligned
                best_prob = prob_aligned
            else:
                best_r = biased_r[ind]
                best_ind = ind_unaligned
                best_prob = prob_unaligned

            # 4. Quality Control (deduplication runs after the sort, below)
            if best_prob[0][1] > 0.0:
                scored_pairs.append(({
                    "r_conf": best_r,
                    "p_conf": p_c,
                    "score": best_prob[0][1]
                }, best_ind))
            else:
                dropped_pairs += 1

    # Every conformer failed the hard cutoffs: no geometry is fit for GSM.
    if n_cut and n_cut == len(r_confs) + len(p_confs):
        raise CalculatorInputError(
            f"Every conformer pair failed the hard cutoffs ({n_cut} conformers)."
        )

    # --- STEP C: Sort, Deduplicate, and Select Top N ---
    # Sort descending by probability of success, then walk down the list keeping
    # each pair that is not a near-duplicate of one already kept, so that of two
    # similar pairs the higher-scoring one survives. Stop at n_conf.
    scored_pairs.sort(key=lambda x: x[0]["score"], reverse=True)

    # "intended" mode keeps only pairs at or above pair_threshold; with none,
    # no geometry is fit for GSM. "top_n" keeps the best pairs whatever their p.
    if config.pair_selection == "intended":
        scored_pairs = [x for x in scored_pairs if x[0]["score"] >= config.pair_threshold]
        if not scored_pairs:
            raise CalculatorInputError(
                f"No conformer pair reached pair_threshold (p >= {config.pair_threshold})."
            )
    approved_pairs = []
    approved_indicators = []
    for pair, ind in scored_pairs:
        if len(approved_pairs) == n_conf:
            break
        if check_uniqueness(ind, approved_indicators):
            approved_indicators.append(ind)
            approved_pairs.append(pair)
        else:
            dropped_pairs += 1

    if verbose:
        print("Number of approved pairs = ", len(approved_pairs))
        print("Number of dropped pairs = ", dropped_pairs)
        print("Number of conformers failing the hard cutoffs = ", n_cut)
    return approved_pairs


def drop_collapsed_conformers(confs, other_adj):
    """
    Removes conformers whose perceived bonding is exactly the other side's graph.

    The comparison is atom-for-atom against `other_adj` rather than by species
    hash, so a reaction whose reactant and product are the same species under
    different atom mappings is not mistaken for a collapse. Any other change of
    graph is left alone: the product pre-optimization deliberately keeps a
    product that is not a minimum of its enumerated graph.

    Returns (kept conformers, number dropped).
    """
    kept = [conf for conf in confs if not compare_adjacency(conf.elements, conf.geo, other_adj)[0]]
    return kept, len(confs) - len(kept)


def align_conformers(conf, biased_conf):
    """
    Uses ASE to minimize rotation and translation (RMSD) between product and reactant.
    Returns a NEW product conformer that is aligned to the reactant.
    """
    atoms = Atoms(symbols=[el.upper() for el in conf.elements], positions=conf.geo)
    biased_atoms = Atoms(symbols=[el.upper() for el in conf.elements], positions=biased_conf.geo)

    # ASE modifies the moving atoms (p_atoms) in-place to align with the target (r_atoms)
    minimize_rotation_and_translation(atoms, biased_atoms)

    aligned_biased_conf = copy.deepcopy(biased_conf)
    aligned_biased_conf.geo = biased_atoms.positions # Extract the rotated/translated coordinates
    aligned_biased_conf.type = f"aligned_{biased_conf.type}"

    return aligned_biased_conf


def check_uniqueness(new_indicators, approved_indicators_list, threshold=0.025):
    """
    Checks if a pair is too geometrically similar to an already approved pair.
    """
    if len(approved_indicators_list) == 0: return True

    min_dis = min([np.linalg.norm(np.array(new_indicators) - np.array(j)) for j in approved_indicators_list])

    if min_dis > threshold: return True
    else: return False
