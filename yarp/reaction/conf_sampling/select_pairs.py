"""
Logic for joint optimization, ASE alignment, and ML-based conformer selection.
"""
import copy
import numpy as np
import pickle
from pathlib import Path

from ase import Atoms
from ase.build import minimize_rotation_and_translation

from yarp.reaction.conf_sampling.joint_opt import ob_joint_optimize
from yarp.reaction.conf_sampling.indicator import return_indicator


def select_gsm_pairs(rxn, config):
    """
    Orchestrates Biasing -> Alignment -> ML Tournament -> QC -> Pairing.
    """
    r_confs = [conf for key, conf in rxn.reactant.conformers.items() if key != "initial_geom"] 
    p_confs = [conf for key, conf in rxn.product.conformers.items() if key != "initial_geom"]

    # --- STEP A: Apply Joint Optimization (Biasing) ---
    lot = config.bias_lot
    mode = config.joint_opt.lower()

    biased_r = r_confs
    biased_p = p_confs
    # biased_p: product geometries guided by reactant geometries
    # biased_r: reactant geometries guided by product geometries
    if mode in ['dual', 'r_only']:
        biased_p = [ob_joint_optimize(c, rxn.reactant.paired_bem, lot) for c in r_confs]
    if mode in ['dual', 'p_only']:
        biased_r = [ob_joint_optimize(c, rxn.product.paired_bem, lot) for c in p_confs]

    # Default to "conformation-poor" model, unless an excess of conformers is present
    n_conf = config.n_conf
    total_conf = len(r_confs) + len(p_confs)
    verbose = getattr(config, "verbose", False)
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

    # Truncate both the original and biased lists when the conformer pool is too large compared to the number of rxn conformers requested
    limit = n_conf * 5
    if len(r_confs) > limit:
        r_confs = r_confs[:limit]    
        biased_p = biased_p[:limit]
    if len(p_confs) > limit:
        p_confs = p_confs[:limit]    
        biased_r = biased_r[:limit]


    # --- STEP B: Reactant-Product Pair Evaluation & Tournament ---
    approved_pairs = []
    approved_indicators = []
    dropped_pairs = 0

    for ind, r_c in enumerate(r_confs):
         
            # 1. Unaligned Evaluation
            ind_unaligned = return_indicator(E=r_c.elements, RG=r_c.geo, PG=biased_p[ind].geo)
            prob_unaligned = model.predict_proba(ind_unaligned)

            # 2. Aligned Evaluation
            aligned_biased_p = align_conformers(r_c, biased_p[ind])
            ind_aligned = return_indicator(E=r_c.elements, RG=r_c.geo, PG=aligned_biased_p.geo)
            prob_aligned = model.predict_proba(ind_aligned)

            # 3. The Tournament (keep the higher probability setup)
            if prob_aligned[0][1] > prob_unaligned[0][1]:
                best_p = aligned_biased_p
                best_ind = ind_aligned
                best_prob = prob_aligned
            else:
                best_p = biased_p[ind]
                best_ind = ind_unaligned
                best_prob = prob_unaligned

            # 4. Quality Control & Deduplication
            if best_prob[0][1] > 0.0 and check_uniqueness(best_ind, approved_indicators):
                approved_indicators.append(best_ind)
                approved_pairs.append({
                    "r_conf": r_c,
                    "p_conf": best_p,
                    "score": best_prob[0][1]
                })
            else:
                dropped_pairs += 1
####################################################################
    for ind, p_c in enumerate(p_confs):

            # 1. Unaligned Evaluation
            ind_unaligned = return_indicator(E=r_c.elements, RG=p_c.geo, PG=biased_r[ind].geo)
            prob_unaligned = model.predict_proba(ind_unaligned)

            # 2. Aligned Evaluation
            aligned_biased_r = align_conformers(p_c, biased_r[ind])
            ind_aligned = return_indicator(E=r_c.elements, RG=p_c.geo, PG=aligned_biased_r.geo)
            prob_aligned = model.predict_proba(ind_aligned)

            # 3. The Tournament (keep the higher probability setup)
            if prob_aligned[0][1] > prob_unaligned[0][1]:
                best_r = aligned_biased_r
                best_ind = ind_aligned
                best_prob = prob_aligned
            else:
                best_r = biased_r[ind]
                best_ind = ind_unaligned
                best_prob = prob_unaligned

            # 4. Quality Control & Deduplication
            if best_prob[0][1] > 0.0 and check_uniqueness(best_ind, approved_indicators):
                approved_indicators.append(best_ind)
                approved_pairs.append({
                    "r_conf": best_r,
                    "p_conf": p_c,
                    "score": best_prob[0][1]
                })
            else:
                dropped_pairs += 1

    # --- STEP C: Sort and Select Top N ---
    # Sort descending by probability of success
    approved_pairs.sort(key=lambda x: x["score"], reverse=True)

    if verbose:
        print("Number of approved pairs = ", len(approved_pairs))
        print("Number of dropped pairs = ", dropped_pairs)
##############################################################################
    # Truncate to the number requested by the user
    return approved_pairs[:n_conf]


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
