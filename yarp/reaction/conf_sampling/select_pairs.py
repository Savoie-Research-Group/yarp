"""
Logic for joint optimization, ASE alignment, and ML-based conformer selection.
"""
import copy
import numpy as np
import pickle
import os


from ase import Atoms
from ase.build import minimize_rotation_and_translation

from yarp.reaction.conf_sampling.joint_opt import ob_joint_optimize
from yarp.reaction.conf_sampling.indicator import return_indicator


def select_gsm_pairs(rxn, config):
    """
    Orchestrates Biasing -> Alignment -> ML Tournament -> QC -> Pairing.
    """
    mode = config.joint_opt.lower()
    n_conf = config.n_conf
    lot = config.bias_lot
    
    r_confs = list(rxn.reactant.conformers.values())
    p_confs = list(rxn.product.conformers.values())
    total_conf = len(r_confs) + len(p_confs)

    # Default to "conformation-poor" model, unless an excess of conformers is present
    if n_conf / total_conf > 3.0:
        model = pickle.load(open('rich_model.sav', 'rb'))
    else:
        model = pickle.load(open('poor_model.sav', 'rb'))
    
    
    # --- STEP A: Apply Joint Optimization (Biasing) ---
    biased_r = [ob_joint_optimize(c, rxn.reactant.paired_bem, lot) for c in r_confs] if mode in ['dual', 'r_only'] else r_confs
    biased_p = [ob_joint_optimize(c, rxn.product.paired_bem, lot) for c in p_confs] if mode in ['dual', 'p_only'] else p_confs

    # --- STEP B: Cross-Product Evaluation & Tournament ---
    # ERM: We'll see... this is probably an unacceptable bottleneck...
    approved_pairs = []
    approved_indicators = []
    
    for r_c in biased_r:
        for p_c in biased_p:
            
            # 1. Unaligned Evaluation
            ind_unaligned = return_indicator(E=r_c.elements, RG=r_c.geo, PG=p_c.geo)
            prob_unaligned = model.predict_proba(ind_unaligned)
            
            # 2. Aligned Evaluation
            aligned_p_c = align_conformers(r_c, p_c)
            ind_aligned = return_indicator(E=r_c.elements, RG=r_c.geo, PG=aligned_p_c.geo)
            prob_aligned = model.predict_proba(ind_aligned)
            
            # 3. The Tournament (keep the higher probability setup)
            if prob_aligned > prob_unaligned:
                best_p = aligned_p_c
                best_ind = ind_aligned
                best_prob = prob_aligned
            else:
                best_p = p_c
                best_ind = ind_unaligned
                best_prob = prob_unaligned
            
            # 4. Quality Control & Deduplication
            if best_prob > 0.0 and check_duplicate(best_ind, approved_indicators):
                approved_indicators.append(best_ind)
                approved_pairs.append({
                    "r_conf": r_c,
                    "p_conf": best_p,
                    "score": best_prob
                })
                
    # --- STEP C: Sort and Select Top N ---
    # Sort descending by probability of success
    approved_pairs.sort(key=lambda x: x["score"], reverse=True)
    
    # Truncate to the number requested by the user
    return approved_pairs[:n_conf]


def align_conformers(r_conf, p_conf):
    """
    Uses ASE to minimize rotation and translation (RMSD) between product and reactant.
    Returns a NEW product conformer that is aligned to the reactant.
    """
    r_atoms = Atoms(symbols=[el.upper() for el in r_conf.elements], positions=r_conf.geo)
    p_atoms = Atoms(symbols=[el.upper() for el in r_conf.elements], positions=p_conf.geo)
    
    # ASE modifies the moving atoms (p_atoms) in-place to align with the target (r_atoms)
    minimize_rotation_and_translation(r_atoms, p_atoms)
    
    aligned_p_conf = copy.deepcopy(p_conf)
    aligned_p_conf.geo = p_atoms.positions # Extract the rotated/translated coordinates
    aligned_p_conf.type = f"aligned_{p_conf.type}"
    
    return aligned_p_conf

def check_duplicate(new_indicators, approved_indicators_list, threshold=0.025):
    """
    Checks if a pair is too geometrically similar to an already approved pair.
    """
    for approved in approved_indicators_list:
        # Calculate Euclidean distance between feature vectors
        if np.linalg.norm(new_indicators - approved) < threshold:
            return False # It's a duplicate
    return True
