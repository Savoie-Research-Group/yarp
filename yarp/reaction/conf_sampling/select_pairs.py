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
    r_confs = list(rxn.reactant.conformers.values())
    p_confs = list(rxn.product.conformers.values())
    
    # --- STEP A: Apply Joint Optimization (Biasing) ---
    lot = config.bias_lot
    mode = config.joint_opt.lower()

#    biased_r = [ob_joint_optimize(f"{c_ind}_r", c, rxn.reactant.paired_bem, lot) for c_ind, c in enumerate(r_confs)] if mode in ['dual', 'r_only'] else r_confs
#    biased_p = [ob_joint_optimize(f"{c_ind}_p", c, rxn.product.paired_bem, lot) for c_ind, c in enumerate(p_confs)] if mode in ['dual', 'p_only'] else p_confs

    # SHQK : Fixed the biased_r and biased_p designations. Currently the way it's written swaps the original reactant and product
    # SHQK : Also added the c_ind to save the input and output geometries for joint optimizations. This files saving part can be commented out in joint_opt.py. Currently keeping it for debugging purpose 
    # SHQK : biased_p: product geometries (guided by reactant geometries)
    # SHQK : biased_r: reactant geometries (guided by product geometries)
    biased_p = [ob_joint_optimize(f"{c_ind}_r", c, rxn.reactant.paired_bem, lot) for c_ind, c in enumerate(r_confs)] if mode in ['dual', 'r_only'] else r_confs
    biased_r = [ob_joint_optimize(f"{c_ind}_p", c, rxn.product.paired_bem, lot) for c_ind, c in enumerate(p_confs)] if mode in ['dual', 'p_only'] else p_confs


    print("Type of biased_r:", type(biased_r))
    print("Type of biased_p:", type(biased_p))

    # --- STEP B: Cross-Product Evaluation & Tournament ---
    # ERM: We'll see... this is probably an unacceptable bottleneck...

    # Default to "conformation-poor" model, unless an excess of conformers is present
    n_conf = config.n_conf
    total_conf = len(r_confs) + len(p_confs)
    print("Total number of reactant + product conformers generated = ", total_conf)

    module_dir = Path(__file__).parent.resolve()
    if total_conf/n_conf > 3.0:
        print("total_conf/n_conf = ", total_conf/n_conf)
        model_path = module_dir / 'rich_model.sav'
        print("rich model is chosen to generate aligned reaction conformers")
    else:
        print("total_conf/n_conf = ", total_conf/n_conf)
        model_path = module_dir / 'poor_model.sav'
        print("poor model is chosen to generate aligned reaction conformers")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)


    if len(biased_r) > n_conf * 5:
        biased_r = biased_r[:n_conf * 5]
    if len(biased_p) > n_conf * 5:
        biased_p = biased_p[:n_conf * 5]

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
            if prob_aligned[0][1] > prob_unaligned[0][1]:
                best_p = aligned_p_c
                best_ind = ind_aligned
                best_prob = prob_aligned
            else:
                best_p = p_c
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

def check_uniqueness(new_indicators, approved_indicators_list, threshold=0.025):
    """
    Checks if a pair is too geometrically similar to an already approved pair.
    """
    if len(approved_indicators_list) == 0: return True

    min_dis = min([np.linalg.norm(np.array(new_indicators) - np.array(j)) for j in approved_indicators_list])

    if min_dis > threshold: return True
    else: return False
