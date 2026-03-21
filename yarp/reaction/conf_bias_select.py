"""
Logic for joint optimization, ASE alignment, and ML-based conformer selection.
"""
import copy
import numpy as np
from openbabel import openbabel as ob

from ase import Atoms
from ase.build import minimize_rotation_and_translation

# =====================================================================
# 1. BIASING (Joint Optimization)
# =====================================================================
def build_ob_mol(elements, coords):
    """Converts a Yarpecule/Conformer geometry to an OpenBabel OBMol."""
    obMol = ob.OBMol()
    for el, coord in zip(elements, coords):
        obAtom = obMol.NewAtom()
        obAtom.SetAtomicNum(ob.GetAtomicNum(el))
        obAtom.SetVector(*coord)
    return obMol

def ob_joint_optimize(conformer, target_bem, ff_name="uff"):
    """
    Applies constraints based on the target BEM and runs an OpenBabel FF optimization.
    Returns a NEW conformer object with the biased geometry.
    """
    obMol = build_ob_mol(conformer.elements, conformer.geo)
    ff = ob.OBForceField.FindForceField(ff_name)
    if not ff:
        ff = ob.OBForceField.FindForceField("uff")
    
    ff.Setup(obMol)
    
    # Mock constraints based on Target BEM
    constraints = ob.OBFFConstraints()
    num_atoms = len(conformer.elements)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if target_bem[i][j] > 0:
                constraints.AddDistanceConstraint(i + 1, j + 1, 1.5) 
                
    ff.SetConstraints(constraints)
    ff.ConjugateGradients(500)
    ff.GetCoordinates(obMol)
    
    biased_conf = copy.deepcopy(conformer)
    biased_conf.geo = np.array([[obMol.GetAtom(i).GetX(), obMol.GetAtom(i).GetY(), obMol.GetAtom(i).GetZ()] 
                                for i in range(1, obMol.NumAtoms() + 1)])
    biased_conf.type = f"biased_{conformer.type}"
    return biased_conf

# =====================================================================
# 2. ALIGNMENT (ASE)
# =====================================================================
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

# =====================================================================
# 3. FEATURE EXTRACTION & ML RANKING
# =====================================================================
def get_indicators(r_conf, p_conf):
    """
    Extracts numerical features (indicators) describing the chemical transformation.
    (Replace this with your actual YARP indicator logic).
    """
    # Mock feature: flattening the distance matrix between the two geometries
    diff = np.array(r_conf.geo) - np.array(p_conf.geo)
    return diff.flatten()

def mock_ml_predict(indicators):
    """
    Mock ML Model. Returns probability of success (0.0 to 1.0).
    """
    import random
    return random.uniform(0.0, 1.0)

def check_duplicate(new_indicators, approved_indicators_list, threshold=0.025):
    """
    Checks if a pair is too geometrically similar to an already approved pair.
    """
    for approved in approved_indicators_list:
        # Calculate Euclidean distance between feature vectors
        if np.linalg.norm(new_indicators - approved) < threshold:
            return False # It's a duplicate
    return True

# =====================================================================
# 4. ORCHESTRATION
# =====================================================================
def select_gsm_pairs(rxn, config):
    """
    Orchestrates Biasing -> Alignment -> ML Tournament -> QC -> Pairing.
    """
    mode = config.joint_opt.lower()
    n_conf = config.n_conf
    lot = config.bias_lot
    
    r_confs = list(rxn.reactant.conformers.values())
    p_confs = list(rxn.product.conformers.values())
    
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
            ind_unaligned = get_indicators(r_c, p_c)
            prob_unaligned = mock_ml_predict(ind_unaligned)
            
            # 2. Aligned Evaluation
            aligned_p_c = align_conformers(r_c, p_c)
            ind_aligned = get_indicators(r_c, aligned_p_c)
            prob_aligned = mock_ml_predict(ind_aligned)
            
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