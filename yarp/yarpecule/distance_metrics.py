"""
Distance metrics between molecular graphs

Distance metrics implemented:

    * soergel
    * delta_bertz
    * crippen_diff
    * delta_tpsa
    * approx_dipole_magnitude
    * maccs_tanimoto_distance
    * mcs_bond_edit_distance
    * am_ged
    * cost_aware_ged
"""
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem, GraphDescriptors, Crippen, rdMolDescriptors, MACCSkeys, DataStructs, rdFMCS
import networkx as nx
import numpy as np

from yarp.yarpecule.graph.smiles import smiles2adjmat

def compute_min_distance(mol_yp, target_smi, metric='soergel'):
    """
    Compute minimum distance between all species in a yarpecule and a target SMILES.
    """
    if mol_yp.canon_smi is None:
        mol_yp.get_smiles()

    all_dist = []

    # Check for separable molecules and compute individual distances
    mols = mol_yp.separate()
    if len(mols) > 1:
        for mol in mols:
            mol.get_smiles()
            smi = mol.map_smi if metric == "am_ged" else mol.canon_smi
            d = compute_distance(smi, target_smi, metric)
            all_dist.append(d)

    # Compute distance for original full molecule node
    smi = mol_yp.map_smi if metric == "am_ged" else mol_yp.canon_smi
    dist = compute_distance(smi, target_smi, metric)
    all_dist.append(dist)

    return min(all_dist)

def compute_distance(smi1, smi2, metric='soergel'):
    """
    Compute distance between two SMILES based on requested metric.
    """
    if metric == 'soergel':
        dist = soergel(smi1, smi2)
    elif metric == 'delta_bertz':
        dist = delta_bertz(smi1, smi2)
    elif metric == 'crippen_diff':
        dist = crippen_diff(smi1, smi2)
    elif metric == 'delta_tpsa':
        dist = delta_tpsa(smi1, smi2)
    elif metric == 'approx_dipole_magnitude':
        dist = approx_dipole_magnitude(smi1, smi2)
    elif metric == 'maccs_tanimoto_distance':
        dist = maccs_tanimoto_distance(smi1, smi2)
    elif metric == 'mcs_bond_edit_distance':
        dist = mcs_bond_edit_distance(smi1, smi2, ringMatchesRingOnly=False)
    elif metric == 'am_ged':
        dist = atom_map_ged(smi1, smi2)
    elif metric == 'cost_aware_ged':
        dist = cost_aware_ged(smi1, smi2)
    else:
        raise RuntimeError(f"Requested distance metric {metric} not implemented!")

    return dist

def soergel(smi1, smi2):
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)

    # Create a Morgan fingerprint generator (radius 2, 2048 bits)
    generator = GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False)

    # Generate fingerprints
    fp1 = generator.GetFingerprint(mol1)
    fp2 = generator.GetFingerprint(mol2)

    # Compute Tanimoto similarity
    similarity = TanimotoSimilarity(fp1, fp2)

    # Convert to Soergel distance
    distance = 1.0 - similarity

    return distance

def delta_bertz(smi_1,smi_2):
    """Absolute difference in RDKit Bertz topological complexity."""
    try:
        m1, m2 = Chem.MolFromSmiles(smi_1), Chem.MolFromSmiles(smi_2)
        if m1 is None or m2 is None:
            return None
        v1 = float(GraphDescriptors.BertzCT(m1))
        v2 = float(GraphDescriptors.BertzCT(m2))
        return abs(v1 - v2)
    except Exception as e:
        return None
    
def crippen_diff(smi1, smi2):
    """Absolute difference in Crippen logP."""
    try:
        m1, m2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)
        if m1 is None or m2 is None: return None
        v1 = Crippen.MolMR(m1)
        v2 = Crippen.MolMR(m2)
        return float(abs(v1 - v2))
    except Exception as e:
        return None
    
def delta_tpsa(smi_1: str, smi_2: str):
    """Absolute difference in Topological Polar Surface Area (TPSA)."""
    try:
        m1, m2 = Chem.MolFromSmiles(smi_1), Chem.MolFromSmiles(smi_2)
        if m1 is None or m2 is None: return None
        v1 = rdMolDescriptors.CalcTPSA(m1)
        v2 = rdMolDescriptors.CalcTPSA(m2)
        return float(abs(v1 - v2))
    except Exception as e:
        return None

def approx_dipole_magnitude(smi_1, smi_2):
    """Approximate dipole moment magnitude difference using Gasteiger charges."""
    try:
        m_1 = Chem.AddHs(Chem.MolFromSmiles(smi_1))
        m_2 = Chem.AddHs(Chem.MolFromSmiles(smi_2))
        AllChem.EmbedMolecule(m_1, randomSeed=42)
        AllChem.EmbedMolecule(m_2, randomSeed=42)
        Chem.rdPartialCharges.ComputeGasteigerCharges(m_1)
        Chem.rdPartialCharges.ComputeGasteigerCharges(m_2)
        coords_1 = m_1.GetConformer().GetPositions()
        coords_2 = m_2.GetConformer().GetPositions()
        charges_1 = np.array([float(a.GetProp('_GasteigerCharge')) for a in m_1.GetAtoms()])
        charges_2 = np.array([float(a.GetProp('_GasteigerCharge')) for a in m_2.GetAtoms()])
        dip_1 = np.sum(coords_1 * charges_1[:, None], axis=0)
        dip_2 = np.sum(coords_2 * charges_2[:, None], axis=0)
        return abs(np.linalg.norm(dip_1) - np.linalg.norm(dip_2))
    except:
        return np.nan
    
def maccs_tanimoto_distance(smi_1, smi_2):
    """MACCS Tanimoto distance."""
    try:
        m1, m2 = Chem.MolFromSmiles(smi_1), Chem.MolFromSmiles(smi_2)
        if m1 is None or m2 is None: return None
        fp1 = MACCSkeys.GenMACCSKeys(m1)
        fp2 = MACCSkeys.GenMACCSKeys(m2)
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        return float(1.0 - sim)
    except Exception as e:
        return None
    
def mcs_bond_edit_distance(smi_1, smi_2, ringMatchesRingOnly, completeRingsOnly=False, timeout=10):
    """MCS-based bond edit distance."""
    try:
        m1, m2 = Chem.MolFromSmiles(smi_1), Chem.MolFromSmiles(smi_2)
        if m1 is None or m2 is None: return None
        b1, b2 = m1.GetNumBonds(), m2.GetNumBonds()
        if (b1 + b2) == 0:
            return 0.0  # both have no bonds; treat as identical
        params = rdFMCS.MCSParameters()
        params.MaximizeBonds = True
        params.CompleteRingsOnly = completeRingsOnly
        params.RingMatchesRingOnly = ringMatchesRingOnly
        params.Timeout = timeout
        res = rdFMCS.FindMCS([m1, m2], params)
        mcs_bonds = res.numBonds if res is not None else 0
        dist = 1.0 - (2.0 * mcs_bonds) / float(b1 + b2)
        # clip numerical noise
        if dist < 0: dist = 0.0
        if dist > 1: dist = 1.0
        return float(dist)
    except Exception as e:
        return None
    
def atom_map_ged(smi_1, smi_2):
    """Unweighted graph edit distance between two atom-mapped SMILES strings."""
    try:
        adj_mat_1, _, atom_info_1 = smiles2adjmat(smi_1)
        adj_mat_2, _, atom_info_2 = smiles2adjmat(smi_2)
        if adj_mat_1 is None or adj_mat_2 is None:
            return None

        keep_1 = [i for i in range(len(atom_info_1)) if atom_info_1[i]["element"] != "h"]
        keep_2 = [i for i in range(len(atom_info_2)) if atom_info_2[i]["element"] != "h"]
        mat_1 = adj_mat_1[np.ix_(keep_1, keep_1)] if keep_1 else np.zeros((0, 0))
        mat_2 = adj_mat_2[np.ix_(keep_2, keep_2)] if keep_2 else np.zeros((0, 0))

        size = max(mat_1.shape[0], mat_2.shape[0])
        if mat_1.shape[0] < size:
            padded = np.zeros((size, size))
            padded[:mat_1.shape[0], :mat_1.shape[1]] = mat_1
            mat_1 = padded
        if mat_2.shape[0] < size:
            padded = np.zeros((size, size))
            padded[:mat_2.shape[0], :mat_2.shape[1]] = mat_2
            mat_2 = padded

        return float(np.sum(np.abs(mat_1 - mat_2)) / 2.0)
    except Exception:
        return None


def cost_aware_ged(smi_1, smi_2):
    """
    Cost-aware graph edit distance between two SMILES strings.

    The edit costs are meant to encode simple chemical heuristics:

    - `node_subst_cost`
        Atom substitution is free when the atomic numbers match, because no
        chemical identity change is needed. Replacing hydrogen with a heavy
        atom, or vice versa, costs `0.5`, which treats H edits as cheaper than
        changing one heavy element into another. Any heavy-atom-to-heavy-atom
        substitution with different atomic numbers costs `1.0`.

    - `node_del_cost` and `node_ins_cost`
        Deleting or inserting hydrogen costs `0.5`, reflecting that changing
        hydrogen count is usually a smaller perturbation than adding or removing
        a heavy atom. Deleting or inserting a heavy atom costs `1.0`.

    - `edge_subst_cost`
        Bond substitution is free when bond orders match. Changing the bond
        order, such as single to double, costs `0.7`, which penalizes
        rehybridization or pi-bond rearrangement but keeps it cheaper than a
        full heavy-atom substitution.

    - `edge_del_cost` and `edge_ins_cost`
        Deleting or inserting a bond costs `0.7`. This makes bond formation or
        bond cleavage significant, but still somewhat cheaper than deleting and
        re-inserting an entire heavy atom.

    These values are the main tuning knobs for changing how strongly the
    metric penalizes atom identity changes, hydrogen-count changes, bond-order
    changes, and bond creation/destruction.
    """
    try:
        mol_1 = Chem.MolFromSmiles(smi_1)
        mol_2 = Chem.MolFromSmiles(smi_2)
        if mol_1 is None or mol_2 is None:
            return None

        graph_1 = nx.Graph()
        for atom in mol_1.GetAtoms():
            graph_1.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum(), symbol=atom.GetSymbol())
        for bond in mol_1.GetBonds():
            graph_1.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_order=bond.GetBondTypeAsDouble())

        graph_2 = nx.Graph()
        for atom in mol_2.GetAtoms():
            graph_2.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum(), symbol=atom.GetSymbol())
        for bond in mol_2.GetBonds():
            graph_2.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_order=bond.GetBondTypeAsDouble())

        return min(nx.optimize_graph_edit_distance(
            graph_1,
            graph_2,
            node_subst_cost=lambda atom1, atom2: 0.0 if atom1["atomic_num"] == atom2["atomic_num"] else 0.5 if 1 in (atom1["atomic_num"], atom2["atomic_num"]) else 1.0,
            node_del_cost=lambda atom: 0.5 if atom["atomic_num"] == 1 else 1.0,
            node_ins_cost=lambda atom: 0.5 if atom["atomic_num"] == 1 else 1.0,
            edge_subst_cost=lambda bond1, bond2: 0.0 if bond1["bond_order"] == bond2["bond_order"] else 0.7,
            edge_del_cost=lambda bond: 0.7,
            edge_ins_cost=lambda bond: 0.7,
        ))
    except Exception:
        return None
