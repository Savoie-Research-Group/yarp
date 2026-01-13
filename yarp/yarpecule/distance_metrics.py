"""
Distance metrics between molecular graphs
"""
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem, GraphDescriptors, Crippen, rdMolDescriptors, MACCSkeys, DataStructs, rdFMCS
import numpy as np

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
        m1, m2 = Chem.MolFromSmiles(smi_1), Chem.MolFromSmiles(smi_2)
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