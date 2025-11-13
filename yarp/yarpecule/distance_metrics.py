"""
Distance metrics between molecular graphs
"""
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity


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