"""
This script tests code functionality starting from
- branch: smi2adj
- commit: 1d22976
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdchem
from yarp.yarpecule.graph.smiles import smiles2adjmat


def analyze_smiles_with_rdkit(smiles):
    """Analyze a SMILES string with RDKit and extract detailed information."""
    print(f"\n=== RDKit Analysis for: {smiles} ===")

    # Parse with RDKit
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Failed to parse with RDKit")
        return None

    # Add explicit hydrogens
    mol_with_h = Chem.AddHs(mol)

    print(f"Number of atoms (without H): {mol.GetNumAtoms()}")
    print(f"Number of atoms (with H): {mol_with_h.GetNumAtoms()}")

    # Get atom information
    print("\nAtom Details (with explicit H):")
    elements = []
    formal_charges = []
    atom_map_nums = []

    for i, atom in enumerate(mol_with_h.GetAtoms()):
        symbol = atom.GetSymbol()
        formal_charge = atom.GetFormalCharge()
        map_num = atom.GetAtomMapNum()
        num_h = atom.GetTotalNumHs()

        elements.append(symbol)
        formal_charges.append(formal_charge)
        atom_map_nums.append(map_num)

        print(
            f"  Atom {i}: {symbol}, FC={formal_charge}, MapNum={map_num}, TotalH={num_h}")

    # Get adjacency matrix
    n_atoms = mol_with_h.GetNumAtoms()
    adj_matrix = np.zeros((n_atoms, n_atoms), dtype=int)

    for bond in mol_with_h.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_order = int(bond.GetBondType())
        adj_matrix[i, j] = bond_order
        adj_matrix[j, i] = bond_order

    print(f"\nElement sequence: {elements}")
    print(f"Adjacency matrix shape: {adj_matrix.shape}")
    print("Adjacency matrix:")
    print(adj_matrix)

    return {
        'elements': elements,
        'formal_charges': formal_charges,
        'atom_map_nums': atom_map_nums,
        'adj_matrix': adj_matrix,
        'n_atoms': n_atoms
    }


def analyze_smiles_with_yarp(smiles):
    """Analyze a SMILES string with yarp parser and extract detailed information."""
    print(f"\n=== YARP Analysis for: {smiles} ===")

    try:
        # Parse with yarp (with verbose output)
        adj_matrix, atom_info = smiles2adjmat(smiles, verbose=True)

        elements = [info[0] for info in atom_info]
        formal_charges = [info[1] for info in atom_info]
        explicit_hydrogens = [info[2] for info in atom_info]
        isotopes = [info[3] for info in atom_info]
        atom_mappings = [info[4] for info in atom_info]
        should_infer_hydrogens = [info[5] for info in atom_info]

        print(f"Number of atoms: {len(atom_info)}")
        print(f"Element sequence: {elements}")

        print("\nAtom Details:")
        for i, info in enumerate(atom_info):
            print(f"  Atom {i}: {info[0]}, FC={info[1]}, ExplicitH={info[2]}, "
                  f"Isotope={info[3]}, Mapping={info[4]}, ShouldInfer={info[5]}")

        print(f"Adjacency matrix shape: {adj_matrix.shape}")
        print("Adjacency matrix:")
        print(adj_matrix.astype(int))

        return {
            'elements': elements,
            'formal_charges': formal_charges,
            'explicit_hydrogens': explicit_hydrogens,
            'isotopes': isotopes,
            'atom_mappings': atom_mappings,
            'should_infer_hydrogens': should_infer_hydrogens,
            'adj_matrix': adj_matrix,
            'n_atoms': len(atom_info)
        }

    except Exception as e:
        print(f"Error parsing with yarp: {e}")
        return None


def compare_results(smiles):
    """Compare RDKit and yarp results for a SMILES string."""
    print(f"\n{'='*60}")
    print(f"COMPARING RESULTS FOR: {smiles}")
    print(f"{'='*60}")

    rdkit_result = analyze_smiles_with_rdkit(smiles)
    yarp_result = analyze_smiles_with_yarp(smiles)

    if rdkit_result and yarp_result:
        print(f"\n=== COMPARISON ===")
        print(f"RDKit elements: {rdkit_result['elements']}")
        print(f"YARP elements:  {yarp_result['elements']}")
        print(
            f"Elements match: {rdkit_result['elements'] == yarp_result['elements']}")

        print(f"\nRDKit atom count: {rdkit_result['n_atoms']}")
        print(f"YARP atom count:  {yarp_result['n_atoms']}")
        print(
            f"Atom counts match: {rdkit_result['n_atoms'] == yarp_result['n_atoms']}")

        print(
            f"\nMatrix shapes - RDKit: {rdkit_result['adj_matrix'].shape}, YARP: {yarp_result['adj_matrix'].shape}")

        if rdkit_result['adj_matrix'].shape == yarp_result['adj_matrix'].shape:
            matrices_match = np.array_equal(
                rdkit_result['adj_matrix'], yarp_result['adj_matrix'])
            print(f"Adjacency matrices match: {matrices_match}")
            if not matrices_match:
                print("Differences in adjacency matrices:")
                diff = rdkit_result['adj_matrix'] - yarp_result['adj_matrix']
                print("RDKit - YARP =")
                print(diff)
        else:
            print("Cannot compare adjacency matrices - different shapes")


# Test the failing cases
if __name__ == "__main__":
    # The two failing SMILES strings
    rad_explicitH_smi = 'C([C]([H])[H])([H])([H])[H]'
    rad_full_map_smi = '[C:1]([C:2]([H:3])[H:7])([H:5])([H:6])[H:4]'
    test_1 = '[C:1]([C:2]([H:3])[H:7])([H:5])([H:6])[H:4]'
    test_2 = '[C:1]([C:2])([H:5])([H:6])[H:4]'
    test_3 = '[C]([C]([H])[H])([H])([H])[H]'
    test_4 = '[C:1]([C:2])'
    test_5 = '[C:1][O:2]'
    test_6 = '[CH2:2][C:1]'
    test_7 = '[C:2][C:3](=[O:1])[O-:4]'
    test_8 = 'Oc1ccoc1'
    test_9 = '[c:0]1([H:15])[c:4]([O:9][H:13])[c:5]([H:20])[c:1]([H:16])[o:6]1'
    # Compare both
    # compare_results(rad_explicitH_smi)
    # compare_results(rad_full_map_smi)
    compare_results(test_1)
    compare_results(test_2)
    compare_results(test_3)
    compare_results(test_4)
    compare_results(test_5)
    compare_results(test_6)
    compare_results(test_7)
    compare_results(test_8)
    compare_results(test_9)
