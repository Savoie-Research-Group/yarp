"""
Testing suite for functions contained in yarp/yarpecule/input_parser.py
"""
from collections import Counter
import pytest
from yarp.yarpecule.input_parsers import xyz_parse
from yarp.yarpecule.input_parsers import xyz_q_parse
from yarp.yarpecule.input_parsers import mol_parse
from yarp.yarpecule.input_parsers import reaction_xyz_parse
from yarp.yarpecule.input_parsers import load_reactions_from_xyz_directory
from yarp.yarpecule.input_parsers import load_reactions_from_smiles_file
from yarp.yarpecule.input_parsers import xyz_from_smiles
from yarp.yarpecule.yarpecule import yarpecule


class TestXYZParser:
    def test_single_molecule_no_types(self, ethene_xyz):
        elements, geo = xyz_parse(ethene_xyz)

        assert elements == ["C", "C", "H", "H", "H", "H"]
        assert geo.shape == (6, 3)
        assert geo[0, 0] == pytest.approx(0.01051, rel=1e-5)
        assert geo[0, 1] == pytest.approx(-0.00247, rel=1e-5)
        assert geo[0, 2] == pytest.approx(0.29143, rel=1e-5)
        assert geo[1, 0] == pytest.approx(-0.04372, rel=1e-5)
        assert geo[1, 1] == pytest.approx(-0.04563, rel=1e-5)
        assert geo[1, 2] == pytest.approx(1.61414, rel=1e-5)
        assert geo[2, 0] == pytest.approx(0.66926, rel=1e-5)
        assert geo[2, 1] == pytest.approx(-0.66369, rel=1e-5)
        assert geo[2, 2] == pytest.approx(-0.26622, rel=1e-5)
        assert geo[3, 0] == pytest.approx(-0.60221, rel=1e-5)
        assert geo[3, 1] == pytest.approx(0.69539, rel=1e-5)
        assert geo[3, 2] == pytest.approx(-0.27400, rel=1e-5)
        assert geo[4, 0] == pytest.approx(0.56900, rel=1e-5)
        assert geo[4, 1] == pytest.approx(-0.74348, rel=1e-5)
        assert geo[4, 2] == pytest.approx(2.17958, rel=1e-5)
        assert geo[5, 0] == pytest.approx(-0.70247, rel=1e-5)
        assert geo[5, 1] == pytest.approx(0.61559, rel=1e-5)
        assert geo[5, 2] == pytest.approx(2.17179, rel=1e-5)

class TestXYZQParser:
    def test_xyz_with_pos_q(self, ammonium_xyz):
        q = xyz_q_parse(ammonium_xyz)
        assert q == pytest.approx(1, rel=1e-5)
        
    def test_xyz_neg_q(self, nitrate_xyz):
        q = xyz_q_parse(nitrate_xyz)
        assert q == pytest.approx(-1, rel=1e-5)
                
    def test_xyz_no_q(self, ethene_xyz):
        q = xyz_q_parse(ethene_xyz)
        assert q == pytest.approx(0, rel=1e-5)

class TestMolParser:
    def test_single_mol_no_charge(self, ethanol_mol):
        elements, geo, adj_mat, q, atom_info = mol_parse(ethanol_mol)
        
        # elements, heavy atoms only
        assert elements == ['C','C',"O"]
        
        # geometry, heavy atoms only
        assert geo.shape == (3, 3)
        assert geo[0, 0] == pytest.approx(-0.9254, rel=1e-5)
        assert geo[0, 1] == pytest.approx(0.0742, rel=1e-5)
        assert geo[0, 2] == pytest.approx(0.0328, rel=1e-5)
        assert geo[1, 0] == pytest.approx(0.5123, rel=1e-5)
        assert geo[1, 1] == pytest.approx(-0.4192, rel=1e-5)
        assert geo[1, 2] == pytest.approx(-0.0743, rel=1e-5)
        assert geo[2, 0] == pytest.approx(1.3778, rel=1e-5)

        # adjacency matrix
        assert adj_mat.shape == (3,3)
        assert adj_mat[0, 0] == 0
        assert adj_mat[0, 1] == 1
        assert adj_mat[0, 2] == 0
        assert adj_mat[1, 0] == 1
        assert adj_mat[1, 1] == 0
        assert adj_mat[1, 2] == 1
        assert adj_mat[2, 0] == 0
        assert adj_mat[2, 1] == 1
        assert adj_mat[2, 2] == 0
        
        # charge
        assert q == pytest.approx(0, rel=1e-5)
        
        # atom info
        assert isinstance(atom_info, dict)
        assert sorted(atom_info) == list(range(len(elements)))
    
    def test_single_mol_with_charge(self, acetate_mol):
        elements, geo, adj_mat, q, atom_info = mol_parse(acetate_mol)
        
        # Elements, heavy atoms only
        assert elements == ['C','C',"O","O"]
        
        #geometry, heavy atoms only
        assert geo.shape == (4, 3)
        assert geo[0, 0] == pytest.approx(0.0000, rel=1e-5)
        assert geo[0, 1] == pytest.approx(2.2356, rel=1e-5)
        assert geo[0, 2] == pytest.approx(0.0000, rel=1e-5)
        assert geo[1, 0] == pytest.approx(1.0700, rel=1e-5)
        assert geo[1, 1] == pytest.approx(1.3328, rel=1e-5)
        assert geo[1, 2] == pytest.approx(0.0000, rel=1e-5)
        assert geo[2, 0] == pytest.approx(2.0260, rel=1e-5)
        assert geo[2, 1] == pytest.approx(2.3555, rel=1e-5)
        assert geo[2, 2] == pytest.approx(0.0000, rel=1e-5)
        assert geo[3, 0] == pytest.approx(0.6413, rel=1e-5)
        assert geo[3, 1] == pytest.approx(0.0000, rel=1e-5)
        assert geo[3, 2] == pytest.approx(0.0000, rel=1e-5)
        
        # adjacency matrix
        assert adj_mat.shape == (4,4)
        assert adj_mat[0, 0] == 0
        assert adj_mat[0, 1] == 1
        assert adj_mat[0, 2] == 0
        assert adj_mat[0, 3] == 0
        assert adj_mat[1, 0] == 1
        assert adj_mat[1, 1] == 0
        assert adj_mat[1, 2] == 1
        assert adj_mat[1, 3] == 1
        assert adj_mat[2, 0] == 0
        assert adj_mat[2, 1] == 1
        assert adj_mat[2, 2] == 0
        assert adj_mat[2, 3] == 0
        assert adj_mat[3, 0] == 0
        assert adj_mat[3, 1] == 1
        assert adj_mat[3, 2] == 0
        assert adj_mat[3, 3] == 0
        
        # charge
        assert q == pytest.approx(-1, rel=1e-5)
        
    def test_mol_with_zwitterion(self, betaine_mol):
        elements, geo, adj_mat, q, atom_info = mol_parse(betaine_mol)
        
        # Elements, heavy atoms only
        assert elements == ['C','N','C','C','C','C','O','O']
        
        # geometry, heavy atoms only
        assert geo.shape == (8, 3)
        assert geo[0, 0] == pytest.approx(2.0981, rel=1e-5)
        assert geo[0, 1] == pytest.approx(0.8660, rel=1e-5)
        assert geo[0, 2] == pytest.approx(0.0000, rel=1e-5)
        assert geo[1, 0] == pytest.approx(2.5981, rel=1e-5)
        assert geo[1, 1] == pytest.approx(0.0000, rel=1e-5)
        assert geo[1, 2] == pytest.approx(0.0000, rel=1e-5)
        assert geo[2, 0] == pytest.approx(3.4641, rel=1e-5)
        assert geo[2, 1] == pytest.approx(0.5000, rel=1e-5)
        assert geo[2, 2] == pytest.approx(0.0000, rel=1e-5)
        assert geo[3, 0] == pytest.approx(3.0981, rel=1e-5)
        assert geo[3, 1] == pytest.approx(-0.8660, rel=1e-5)
        assert geo[3, 2] == pytest.approx(0.0000, rel=1e-5)
        assert geo[4, 0] == pytest.approx(1.7321, rel=1e-5)
        assert geo[4, 1] == pytest.approx(-0.5000, rel=1e-5)
        assert geo[4, 2] == pytest.approx(0.0000, rel=1e-5)
        assert geo[5, 0] == pytest.approx(0.8660, rel=1e-5)
        assert geo[5, 1] == pytest.approx(0.0000, rel=1e-5)
        assert geo[5, 2] == pytest.approx(0.0000, rel=1e-5)
        assert geo[6, 0] == pytest.approx(0.8660, rel=1e-5)
        assert geo[6, 1] == pytest.approx(1.0000, rel=1e-5)
        assert geo[6, 2] == pytest.approx(0.0000, rel=1e-5)
        assert geo[7, 0] == pytest.approx(0.0000, rel=1e-5)
        assert geo[7, 1] == pytest.approx(-0.5000, rel=1e-5)
        assert geo[7, 2] == pytest.approx(0.0000, rel=1e-5)
        
        # adjacency matrix  
        assert adj_mat.shape == (8,8)
        assert adj_mat[0, 0] == 0
        assert adj_mat[0, 1] == 1
        assert adj_mat[0, 2] == 0
        assert adj_mat[0, 3] == 0
        assert adj_mat[0, 4] == 0
        assert adj_mat[0, 5] == 0
        assert adj_mat[0, 6] == 0
        assert adj_mat[0, 7] == 0
        assert adj_mat[1, 0] == 1
        assert adj_mat[1, 1] == 0
        assert adj_mat[1, 2] == 1
        assert adj_mat[1, 3] == 1
        assert adj_mat[1, 4] == 1
        assert adj_mat[1, 5] == 0
        assert adj_mat[1, 6] == 0
        assert adj_mat[1, 7] == 0
        assert adj_mat[2, 0] == 0
        assert adj_mat[2, 1] == 1
        assert adj_mat[2, 2] == 0
        assert adj_mat[2, 3] == 0
        assert adj_mat[2, 4] == 0
        assert adj_mat[2, 5] == 0
        assert adj_mat[2, 6] == 0
        assert adj_mat[2, 7] == 0
        assert adj_mat[3, 0] == 0
        assert adj_mat[3, 1] == 1
        assert adj_mat[3, 2] == 0
        assert adj_mat[3, 3] == 0
        assert adj_mat[3, 4] == 0
        assert adj_mat[3, 5] == 0
        assert adj_mat[3, 6] == 0
        assert adj_mat[3, 7] == 0
        assert adj_mat[4, 0] == 0
        assert adj_mat[4, 1] == 1
        assert adj_mat[4, 2] == 0
        assert adj_mat[4, 3] == 0
        assert adj_mat[4, 4] == 0
        assert adj_mat[4, 5] == 1
        assert adj_mat[4, 6] == 0
        assert adj_mat[4, 7] == 0
        assert adj_mat[5, 0] == 0
        assert adj_mat[5, 1] == 0
        assert adj_mat[5, 2] == 0
        assert adj_mat[5, 3] == 0
        assert adj_mat[5, 4] == 1
        assert adj_mat[5, 5] == 0
        assert adj_mat[5, 6] == 1
        assert adj_mat[5, 7] == 1
        assert adj_mat[6, 0] == 0
        assert adj_mat[6, 1] == 0
        assert adj_mat[6, 2] == 0
        assert adj_mat[6, 3] == 0
        assert adj_mat[6, 4] == 0
        assert adj_mat[6, 5] == 1
        assert adj_mat[6, 6] == 0
        assert adj_mat[6, 7] == 0
        assert adj_mat[7, 0] == 0
        assert adj_mat[7, 1] == 0
        assert adj_mat[7, 2] == 0
        assert adj_mat[7, 3] == 0
        assert adj_mat[7, 4] == 0
        assert adj_mat[7, 5] == 1
        assert adj_mat[7, 6] == 0
        assert adj_mat[7, 7] == 0

        # charge
        assert q == 0


class TestXYZRxn:
    def test_initialize_xyz_reaction_folder(self, test_xyz_dir, capsys):
        reactions = load_reactions_from_xyz_directory(test_xyz_dir)
        captured = capsys.readouterr()

        structure_error = (
            "must contain exactly two coordinate sets (reactant first, product second) "
            "where first line of each set is the number of atoms and the second line is a "
            "comment or optionally contains charge information with the format `q <charge>`"
        )

        assert f"Failed to initialize 6 reaction(s) from {test_xyz_dir}:" in captured.out
        assert f"{test_xyz_dir / 'failure1.xyz'} has mismatched reactant/product atom counts." in captured.out
        assert f"{test_xyz_dir / 'failure2.xyz'} requires identical atom ordering between reactant and product." in captured.out
        assert (
            f"{test_xyz_dir / 'failure3.xyz'}: ERROR in reaction_xyz_parse: "
            f"{test_xyz_dir / 'failure3.xyz'} {structure_error}"
        ) in captured.out
        assert (
            f"{test_xyz_dir / 'failure4.xyz'}: ERROR in reaction_xyz_parse: "
            f"{test_xyz_dir / 'failure4.xyz'} {structure_error}"
        ) in captured.out
        assert f"{test_xyz_dir / 'failure5.xyz'}: list assignment index out of range" in captured.out
        assert (
            f"{test_xyz_dir / 'failure6.xyz'}: ERROR in reaction_xyz_parse: "
            f"{test_xyz_dir / 'failure6.xyz'} {structure_error}"
        ) in captured.out

        assert len(reactions) == 5

    def test_reaction1_xyz_parse(self, test_xyz_dir):
        xyz_file = test_xyz_dir / "reaction1.xyz"
        reactant_elements, reactant_geo, reactant_q, product_elements, product_geo, product_q = reaction_xyz_parse(str(xyz_file))

        assert reactant_elements == ["n", "c", "c", "o", "n", "n", "h", "h", "h"]
        assert product_elements == ["n", "c", "c", "o", "n", "n", "h", "h", "h"]
        assert reactant_geo.shape == (9, 3)
        assert product_geo.shape == (9, 3)
        assert reactant_q == pytest.approx(0, rel=1e-5)
        assert product_q == pytest.approx(0, rel=1e-5)

        assert reactant_geo[0, 0] == pytest.approx(-1.31331, rel=1e-5)
        assert reactant_geo[0, 1] == pytest.approx(0.63116, rel=1e-5)
        assert reactant_geo[0, 2] == pytest.approx(-0.60438, rel=1e-5)
        assert reactant_geo[8, 0] == pytest.approx(0.04874, rel=1e-5)
        assert reactant_geo[8, 1] == pytest.approx(-2.22984, rel=1e-5)
        assert reactant_geo[8, 2] == pytest.approx(-0.36590, rel=1e-5)

        assert product_geo[0, 0] == pytest.approx(-1.33568, rel=1e-5)
        assert product_geo[0, 1] == pytest.approx(0.52530, rel=1e-5)
        assert product_geo[0, 2] == pytest.approx(-0.03371, rel=1e-5)
        assert product_geo[8, 0] == pytest.approx(0.21102, rel=1e-5)
        assert product_geo[8, 1] == pytest.approx(-2.14752, rel=1e-5)
        assert product_geo[8, 2] == pytest.approx(-0.10526, rel=1e-5)

    def test_reaction2_xyz_parse(self, test_xyz_dir):
        xyz_file = test_xyz_dir / "reaction2.xyz"
        reactant_elements, reactant_geo, reactant_q, product_elements, product_geo, product_q = reaction_xyz_parse(str(xyz_file))

        assert reactant_elements == ["n", "c", "c", "c", "n", "n", "o", "h", "h", "h"]
        assert product_elements == ["n", "c", "c", "c", "n", "n", "o", "h", "h", "h"]
        assert reactant_geo.shape == (10, 3)
        assert product_geo.shape == (10, 3)
        assert reactant_q == pytest.approx(0, rel=1e-5)
        assert product_q == pytest.approx(0, rel=1e-5)

        assert reactant_geo[0, 0] == pytest.approx(-0.54214, rel=1e-5)
        assert reactant_geo[0, 1] == pytest.approx(0.96718, rel=1e-5)
        assert reactant_geo[0, 2] == pytest.approx(1.01861, rel=1e-5)
        assert reactant_geo[9, 0] == pytest.approx(1.33727, rel=1e-5)
        assert reactant_geo[9, 1] == pytest.approx(-1.59717, rel=1e-5)
        assert reactant_geo[9, 2] == pytest.approx(1.83977, rel=1e-5)

        assert product_geo[0, 0] == pytest.approx(-0.25240, rel=1e-5)
        assert product_geo[0, 1] == pytest.approx(1.11134, rel=1e-5)
        assert product_geo[0, 2] == pytest.approx(0.03026, rel=1e-5)
        assert product_geo[9, 0] == pytest.approx(-0.60766, rel=1e-5)
        assert product_geo[9, 1] == pytest.approx(-2.19988, rel=1e-5)
        assert product_geo[9, 2] == pytest.approx(-0.23692, rel=1e-5)

    def test_reaction3_xyz_parse(self, test_xyz_dir):
        xyz_file = test_xyz_dir / "reaction3.xyz"
        reactant_elements, reactant_geo, reactant_q, product_elements, product_geo, product_q = reaction_xyz_parse(str(xyz_file))

        assert reactant_elements == ["n", "o", "o", "o"]
        assert product_elements == ["n", "o", "o", "o"]
        assert reactant_geo.shape == (4, 3)
        assert product_geo.shape == (4, 3)
        assert reactant_q == pytest.approx(-1, rel=1e-5)
        assert product_q == pytest.approx(-1, rel=1e-5)

        assert reactant_geo[1, 0] == pytest.approx(1.24218, rel=1e-5)
        assert reactant_geo[1, 1] == pytest.approx(0.00000, rel=1e-5)
        assert reactant_geo[1, 2] == pytest.approx(0.00000, rel=1e-5)
        assert reactant_geo[3, 0] == pytest.approx(-0.62109, rel=1e-5)
        assert reactant_geo[3, 1] == pytest.approx(-1.07576, rel=1e-5)
        assert reactant_geo[3, 2] == pytest.approx(0.00000, rel=1e-5)

        assert product_geo[1, 0] == pytest.approx(1.25218, rel=1e-5)
        assert product_geo[1, 1] == pytest.approx(0.00000, rel=1e-5)
        assert product_geo[1, 2] == pytest.approx(0.00000, rel=1e-5)
        assert product_geo[3, 0] == pytest.approx(-0.64102, rel=1e-5)
        assert product_geo[3, 1] == pytest.approx(-1.07588, rel=1e-5)
        assert product_geo[3, 2] == pytest.approx(0.00000, rel=1e-5)

    def test_reaction4_xyz_parse(self, test_xyz_dir):
        xyz_file = test_xyz_dir / "reaction4.xyz"
        reactant_elements, reactant_geo, reactant_q, product_elements, product_geo, product_q = reaction_xyz_parse(str(xyz_file))

        assert reactant_elements == ["o", "c", "c", "o", "c", "c", "c", "c", "c", "h", "h", "h", "h", "h", "h", "h", "h", "h", "h"]
        assert product_elements == ["o", "c", "c", "o", "c", "c", "c", "c", "c", "h", "h", "h", "h", "h", "h", "h", "h", "h", "h"]
        assert reactant_geo.shape == (19, 3)
        assert product_geo.shape == (19, 3)
        assert reactant_q == pytest.approx(0, rel=1e-5)
        assert product_q == pytest.approx(0, rel=1e-5)

        assert reactant_geo[0, 0] == pytest.approx(-2.1948489, rel=1e-5)
        assert reactant_geo[0, 1] == pytest.approx(0.1648506, rel=1e-5)
        assert reactant_geo[0, 2] == pytest.approx(0.0001947, rel=1e-5)
        assert reactant_geo[18, 0] == pytest.approx(2.2861461, rel=1e-5)
        assert reactant_geo[18, 1] == pytest.approx(0.0419516, rel=1e-5)
        assert reactant_geo[18, 2] == pytest.approx(-0.0001223, rel=1e-5)

        assert product_geo[0, 0] == pytest.approx(-3.2453563, rel=1e-5)
        assert product_geo[0, 1] == pytest.approx(0.0109114, rel=1e-5)
        assert product_geo[0, 2] == pytest.approx(-0.1465026, rel=1e-5)
        assert product_geo[18, 0] == pytest.approx(-0.6064293, rel=1e-5)
        assert product_geo[18, 1] == pytest.approx(2.0484554, rel=1e-5)
        assert product_geo[18, 2] == pytest.approx(0.4130904, rel=1e-5)

    def test_reaction5_xyz_parse(self, test_xyz_dir):
        xyz_file = test_xyz_dir / "reaction5.xyz"
        reactant_elements, reactant_geo, reactant_q, product_elements, product_geo, product_q = reaction_xyz_parse(str(xyz_file))

        assert reactant_elements == ["n", "h", "h", "h", "h"]
        assert product_elements == ["n", "h", "h", "h", "h"]
        assert reactant_geo.shape == (5, 3)
        assert product_geo.shape == (5, 3)
        assert reactant_q == pytest.approx(1, rel=1e-5)
        assert product_q == pytest.approx(1, rel=1e-5)

        assert reactant_geo[0, 0] == pytest.approx(0.00040, rel=1e-5)
        assert reactant_geo[0, 1] == pytest.approx(-0.00000, rel=1e-5)
        assert reactant_geo[0, 2] == pytest.approx(-0.00000, rel=1e-5)
        assert reactant_geo[4, 0] == pytest.approx(-0.34185, rel=1e-5)
        assert reactant_geo[4, 1] == pytest.approx(-0.48402, rel=1e-5)
        assert reactant_geo[4, 2] == pytest.approx(-0.83835, rel=1e-5)

        assert product_geo[0, 0] == pytest.approx(0.00040, rel=1e-5)
        assert product_geo[0, 1] == pytest.approx(-0.00000, rel=1e-5)
        assert product_geo[0, 2] == pytest.approx(-0.00000, rel=1e-5)
        assert product_geo[4, 0] == pytest.approx(-0.34185, rel=1e-5)
        assert product_geo[4, 1] == pytest.approx(-0.50402, rel=1e-5)
        assert product_geo[4, 2] == pytest.approx(-0.83837, rel=1e-5)


class TestSMILESRxn:
    def test_initialize_smiles_reaction_file(self, test_smiles_file, capsys):
        reactions = load_reactions_from_smiles_file(test_smiles_file)
        captured = capsys.readouterr()

        assert f"Failed to initialize 3 reaction(s) from {test_smiles_file}:" in captured.out
        assert f"Line 5 in {test_smiles_file}: Unmapped smiles string. Please provide mapped reaction for this particular type of initialization" in captured.out
        assert "Line 6: No >> or more than 1 >>" in captured.out
        assert f"Line 7 in {test_smiles_file}: Mismatched atom mapping. Check again" in captured.out

        assert len(reactions) == 4

    def test_reaction1_smiles_parse(self, test_smiles_file):
        with open(test_smiles_file, "r") as f:
            reactant_smiles, product_smiles = [_.strip() for _ in f.readline().strip().split(">>")]

        reactant = yarpecule(reactant_smiles, mode="yarp", canon=False)
        product = yarpecule(product_smiles, mode="yarp", canon=False)
        reactant.get_smiles()
        product.get_smiles()

        assert reactant.elements == ['c', 'c', 'o', 'o', 'h', 'h', 'h', 'h', 'n', 'h', 'h', 'h']
        assert product.elements == ['c', 'c', 'o', 'h', 'h', 'h', 'n', 'h', 'h', 'o', 'h', 'h']
        assert reactant.map_smi == "[C:1]([C:2](=[O:3])[O:4][H:5])([H:6])([H:7])[H:8].[N:9]([H:10])([H:11])[H:12]"
        assert product.map_smi == "[C:1]([C:2]([O-:3])=[N+:9]([H:10])[H:11])([H:6])([H:7])[H:8].[O:4]([H:5])[H:12]"

    def test_reaction2_smiles_parse(self, test_smiles_file):
        with open(test_smiles_file, "r") as f:
            lines = f.readlines()
        reactant_smiles, product_smiles = [_.strip() for _ in lines[1].strip().split(">>")]

        reactant = yarpecule(reactant_smiles, mode="yarp", canon=False)
        product = yarpecule(product_smiles, mode="yarp", canon=False)
        reactant.get_smiles()
        product.get_smiles()

        assert reactant.elements == ['n', 'c', 'c', 'c', 'n', 'n', 'o', 'h', 'h', 'h']
        assert product.elements == ['n', 'c', 'c', 'c', 'n', 'n', 'o', 'h', 'h', 'h']
        assert reactant.map_smi == "[N:0]1([H:7])[C:1](=[O:6])[N:4]1[C@@:2]1([H:8])[C:3]([H:9])=[N:5]1"
        assert product.map_smi == "[n+:0]1([H:7])[c:1]([O-:6])[c:3]([H:9])[n:5][c:2]([H:8])[n:4]1"

    def test_reaction3_smiles_parse(self, test_smiles_file):
        with open(test_smiles_file, "r") as f:
            lines = f.readlines()
        reactant_smiles, product_smiles = [_.strip() for _ in lines[2].strip().split(">>")]

        reactant = yarpecule(reactant_smiles, mode="yarp", canon=False)
        product = yarpecule(product_smiles, mode="yarp", canon=False)
        reactant.get_smiles()
        product.get_smiles()

        assert reactant.elements == ['c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'n', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h']
        assert product.elements == ['c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'n', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h']
        assert reactant.map_smi == "[C:0]([C:3]([H:11])([H:12])[H:13])([C:4]([H:14])([H:15])[H:16])([C:5]([H:17])([H:18])[H:19])[N:8]1[C@@:1]([C:6]([H:20])([H:22])[H:23])([H:9])[C@@:2]1([C:7]([H:21])([H:24])[H:25])[H:10]"
        assert product.map_smi == "[C:0]([C:3]([H:11])([H:12])[H:13])([C:4]([H:14])([H:15])[H:16])([C:5]([H:17])([H:18])[H:19])/[N+:8](=[C:1](\\[C:6]([H:20])([H:22])[H:23])[H:9])[C-:2]([C:7]([H:21])([H:24])[H:25])[H:10]"

    def test_reaction4_smiles_parse(self, test_smiles_file):
        with open(test_smiles_file, "r") as f:
            lines = f.readlines()
        reactant_smiles, product_smiles = [_.strip() for _ in lines[3].strip().split(">>")]

        reactant = yarpecule(reactant_smiles, mode="yarp", canon=False)
        product = yarpecule(product_smiles, mode="yarp", canon=False)
        reactant.get_smiles()
        product.get_smiles()

        assert reactant.elements == ['o', 'c', 'c', 'o', 'c', 'h', 'h', 'h', 'h', 'c', 'c', 'c', 'c', 'h', 'h', 'h', 'h', 'h', 'h']
        assert product.elements == ['o', 'c', 'c', 'o', 'c', 'c', 'c', 'c', 'c', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h']
        assert reactant.map_smi == "[C:4](=[C:5]([C:6](=[C:7]([H:15])[H:16])[H:14])[H:13])([H:11])[H:12].[O:0]=[C:1]([C:2]([O:3][H:10])=[C:8]([H:17])[H:18])[H:9]"
        assert product.map_smi == "[O:0]=[C:1]([C@:2]1([O:3][H:10])[C:4]([H:11])([H:12])[C:5]([H:13])=[C:6]([H:14])[C:7]([H:15])([H:16])[C:8]1([H:17])[H:18])[H:9]"
