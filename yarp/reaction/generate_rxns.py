"""
Wrapper function to manage the generation of reaction objects during main_yarp routine
"""
import os
import fnmatch
import pickle
import numpy as np
from openbabel import pybel
from pathlib import Path
from rdkit import Chem

from yarp.yarpecule.yarpecule import yarpecule
from yarp.yarpecule.input_parsers import reaction_xyz_parse
from yarp.yarpecule.graph.adjacency import table_generator
from yarp.reaction.reaction import reaction
from yarp.reaction.enum import enumerate_products
from yarp.reaction.filters import filter_enum_candidates, filter_enum_products
from yarp.util.write_files import mol_write_yp


def _load_reaction_from_xyz_file(xyz_file):
    reactant_elements, reactant_geo, reactant_q, product_elements, product_geo, product_q = reaction_xyz_parse(str(xyz_file))

    reactant_adj = table_generator(reactant_elements, reactant_geo)
    product_adj = table_generator(product_elements, product_geo)

    reactant = yarpecule((reactant_adj, reactant_geo, reactant_elements, reactant_q), canon=False)
    product = yarpecule((product_adj, product_geo, product_elements, product_q), canon=False)

    return reaction(reactant, product)


def _load_reactions_from_xyz_directory(xyz_dir):
    xyz_files = sorted([_ for _ in xyz_dir.iterdir() if _.is_file() and _.suffix.lower() == ".xyz"])

    if len(xyz_files) == 0:
        raise RuntimeError(f"No xyz reaction files were found in {xyz_dir}.")

    output = dict()
    for xyz_file in xyz_files:
        rxn = _load_reaction_from_xyz_file(xyz_file)
        output[rxn.hash] = rxn

    return output


def _parse_mapped_reaction_smiles(smiles, line_number, source_path):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)

    if mol is None:
        raise RuntimeError(
            f"Line {line_number} in {source_path}: could not parse reaction SMILES."
        )

    atom_maps = []
    for atom in mol.GetAtoms():
        atom_props = atom.GetPropsAsDict()
        if "molAtomMapNumber" not in atom_props:
            raise RuntimeError(
                f"Line {line_number} in {source_path}: Unmapped smiles string. Please provide mapped reaction for this particular type of initialization"
            )
        atom_map = int(atom_props["molAtomMapNumber"])
        atom_maps.append(atom_map)

    if len(atom_maps) != len(set(atom_maps)):
        raise RuntimeError(
            f"Line {line_number} in {source_path}: Mismatched atom mapping. Check again"
        )

    return atom_maps


def _load_reactions_from_smiles_file(source_path):
    output = dict()

    with open(source_path, 'r') as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if len(line) == 0 or line.startswith("#"):
                continue

            if line.count(">>") != 1:
                raise RuntimeError(
                    f"Line {line_number} in {source_path}: No >> or more than 1 >>"
                )

            reactant_smiles, product_smiles = [_.strip() for _ in line.split(">>")]
            reactant_maps = _parse_mapped_reaction_smiles(reactant_smiles, line_number, source_path)
            product_maps = _parse_mapped_reaction_smiles(product_smiles, line_number, source_path)

            if set(reactant_maps) != set(product_maps):
                raise RuntimeError(
                    f"Line {line_number} in {source_path}: Mismatched atom mapping. Check again"
                )

            reactant = yarpecule(reactant_smiles, mode="yarp", canon=False)
            product = yarpecule(product_smiles, mode="yarp", canon=False)

            rxn = reaction(reactant, product)
            output[rxn.hash] = rxn

    return output


def generate_rxns(inp):
    """
    Wrapper function to manage the generation of reaction objects during main_yarp routine

    Parameters:
    -----------
    inp : InputParser object
        Settings parsed from user provided input file.
        TO-DO: Take advantage of more modular InputParser object and feed in only the pieces of
        the input file that are relevant to this part of the code.
        Will allow for more straightforward testing of this function.

    Returns:
    --------
    output : dict of reaction objects
        Reactant-product pairs contained in a reaction object and ready for further processing!
    """
    output = None
    verbose = inp.verbose

    # Initialize reactions for product enumeration
    if inp.enum.enumerate:
        print("Product enumeration enabled. Enumerating products.")
        if fnmatch.fnmatch(inp.d0_node, "*.p") or fnmatch.fnmatch(inp.d0_node, "*.pickle") or fnmatch.fnmatch(inp.d0_node, "*.pkl"):
            if verbose:
                print(" - Processing starting node(s) as YARP generated pickle file")

            og_rxns = pickle.load(open(inp.d0_node, 'rb'))
            assert isinstance(og_rxns, dict), "Input pickle file must contain a dictionary!"
            assert all(isinstance(v, reaction) for v in og_rxns.values()), "YARP requires a dictionary of reaction objects to continue"

            og_rxns_hash = set(og_rxns.keys())

            candidates = filter_enum_candidates(
                og_rxns, separate_prods=inp.enum_filters.separate_prods,
                dG_cutoff=inp.enum_filters.dG_cutoff, dG_source=inp.enum_filters.dG_source,
                netconfig=inp.net_explore, verbose=verbose
            )

            new_rxns = dict()
            for mol in candidates:
                if verbose:
                    print(f" - Enumerating from {mol.inchi} ({mol.canon_smi}) node")
                raw_products = enumerate_products(
                    r_yp=mol, n_break=inp.enum.n_break, n_form=inp.enum.n_form,
                    react=inp.enum.react_atoms, mode=inp.enum.mode, verbose=verbose
                )

                clean_products = filter_enum_products(
                    raw_products, l_cutoff=inp.enum_filters.l_cutoff,
                    fc_cutoff=inp.enum_filters.fc_cutoff, ring_filter=inp.enum_filters.ring_filter, verbose=verbose
                )

                for prod in clean_products:
                    prod = quick_geom_opt(prod)
                    r2p = reaction(mol, prod)
                    p2r = reaction(mol, prod)

                    # Skip reactions already discovered (forward/reverse)
                    if r2p.hash in og_rxns_hash or p2r.hash in og_rxns_hash:
                        continue
                    new_rxns[r2p.hash] = r2p
            
            output = og_rxns | new_rxns
            
        else:
            if verbose:
                print(f" - Initializing starting reactant node from {inp.d0_node}")
            output = dict()
            reactant = yarpecule(inp.d0_node, mode="yarp")

            raw_products = enumerate_products(
                    r_yp=reactant, n_break=inp.enum.n_break, n_form=inp.enum.n_form,
                    react=inp.enum.react_atoms, mode=inp.enum.mode, verbose=verbose
                )

            clean_products = filter_enum_products(
                raw_products, l_cutoff=inp.enum_filters.l_cutoff,
                fc_cutoff=inp.enum_filters.fc_cutoff, ring_filter=inp.enum_filters.ring_filter, verbose=verbose
            )

            for prod in clean_products:
                r2p = reaction(reactant, prod)
                output[r2p.hash] = r2p

    else:
        print(f"Product enumeration not enabled. Initializing reactions from input node(s).")
        if fnmatch.fnmatch(inp.d0_node, "*.p") or fnmatch.fnmatch(inp.d0_node, "*.pickle") or fnmatch.fnmatch(inp.d0_node, "*.pkl"):
            if verbose:
                print(" - Processing starting node(s) as YARP generated pickle file")

            og_rxns = pickle.load(open(inp.d0_node, 'rb'))
            assert isinstance(og_rxns, dict), "Input pickle file must contain a dictionary!"
            assert all(isinstance(v, reaction) for v in og_rxns.values()), "YARP requires a dictionary of reaction objects to continue"

            output = og_rxns
        elif Path(inp.d0_node).is_dir():
            print(f" - Processing starting node(s) as reaction xyz files in {inp.d0_node}")
            output = _load_reactions_from_xyz_directory(Path(inp.d0_node))
        elif Path(inp.d0_node).is_file():
            print(f" - Processing starting node(s) as mapped reaction SMILES in {inp.d0_node}")
            output = _load_reactions_from_smiles_file(inp.d0_node)
        else:
            raise RuntimeError("We can only start from a YARP pickle file, a directory of reaction xyz files, or a mapped reaction SMILES file currently, sorry friend!")

    return output


def quick_geom_opt(molecule, lot="uff"):
    '''
    Perform low-level level geometry optimization on yarpecule using openbabel.

    ERM: Can we just change the forcefield from UFF if we want?

    Parameters:
    ----------
    molecule : yarpecule object
        molecule to be optimized 

    lot : string
        Level of theory used for quick optimization

    Returns
    -------
    molecule : yarpecule object
        optimized molecule
    '''

    # Write yarpecule object to a temporary mol file
    mol_file = '.tmp.mol'
    mol_write_yp(mol_file, molecule.elements, molecule.geo,
                 molecule.bond_mats[0], molecule.adj_mat)

    # Use openbabel to perform geometry optimization
    mol = next(pybel.readfile("mol", mol_file))
    mol.localopt(forcefield=lot)

    # Update yarpecule with optimized geometry coordinates
    for count_i, i in enumerate(molecule.geo):
        molecule.geo[count_i] = mol.atoms[count_i].coords

    # Delete temporary mol file
    os.system("rm {}".format(mol_file))

    return molecule
