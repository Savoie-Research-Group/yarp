"""
Wrapper function to manage the generation of reaction objects during main_yarp routine
"""
import os
import fnmatch
import pickle
import numpy as np
from pathlib import Path
from openbabel import pybel
from rdkit import Chem

from yarp.yarpecule.yarpecule import yarpecule
from yarp.reaction.reaction import reaction
from yarp.reaction.enum import enumerate_products
from yarp.reaction.filters import filter_enum_candidates, filter_enum_products
from yarp.util.write_files import mol_write_yp


def _resolve_xyz_pair(directory):
    xyz_files = sorted(
        [
            path for path in Path(directory).iterdir()
            if path.is_file() and path.suffix.lower() == ".xyz"
        ]
    )
    reactant_matches = [path for path in xyz_files if "reactant" in path.name.lower()]
    product_matches = [path for path in xyz_files if "product" in path.name.lower()]

    if (
        len(reactant_matches) == 1
        and len(product_matches) == 1
        and reactant_matches[0] != product_matches[0]
    ):
        return reactant_matches[0], product_matches[0]

    if len(xyz_files) == 2:
        return xyz_files[0], xyz_files[1]

    raise RuntimeError(
        "Could not uniquely identify a reactant/product XYZ pair in "
        f"'{directory}'. Provide exactly one file containing 'reactant' and one containing "
        "'product' (case-insensitive), or provide exactly two .xyz files total."
    )


def _build_direct_reaction(reactant_source, product_source, source_label):
    reactant = yarpecule(str(reactant_source), mode="yarp", canon=False)
    product = yarpecule(str(product_source), mode="yarp", canon=False)

    if len(reactant.elements) != len(product.elements):
        raise RuntimeError(
            "Direct reaction initialization requires reactant and product to have the same atom count. "
            f"Got {len(reactant.elements)} atoms for reactant '{reactant_source}' and "
            f"{len(product.elements)} atoms for product '{product_source}' from {source_label}. "
            "For XYZ inputs, YARP assumes the user has already ensured matching atom ordering."
        )

    return reaction(reactant, product)


def _load_reaction_from_xyz_directory(directory):
    reactant_path, product_path = _resolve_xyz_pair(directory)
    rxn = _build_direct_reaction(reactant_path, product_path, f"directory '{directory}'")
    return {rxn.hash: rxn}


def _parse_mapped_smiles(smiles, role, line_number, source_path):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise RuntimeError(
            f"Could not parse {role} SMILES on line {line_number} of '{source_path}': {smiles}"
        )

    if any(not atom.HasProp("molAtomMapNumber") for atom in mol.GetAtoms()):
        raise RuntimeError(
            f"{role.capitalize()} SMILES on line {line_number} of '{source_path}' must be fully atom-mapped."
        )

    maps = [int(atom.GetProp("molAtomMapNumber")) for atom in mol.GetAtoms()]
    if len(maps) != len(set(maps)):
        dupes = sorted({atom_map for atom_map in maps if maps.count(atom_map) > 1})
        raise RuntimeError(
            f"{role.capitalize()} SMILES on line {line_number} of '{source_path}' contains duplicate atom maps: {dupes}"
        )

    return set(maps)


def _load_reactions_from_smiles_file(source_path):
    output = {}

    with open(source_path, "r") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.count(">>") != 1:
                raise RuntimeError(
                    f"Reaction SMILES line {line_number} of '{source_path}' must contain exactly one '>>': {line}"
                )

            reactant_smiles, product_smiles = [part.strip() for part in line.split(">>")]
            if not reactant_smiles or not product_smiles:
                raise RuntimeError(
                    f"Reaction SMILES line {line_number} of '{source_path}' must include both reactant and product."
                )

            reactant_maps = _parse_mapped_smiles(reactant_smiles, "reactant", line_number, source_path)
            product_maps = _parse_mapped_smiles(product_smiles, "product", line_number, source_path)
            if reactant_maps != product_maps:
                raise RuntimeError(
                    f"Reaction SMILES line {line_number} of '{source_path}' has mismatched atom-map sets. "
                    f"Reactant maps: {sorted(reactant_maps)}; Product maps: {sorted(product_maps)}"
                )

            rxn = _build_direct_reaction(
                reactant_smiles, product_smiles, f"line {line_number} of '{source_path}'"
            )
            if rxn.hash in output:
                print(
                    f" - Skipping duplicate direct reaction on line {line_number} of '{source_path}' "
                    f"(hash {rxn.hash})"
                )
                continue

            output[rxn.hash] = rxn

    if not output:
        raise RuntimeError(f"No direct reactions were found in '{source_path}'.")

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

    # Initialize reactions for product enumeration
    if inp.enum.enumerate:

        print("Product enumeration routine selected")
        if fnmatch.fnmatch(inp.d0_node, "*.p") or fnmatch.fnmatch(inp.d0_node, "*.pickle") or fnmatch.fnmatch(inp.d0_node, "*.pkl"):
            print(" - Processing starting node(s) as YARP generated pickle file")

            og_rxns = pickle.load(open(inp.d0_node, 'rb'))
            assert isinstance(og_rxns, dict), "Input pickle file must contain a dictionary!"
            assert all(isinstance(v, reaction) for v in og_rxns.values()), "YARP requires a dictionary of reaction objects to continue"

            og_rxns_hash = set(og_rxns.keys())

            candidates = filter_enum_candidates(
                og_rxns, separate_prods=inp.enum_filters.separate_prods,
                dG_cutoff=inp.enum_filters.dG_cutoff, dG_source=inp.enum_filters.dG_source,
                netconfig=inp.net_explore
            )

            new_rxns = dict()
            for mol in candidates:
                print(f" - Enumerating from {mol.inchi} ({mol.canon_smi}) node")
                raw_products = enumerate_products(
                    r_yp=mol, n_break=inp.enum.n_break, n_form=inp.enum.n_form,
                    react=inp.enum.react_atoms, mode=inp.enum.mode
                )

                clean_products = filter_enum_products(
                    raw_products, l_cutoff=inp.enum_filters.l_cutoff,
                    fc_cutoff=inp.enum_filters.fc_cutoff, ring_filter=inp.enum_filters.ring_filter
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
            print(f" - Initializing starting reactant node from {inp.d0_node}")
            output = dict()
            reactant = yarpecule(inp.d0_node, mode="yarp")

            raw_products = enumerate_products(
                    r_yp=reactant, n_break=inp.enum.n_break, n_form=inp.enum.n_form,
                    react=inp.enum.react_atoms, mode=inp.enum.mode
                )

            clean_products = filter_enum_products(
                raw_products, l_cutoff=inp.enum_filters.l_cutoff,
                fc_cutoff=inp.enum_filters.fc_cutoff, ring_filter=inp.enum_filters.ring_filter
            )

            for prod in clean_products:
                r2p = reaction(reactant, prod)
                output[r2p.hash] = r2p

    else:
        print("Loading reactions")
        if fnmatch.fnmatch(inp.d0_node, "*.p") or fnmatch.fnmatch(inp.d0_node, "*.pickle") or fnmatch.fnmatch(inp.d0_node, "*.pkl"):
            print(" - Processing starting node(s) as YARP generated pickle file")

            og_rxns = pickle.load(open(inp.d0_node, 'rb'))
            assert isinstance(og_rxns, dict), "Input pickle file must contain a dictionary!"
            assert all(isinstance(v, reaction) for v in og_rxns.values()), "YARP requires a dictionary of reaction objects to continue"

            output = og_rxns
        else:
            source_path = Path(inp.d0_node)
            if source_path.is_dir():
                print(f" - Loading direct reaction from XYZ directory {source_path}")
                output = _load_reaction_from_xyz_directory(source_path)
            elif source_path.is_file():
                print(f" - Loading direct reaction(s) from mapped SMILES file {source_path}")
                output = _load_reactions_from_smiles_file(source_path)
            else:
                raise RuntimeError(
                    "When enumeration is disabled, 'initial species' must point to a YARP pickle file, "
                    "an XYZ directory, or a mapped reaction-SMILES text file."
                )

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
