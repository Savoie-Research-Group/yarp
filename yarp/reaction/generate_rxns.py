"""
Wrapper function to manage the generation of reaction objects during main_yarp routine
"""
import pickle
import numpy as np
from pathlib import Path

from yarp.yarpecule.yarpecule import yarpecule
from yarp.yarpecule.input_parsers import load_reaction_from_xyz_file, load_reactions_from_xyz_directory, load_reactions_from_smiles_file
from yarp.reaction.reaction import reaction
from yarp.reaction.enum import enumerate_products
from yarp.reaction.filters import filter_enum_candidates, filter_enum_products
from yarp.util.rdkit import rdkit_ff_opt
from yarp.util.obabel import obabel_ff_opt
from yarp.yarpecule.graph.adjacency import table_generator


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
    source = Path(inp.init_struct.source)

    # Initialize reactions for product enumeration
    if inp.enum.ON:
        print("Product enumeration enabled. Enumerating products.")

        # Enumerating from single starting species
        if inp.init_struct.mode == 'species':
            if verbose:
                print(f" - Initializing starting reactant node from {source}")
                print(f" - Processing starting node as a single species. No pre-enumeration filters will be applied!")

            output = dict()
            reactant = yarpecule(inp.init_struct.source, mode="yarp")

            raw_products = enumerate_products(
                    r_yp=reactant, n_break=inp.enum.n_break, n_form=inp.enum.n_form,
                    react=inp.enum.react_atoms, mode=inp.enum.mode, verbose=verbose
                )

            clean_products = filter_enum_products(
                raw_products, l_cutoff=inp.enum.post_enum_filters.lewis_score,
                fc_cutoff=inp.enum.post_enum_filters.formal_charge, ring_filter=inp.enum.post_enum_filters.ring_filter,
                verbose=verbose
            )

            for prod in clean_products:
                prod = quick_geom_opt(prod)
                if prod is None:
                    if verbose:
                        print(f"  + SKIPPED! Unable to form valid product ({prod.canon_smi}) geom from reactant ({mol.canon_smi}) geom")
                    continue
                r2p = reaction(reactant, prod)
                output[r2p.hash] = r2p

        # Enumerating from reaction object(s)
        if inp.init_struct.mode == 'reaction':
            if verbose:
                print(f" - Initializing starting reactant node(s) from {inp.init_struct.source}")
                print(f" - Initial source identified as reactant(s); pre-enumeration filters will be applied!")

            # 3 different modes for initializing reaction object(s)
            og_rxns = None
            if inp.init_struct.type == 'yarp_pickle':
                if verbose: print(" - Processing starting node(s) as YARP generated pickle file")

                og_rxns = pickle.load(open(inp.init_struct.source, 'rb'))
                assert isinstance(og_rxns, dict), "Input pickle file must contain a dictionary!"
                assert all(isinstance(v, reaction) for v in og_rxns.values()), "YARP requires a dictionary of reaction objects to continue"

            elif inp.init_struct.type == 'xyz':
                if source.is_dir():
                    if verbose: print(f" - Processing starting node(s) as reaction xyz files in {source}")
                    og_rxns = load_reactions_from_xyz_directory(source)
                elif source.is_file() and source.suffix.lower() == ".xyz":
                    if verbose: print(f" - Processing starting node(s) as reaction xyz file {source}")
                    rxn = load_reaction_from_xyz_file(source)
                    og_rxns = {rxn.hash: rxn}

            elif inp.init_struct.type == 'smiles':
                if verbose: print(f" - Processing starting node(s) as mapped reaction SMILES in {source}")
                og_rxns = load_reactions_from_smiles_file(source)

            else:
                raise RuntimeError("We can only start from a YARP pickle file, a reaction xyz file, a directory of reaction xyz files, or a mapped reaction SMILES file currently, sorry friend!")

            og_rxns_hash = set(og_rxns.keys())

            candidates = filter_enum_candidates(
                og_rxns, separate_prods=inp.enum.pre_enum_filters.separate_prods,
                prop_filter=inp.enum.pre_enum_filters.property_filter,
                netconfig=inp.enum.pre_enum_filters.product_blinders, verbose=verbose)

            new_rxns = dict()
            for mol in candidates:
                if verbose:
                    print(f" - Enumerating from {mol.inchi} ({mol.canon_smi}) node")

                # Reactive atom maps are resolved inside enumerate_products
                # against this final candidate graph. That is deliberately
                # later than candidate filtering/product separation, because a
                # split fragment can have a smaller atom-map subset and a new
                # local atom index order.
                raw_products = enumerate_products(
                    r_yp=mol, n_break=inp.enum.n_break, n_form=inp.enum.n_form,
                    react=inp.enum.react_atoms, mode=inp.enum.mode, verbose=verbose,
                )

                clean_products = filter_enum_products(
                    raw_products, l_cutoff=inp.enum.post_enum_filters.lewis_score,
                    fc_cutoff=inp.enum.post_enum_filters.formal_charge, ring_filter=inp.enum.post_enum_filters.ring_filter,
                    verbose=verbose
                )

                for prod in clean_products:
                    prod = quick_geom_opt(prod)
                    if prod is None:
                        if verbose:
                            print(f"  + SKIPPED! Unable to form valid product ({prod.canon_smi}) geom from reactant ({mol.canon_smi}) geom")
                        continue
                    r2p = reaction(mol, prod)
                    p2r = reaction(mol, prod)

                    # Skip reactions already discovered (forward/reverse)
                    if r2p.hash in og_rxns_hash or p2r.hash in og_rxns_hash:
                        continue
                    new_rxns[r2p.hash] = r2p
            
            output = og_rxns | new_rxns

    # Initialize reactions characterization without product enumeration
    else:
        print(f"Product enumeration not enabled. Initializing reactions from input node(s).")
        print(f" - Input node source: {source}")

        # 3 different modes for initializing reaction object(s)
        output = None
        if inp.init_struct.type == 'yarp_pickle':
            if verbose: print(" - Processing starting node(s) as YARP generated pickle file")

            output = pickle.load(open(inp.init_struct.source, 'rb'))
            assert isinstance(output, dict), "Input pickle file must contain a dictionary!"
            assert all(isinstance(v, reaction) for v in output.values()), "YARP requires a dictionary of reaction objects to continue"

        elif inp.init_struct.type == 'xyz':
            if source.is_dir():
                if verbose: print(f" - Processing starting node(s) as reaction xyz files in {source}")
                output = load_reactions_from_xyz_directory(source)
            elif source.is_file() and source.suffix.lower() == ".xyz":
                if verbose: print(f" - Processing starting node(s) as reaction xyz file {source}")
                rxn = load_reaction_from_xyz_file(source)
                output = {rxn.hash: rxn}

        elif inp.init_struct.type == 'smiles':
            if verbose: print(f" - Processing starting node(s) as mapped reaction SMILES in {source}")
            output = load_reactions_from_smiles_file(source)

        else:
            raise RuntimeError("We can only start from a YARP pickle file, a reaction xyz file, a directory of reaction xyz files, or a mapped reaction SMILES file currently, sorry friend!")

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

    # First, attempt to optimize with RDKit
    rd_opt_g = rdkit_ff_opt(molecule, lot=lot)

    # Check if optimization preserved starting connectivity
    rd_adj = table_generator(molecule.elements, rd_opt_g)
    rd_diff = rd_adj - molecule.adj_mat

    # If RDKit generated a garbage geom, try Open Babel
    if not np.all(rd_diff == 0):
        ob_opt_g = obabel_ff_opt(molecule, lot=lot)

        # If Open Babel fails too, we return None
        ob_adj = table_generator(molecule.elements, ob_opt_g)
        ob_diff = ob_adj - molecule.adj_mat
        if not np.all(ob_diff == 0):
            return None
        
        # If all goes well, update geometry and return
        molecule._geo = ob_opt_g
        return molecule

    # Otherwise, if RDKit gave a valid geom, use that one
    else:
        molecule._geo = rd_opt_g
        return molecule
