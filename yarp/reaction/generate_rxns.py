"""
Wrapper function to manage the generation of reaction objects during main_yarp routine
"""
import os
import fnmatch
import pickle
import numpy as np
from openbabel import pybel

from yarp.yarpecule.yarpecule import yarpecule
from yarp.reaction.reaction import reaction
from yarp.reaction.enum import enumerate_products
from yarp.reaction.filters import filter_enum_candidates, filter_enum_products
from yarp.util.write_files import mol_write_yp


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
    if inp.enum.ON:
        print("Product enumeration enabled. Enumerating products.")
        if inp.init_struct.type == 'yarp_pickle':
            if verbose:
                print(" - Processing starting node(s) as YARP generated pickle file")

            og_rxns = pickle.load(open(inp.init_struct.source, 'rb'))
            assert isinstance(og_rxns, dict), "Input pickle file must contain a dictionary!"
            assert all(isinstance(v, reaction) for v in og_rxns.values()), "YARP requires a dictionary of reaction objects to continue"

            og_rxns_hash = set(og_rxns.keys())

            candidates = filter_enum_candidates(
                og_rxns, separate_prods=inp.enum_filters.separate_prods,
                dG_cutoff=inp.enum_filters.dG_cutoff, dG_source=inp.enum_filters.dG_source,
                netconfig=inp.net_explore, verbose=verbose)

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
                    r2p = reaction(mol, prod)
                    p2r = reaction(mol, prod)

                    # Skip reactions already discovered (forward/reverse)
                    if r2p.hash in og_rxns_hash or p2r.hash in og_rxns_hash:
                        continue
                    new_rxns[r2p.hash] = r2p
            
            output = og_rxns | new_rxns
            
        else:
            if verbose:
                print(f" - Initializing starting reactant node from {inp.init_struct.source}")
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
                r2p = reaction(reactant, prod)
                output[r2p.hash] = r2p

    else:
        print(f"Product enumeration not enabled. Initializing reactions from input node(s).")
        if inp.init_struct.type == 'yarp_pickle':
            if verbose:
                print(" - Processing starting node(s) as YARP generated pickle file")

            og_rxns = pickle.load(open(inp.init_struct.source, 'rb'))
            assert isinstance(og_rxns, dict), "Input pickle file must contain a dictionary!"
            assert all(isinstance(v, reaction) for v in og_rxns.values()), "YARP requires a dictionary of reaction objects to continue"

            output = og_rxns
        else:
            raise RuntimeError("We can only start from a YARP pickle file currently, sorry friend!")

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
