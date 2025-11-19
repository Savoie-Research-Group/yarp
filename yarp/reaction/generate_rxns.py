import os
import fnmatch
import pickle
import numpy as np
from openbabel import pybel

from yarp.yarpecule.yarpecule import yarpecule
from yarp.reaction.reaction import reaction
from yarp.reaction.enum import break_bonds, form_n_bonds, bmfn
from yarp.util.write_files import mol_write_yp

# NOTE: Long-term I'd like to have all the default user inputs and error handling set up in a different centralized location
# But I need to recruit some users and devs to help me hammer this out well


def generate_rxns(inp):
    """
    init : dict
        literally the stuff contained under "initialize" in the input YAML file

    Returns a dictionary of reaction objects

    Should this be a class?
    """
    output = dict()

    # Initialize reactions for product enumeration
    if inp.enum_on:

        print("Product enumeration routine selected")

        if fnmatch.fnmatch(inp.d0_node, "*.p") or fnmatch.fnmatch(inp.d0_node, "*.pickle") or fnmatch.fnmatch(inp.d0_node, "*.pkl"):
            print(" - Processing starting node(s) as YARP generated pickle file")

            og_rxns = pickle.load(open(inp.d0_node, 'rb'))
            assert isinstance(og_rxns, dict), "Input pickle file must contain a dictionary!"
            assert all(isinstance(v, reaction) for v in og_rxns.values()), "YARP requires a dictionary of reaction objects to continue"

            print(f" - Reading in {len(og_rxns)} total reactions")

            if inp.separate_prods == 'all':
                print(f" - Performing product separation on all reactions prior to enumeration")
            elif isinstance(inp.separate_prods, list) and len(inp.separate_prods) > 0:
                print(f" - Separating products for reaction indexes: {inp.separate_prods}")
            else:
                print(" - No product separation will be performed prior to enumeration")

            if isinstance(inp.dG_source, str):
                print(f" - Barrier filtering selected! Reactions with {inp.dG_source} dG above {inp.dG_cutoff} kcal/mol will be excluded from enumeration.")
            else:
                print(" - No barrier filtering will be performed prior to enumeration")

            # Check that new enumeration nodes have not already been enumerated from
            # For this, it should be enough to ensure that a product doesn't appear as a reactant
            # But maybe I'll need to re-evaluate this beyond depth2...
            p_nodes = []
            p_hashes = set()
            for count_r, rxn in enumerate(og_rxns.values()):
                # Apply dG cutoff
                if isinstance(inp.dG_source, str):
                    dG = rxn.barrier.get(inp.dG_source, None)
                    if dG is not None and dG > inp.dG_cutoff:
                        print(f"  + Excluding {rxn.id} (dG = {dG}) from enumeration")
                        continue

                # Add old reactions to output
                output[rxn.hash] = rxn

                # Ensure products have never been enumerated as reactants before
                r_hash = rxn.reactant.hash

                # Product separation check
                prod = rxn.product.graph
                if inp.separate_prods == 'all':
                    sep_mols = prod.separate()
                    if len(sep_mols) > 1:
                        print(f"  + Separating {prod.inchi} into {len(sep_mols)} nodes")

                    for mol in sep_mols:
                        p_hash = mol.hash
                        if p_hash not in p_hashes: # also ensure no duplicate products!?
                            p_hashes.add(p_hash)
                            mol.get_inchi() # need to recompute this, since fresh yarpecule
                            p_nodes.append(mol)

                elif isinstance(inp.separate_prods, list) and len(inp.separate_prods) > 0:
                    sep_targets = set(inp.separate_prods)
                    if count_r in sep_targets:
                        sep_mols = prod.separate()
                        print(f"  + Separating {prod.inchi} into {len(sep_mols)} nodes")

                        for mol in sep_mols:
                            p_hash = mol.hash
                            if p_hash not in p_hashes: # also ensure no duplicate products!?
                                p_hashes.add(p_hash)
                                mol.get_inchi()
                                p_nodes.append(mol)

                    else:
                        # Move forward without separating products
                        p_hash = prod.hash
                        if p_hash not in p_hashes: # also ensure no duplicate products!?
                            p_hashes.add(p_hash)
                            p_nodes.append(prod)

                else:
                    # Move forward without separating products
                    p_hash = prod.hash
                    if p_hash not in p_hashes: # also ensure no duplicate products!?
                        p_hashes.add(p_hash)
                        p_nodes.append(prod)

            print(f" - {len(p_nodes)} unique products identified for enumeration")

            # Do the enumeration thing!
            for node in p_nodes:
                print(f" - Enumerating from {node.inchi} node")
                products = enumerate_products(
                    node, inp.n_break, inp.n_form, react=inp.react_atoms, mode=inp.enum_mode,
                    l_cutoff=inp.l_cutoff, fc_cutoff=inp.fc_cutoff, ring_filter=inp.ring_filter
                )

                for prod in products:
                    # prod = quick_geom_opt(prod, inp.quick_opt_lot)
                    rxn = reaction(node, prod)
                    output[rxn.hash] = rxn
            
        else:
            print(f" - Initializing starting reactant node from {inp.d0_node}")
            reactant = yarpecule(inp.d0_node, mode="yarp")

            products = enumerate_products(
                reactant, inp.n_break, inp.n_form, react=inp.react_atoms, mode=inp.enum_mode,
                l_cutoff=inp.l_cutoff, fc_cutoff=inp.fc_cutoff, ring_filter=inp.ring_filter
            )

            for i, prod in enumerate(products):
                # Do a quick optimization to make product geometries reflect new bonding
                # prod = quick_geom_opt(prod, inp.quick_opt_lot)

                # Generate a reaction object from reactant/product pairs
                rxn = reaction(reactant, prod)

                # Add reaction to dictionary paired with its ID
                output[rxn.hash] = rxn

    else:
        raise RuntimeError("Non-enumeration routines are not yet implemented!")

    # If visualization mode is ON, then dump bond electron matrix drawings to a visuals folder
    if inp.prod_visuals_on:
        for index, rxn in enumerate(output.values()):
            folder = f"rxn_visuals/rxn{index}_{rxn.id}"
            os.makedirs(folder, exist_ok=True)

            rxn.reactant.graph.draw_bmats(f"{folder}/reactant_bemat.pdf")
            rxn.product.graph.draw_bmats(f"{folder}/product_bemat.pdf")

    return output


def enumerate_products(r_yp, n_break, n_form, react=[], mode="concerted", l_cutoff=0.0, fc_cutoff=2.0, ring_filter=False):
    """
    r_yp : yarpecule object
        The reactant from which all products are enumerated

    n_break : int
        Number of bonds to break

    n_form : int
        Number of bonds to form

    react : set (default = None)
        When supplied this is used to restrict bond formations only to those atoms in this set.
        If supplied, then `react` must have a searchable list or set
        (i.e., the function uses an `in` call, so sets are better) per `yarpecule`.
        An empty list is interpreted as all atoms being available to react. 

    mode : string
        Toggle between the two available product enumeration modes:
        concerted (default) and sequential enumeration.

    l_cutoff : float (default = 0.0)
        Threshold used in sequential enumeration to discard unphysical Lewis structures
        with bond-electron matrix scores above this value.

    fc_cutoff : float (default = 2.0)
        Threshold used in sequential enumeration to discard unphysical Lewis structures
        with total formal charges at or above this value.

    ring_filter : bool (default = False)
        Filter out 3 and 4 member rings from enumerated products.
    """

    print(f"  * Product enumeration with break {n_break}, form {n_form} "
          f"will be performed in {mode} mode.")

    if react != []:
        react_list = list(react[0])
        element_list = []
        for i in react_list:
            element_list.append(r_yp.elements[i])
        print(f"   + Reactive atoms defined as: index {react_list} --> element {element_list}")

    if mode == "sequential":
        print(f"   WARNING: Sequential mode is expensive and "
              "may cause memory blow-up issues!")

        # Break bonds
        break_mol = list(break_bonds(r_yp, n=n_break, react=react))
        print(f"   + Breaking {n_break} bonds formed "
              f"{len(break_mol)} intermediates")

        # Form bonds
        products = form_n_bonds(break_mol, n=n_form, react=react, hashes={r_yp.hash})
        print(f"   + Forming {n_form} bonds formed "
              f"{len(products)} potential products")

    elif mode == "concerted":
        products = list(bmfn(r_yp, n_break, n_form, hashes={r_yp.hash}, react=react))
        print(f"   + Enumerated {len(products)} products")

    else:
        raise RuntimeError("Please select either concerted or sequential as the product enumeration mode!")

    # Filter out the garbage potential products according to Lewis threshold
    print(f"   + Applying Lewis score cutoff of {l_cutoff}")
    products = [_ for _ in products if _.bond_mat_scores[0] <= l_cutoff]

    # Filter out garbage potential products according to formal charge
    print(f"   + Applying formal charge cutoff of {fc_cutoff}")
    products = [_ for _ in products if sum(np.abs(_.fc)) < fc_cutoff]

    # Filter out 3 and 4 member rings from potential products
    if ring_filter:
        print(f"   + Removing 3 and 4 member rings")
        product = []
        for _ in products:
            if _.rings != []:
                if len(_.rings[0]) > 4:
                    product.append(_)
            else:
                product.append(_)
        products = product

    print(f"   + Returning {len(products)} products after filtering")

    return products


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
