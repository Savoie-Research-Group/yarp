import os
import fnmatch
import numpy as np
from openbabel import pybel

from yarp.yarpecule.yarpecule import yarpecule
from yarp.reaction.reaction import reaction
from yarp.reaction.enum import break_bonds, form_n_bonds, bmfn
from yarp.util.misc import mol_write_yp

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

        if fnmatch.fnmatch(inp.d0_node, "*.p") or fnmatch.fnmatch(inp.d0_node, "*.pickle"):
            print(" - Processing starting node as YARP generated pickle file")
            raise RuntimeError("Not yet implemented!")
        else:
            print(f" - Initializing starting reactant node from {inp.d0_node}")
            reactant = yarpecule(inp.d0_node, mode="yarp")

        products = enumerate_products(
            reactant, inp.n_break, inp.n_form, inp.enum_mode, inp.l_cutoff)

        for prod in products:
            # Do a quick optimization to make product geometries reflect new bonding
            prod = quick_geom_opt(prod, inp.quick_opt_lot)

            # Generate a reaction object from reactant/product pairs
            rxn = reaction(reactant, prod)

            # Add reaction to dictionary paired with its ID
            output[rxn.id] = rxn

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


def enumerate_products(r_yp, n_break, n_form, mode="concerted", cutoff=0.0, ring_mode=False):
    """
    r_yp : yarpecule object
        The reactant from which all products are enumerated

    n_break : int
        Number of bonds to break

    n_form : int
        Number of bonds to form

    mode : string
        Toggle between the two available product enumeration modes:
        Concerted (default) and sequential enumeration.

    cutoff : float
        Threshold used in sequential enumeration to discard unphysical Lewis structures
        with bond-electron matrix scores above this value.
    """

    print(f" - Product enumeration with break {n_break}, form {n_form} "
          f"will be performed in {mode} mode.")

    if mode == "sequential":
        print(f"   WARNING: Sequential mode is expensive and "
              "may cause memory blow-up issues!")

        # Break bonds
        break_mol = list(break_bonds(r_yp, n=n_break))
        print(f"   + Breaking {n_break} bonds formed "
              f"{len(break_mol)} intermediates")

        # Form bonds
        products = form_n_bonds(break_mol, n=n_form)
        print(f"   + Forming {n_form} bonds formed "
              f"{len(products)} potential products")

        # Filter out the garbage potential products
        products = [_ for _ in products if _.bond_mat_scores[0]
                    <= cutoff and sum(np.abs(_.fc)) < 2.0]

        # This makes no sense to me... it only seems to throw away
        # ring-open structures... - ERM
        if ring_mode:
            product = []
            for _ in products:
                if _.rings != []:
                    if len(_.rings[0]) > 4:
                        product.append(_)
                    else:
                        product.append(_)
            products = product
        print(f"   + {len(products)} cleaned products after filtering")

    elif mode == "concerted":
        products = list(bmfn(r_yp, n_break, n_form))
        print(f"   + Enumerated {len(products)} products")
    else:
        raise RuntimeError("Please select either concerted or sequential as the "
                           "product enumeration mode!")

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
