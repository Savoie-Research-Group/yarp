import fnmatch
import numpy as np
from yarp.yarpecule.yarpecule import yarpecule
from yarp.reaction.enum import break_bonds, form_n_bonds, bmfn

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

        print("Enumeration ahoy!")

        if fnmatch.fnmatch(inp.d0_node, "*.p") or fnmatch.fnmatch(inp.d0_node, "*.pickle"):
            print("Processing starting node as YARP generated pickle file")
            raise RuntimeError("Not yet implemented!")
        else:
            print("Letting yarpecule object figure out what this is")
            molecule = yarpecule(inp.d0_node, mode="yarp")
            print(molecule.elements)
            print(molecule.adj_mat)

        products = enumerate_products(
            molecule, inp.n_break, inp.n_form, inp.enum_mode, inp.l_cutoff)
    else:
        raise RuntimeError("Non-enumeration routines are not yet implemented!")

    return output


def enumerate_products(r_yp, n_break, n_form, mode="concerted", cutoff=0.0):
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

    print(f"Product enumeration with break {n_break}, form {n_form} "
          f"will be performed in {mode} mode.")

    if mode == "sequential":
        print(f" * WARNING: Sequential mode is expensive and "
              "may cause memory blow-up issues!")

        # Break bonds
        break_mol = list(break_bonds(r_yp, n=n_break))
        print(f" - Breaking {n_break} bonds formed "
              f"{len(break_mol)} intermediates")

        # Form bonds
        products = form_n_bonds(break_mol, n=n_form)
        print(f" - Forming {n_form} bonds formed "
              f"{len(products)} potential products")

        # Filter out the garbage potential products
        products = [_ for _ in products if _.bond_mat_scores[0]
                    <= cutoff and sum(np.abs(_.fc)) < 2.0]

        # This makes no sense to me... figure it out later - ERM
        product = []
        for _ in products:
            if _.rings != []:
                if len(_.rings[0]) > 4:
                    product.append(_)
                else:
                    product.append(_)
        products = product
        print(f" - {len(products)} cleaned products after filtering")

    elif mode == "concerted":
        products = list(bmfn(r_yp, n_break, n_form))
        print(f" - Enumerated {len(products)} products")
    else:
        raise RuntimeError("Please select either concerted or sequential as the "
                           "product enumeration mode!")

    return products
