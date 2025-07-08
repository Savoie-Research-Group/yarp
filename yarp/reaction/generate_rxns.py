import fnmatch
import numpy as np
from yarp.yarpecule.yarpecule import yarpecule
from yarp.reaction.enum import break_bonds, form_n_bonds, bmfn

# NOTE: Long-term I'd like to have all the default user inputs and error handling set up in a different centralized location
# But I need to recruit some users and devs to help me hammer this out well


def generate_rxns(init):
    """
    init : dict
        literally the stuff contained under "initialize" in the input YAML file

    Returns a dictionary of reaction objects

    Should this be a class?
    """
    output = dict()

    print("Received initialization node:")
    print(init)

    # Check product enumeration status (default to False if not provided)
    enum_flag = init.get("enumerate", False)

    # Initialize reactions for product enumeration
    if enum_flag:

        print("Enumeration ahoy!")
        # Check formatting for initial species
        d0_node = init.get("initial species", None)
        if not d0_node:
            raise RuntimeError("Please provide an initial species for enumeration. "
                               "Can be a single structure SMILES or XYZ, "
                               "or a previous YARP pickle file.")

        if fnmatch.fnmatch(d0_node, "*.p") or fnmatch.fnmatch(d0_node, "*.pickle"):
            print("Processing starting node as YARP generated pickle file")
            raise RuntimeError("Not yet implemented!")
        else:
            print("Letting yarpecule object figure out what this is")
            molecule = yarpecule(d0_node, mode="yarp")
            print(molecule.elements)
            print(molecule.adj_mat)

        products = enumerate_products(molecule, init)
    else:
        raise RuntimeError("Non-enumeration routines are not yet implemented!")

    return output


def enumerate_products(r_yp, inp):
    """
    r_yp : yarpecule object
        The reactant from which all products are enumerated
    """
    mode = inp.get("mode", "concerted")
    n_break = inp.get("bonds to break", 2)
    n_form = inp.get("bonds to form", 2)
    cutoff = inp.get("lewis score", 0.0)

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
