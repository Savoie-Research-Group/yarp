import fnmatch
from yarp.yarpecule.yarpecule import yarpecule

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

    # Time to start copying over from main_xtb's run_enumeration function
    products = []

    return products
