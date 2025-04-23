
def generate_rxns(init):
    """
    init : dict
        literally the stuff contained under "initialize" in the input YAML file

    Returns a dictionary of reaction objects
    """
    output = dict()

    if "enumerate" in init:
        print("Product enumeration routine selected!")

        if isinstance(init.get('reactants'), str):
            print(f"Starting from reactant: {init.get('reactants')}")
            # Do product enumeration and then create a bunch of reaction objects
            output = enumerate_products(init)
        else:
            raise RuntimeError(
                "Product enumeration can only be done from a single reactant!")

    else:
        print("No product enumeration will be performed.")
        if "yarp pickle" in init:
            print("Loading reaction objects from prior YARP run.")
            # Load in prior reaction objects from a pickle file
            output = load_from_pickle(init.get('yarp pickle'))
        elif "reactants" in init & "products" in init:
            print("Loading reaction objects from reactants and products.")
            # Create reaction objects from user provided reactants and products
            # I imagine this will be something that `enumerate_products()` calls internally
            output = create_reaction(
                init.get('reactants'), init.get('products'))
        else:
            raise RuntimeError(
                "No reactants or products provided for reaction object generation.")

    return output
