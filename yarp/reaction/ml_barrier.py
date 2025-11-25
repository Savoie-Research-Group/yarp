"""
Placeholder for code allowing for ML predicted reaction barriers (and other reaction properties?)
"""


def get_egat_barriers(yp_rxns, model):
    """
    yp_rxns : dict
        Dictionary of reaction class objects (values) stored by reaction hash (key)

    model : ???
        Loaded pytorch model
    """

    dataframe = []
    for rxn in yp_rxns.values():
        rsmiles = rxn.reactant.map_smi
        psmiles = rxn.product.map_smi

    # dataframe to CSV file

    # Fead CSV file into model

    # Update reaction objects with EGAT barriers
    for rxn in yp_rxns.values():
        rxn.barrier['egat'] = 42.0

    return 0


