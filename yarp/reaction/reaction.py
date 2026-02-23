"""
Definition of the reaction object class.
"""


from yarp.reaction.conformer import select_conformer_pair
from yarp.reaction.state import state

from yarp.yarpecule.hashes import reaction_hash

class reaction:
    """
    Base class for describing a reaction in YARP

    Parameters:
    -----------
    reactant : yarpecule
        Molecular graph of all species involved in the reactant-side state of the reaction

    product : yarpecule
        Molecular graph of all species involved in the product-side state of the reaction


    Attributes:
    -----------

    reactant : state object
        The reactant-side state of the reaction.
        Contains both molecular graph and conformer information.

    product : state object
        The product-side state of the reaction.
        Contains both molecular graph and conformer information.

    ts : dict
        Transition state geometry of the reaction.
        Key corresponds to level of theory, while the value should hold
        the actual 3D conformer geometry found at said level of theory.

    barrier : dict
        Energy of activation barrier (dG) of the reaction R --> P.
        Key corresponds to level of theory. dG values should be stored
        in units of kcal/mol.

    reverse_barrier : dict
        Energy of activation barrier (dG) of the reaction P --> R.
        Key corresponds to level of theory. dG values should be stored
        in units of kcal/mol.

    heat_of_reaction : dict
        Heat of reaction (dH) of the reaction R --> P.
        Key corresponds to level of theory. dH values should be stored
        in units of kcal/mol.

    max_flux : float
        Maximum reaction flux (units?) of the reaction R --> P.
        Initialized to zero, and intended as a placeholder to be updated
        by microkinetics run on a reaction network.

    id : str
        Name of reaction used to generate folders/files related to the reaction.
        This is NOT unique, but gives a reasonably human-readable label.

    hash : float
        Unique identifier for a reaction object.

    """

    def __init__(self, reactant, product):

        self.reactant = state(reactant)
        self.product = state(product)

        self.ts = dict()

        self.barrier = dict()
        self.reverse_barrier = dict()

        self.heat_of_rxn = dict()

        self.id = self.reactant.inchi + "_to_" + self.product.inchi
        self.hash = reaction_hash(self)
        
        self.network_meta = dict()  # placeholder for storing metadata related to reaction network generation + subnetwork kinetics

    def gen_initial_path(self, input):

        self.reactant.gen_conformers(input)
        self.product.gen_conformers(input)

        select_conformer_pair(self.reactant, self.product, input)

        # self.path = edge(self.reactant, self.product, input)

    def refine_reaction(self, input):

        self.path.refine_edge(input)
        self.reactant.refine_node(input)
        self.product.refine_node(input)

    def refine_ts(self, input):

        self.path.refine_edge(input)  # put the IRC toggle inside this function

    def refine_reactant(self, input):

        self.reactant.refine_node(input)

    def refine_product(self, input):
        # This is a small DRY violation, so revisit later
        self.product.refine_node(input)
