"""
Definition of the reaction object class.
"""
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

    product : state object
        The product-side state of the reaction.

    ts_geom : dict
        Transition state geometry of the reaction.
        Keys correspond to '{lot}-{software}' or specific stage tags (e.g., 'tsguess').

    barrier : dict
        Energy of activation barrier (dG) of the reaction R --> P (kcal/mol).

    reverse_barrier : dict
        Energy of activation barrier (dG) of the reaction P --> R (kcal/mol).

    heat_of_rxn : dict
        Heat of reaction (dH) of the reaction R --> P (kcal/mol).

    dg_rxn : dict
        Change in Gibbs free energy (dG) of the reaction R --> P (kcal/mol).
        Computed as reverse_barrier - barrier (for EGAT)

    id : str
        Human-readable name of reaction used to generate folders/files.

    hash : str/float
        Unique identifier for a reaction object.
        
    outcome_label : dict
        Intended/unintended logic labels for specific levels of theory.
        
    network_meta : dict
        Placeholder for storing metadata related to network generation.
    """

    def __init__(self, reactant, product):
        # Geometries
        self.reactant = state(reactant)
        self.product = state(product)

        self.ts_geom = dict()

        # Set up bond change information
        self.bond_changes = self.reactant.graph.adj_mat - self.product.graph.adj_mat
        self.reactant.paired_bem = self.product.graph.bond_mats[0]
        self.product.paired_bem = self.reactant.graph.bond_mats[0]

        # Properties
        self.barrier = dict()
        self.reverse_barrier = dict()
        self.heat_of_rxn = dict()
        self.dg_rxn = dict()

        # Identifiers & Metadata
        self.id = self.reactant.inchi + "_to_" + self.product.inchi
        self.hash = reaction_hash(self)
        
        self.outcome_label = dict()
        self.network_meta = dict()
