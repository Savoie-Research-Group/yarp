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
        self._validate_reaction()

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
        self.hash = reaction_hash(self, _validated=True)
        
        self.outcome_label = dict()
        self.network_meta = dict()

    ######################
    # Internal Functions #
    ######################

    def _validate_reaction(self):
        """
        Validate that the reactant and product are atom-balanced and have
        consistent atom-map sets.

        Element-inconsistent mappings are reported but accepted because atom
        maps are treated as user-supplied correspondence labels.
        """
        reactant = self.reactant.graph
        product = self.product.graph

        if reactant.adj_mat.shape != product.adj_mat.shape:
            raise ValueError(
                "Reactant and product adjacency matrices must have the same shape."
            )

        # Endpoint atom order may differ.
        if sorted(reactant.elements) != sorted(product.elements):
            raise ValueError(
                "Reactant and product must contain the same element composition."
            )

        reactant_maps = [
            reactant.atom_info[i]["atom_map"]
            for i in range(len(reactant.elements))
        ]
        product_maps = [
            product.atom_info[i]["atom_map"]
            for i in range(len(product.elements))
        ]

        if (
            None in reactant_maps
            or None in product_maps
            or len(set(reactant_maps)) != len(reactant_maps)
            or len(set(product_maps)) != len(product_maps)
            or set(reactant_maps) != set(product_maps)
        ):
            raise ValueError(
                "Reaction endpoints require identical unique atom-map sets."
            )

        product_by_map = {
            atom_map: i for i, atom_map in enumerate(product_maps)
        }

        element_mismatches = [
            (
                atom_map,
                reactant.elements[reactant_index],
                product.elements[product_by_map[atom_map]],
            )
            for reactant_index, atom_map in enumerate(reactant_maps)
            if (
                reactant.elements[reactant_index]
                != product.elements[product_by_map[atom_map]]
            )
        ]

        if element_mismatches:
            mismatch_text = ", ".join(
                f"map {atom_map}: {reactant_element}->{product_element}"
                for atom_map, reactant_element, product_element
                in element_mismatches
            )
            print(
                "WARNING: Element-inconsistent atom maps detected "
                f"({mismatch_text}). Check the maps for this reaction; "
                "while exciting in principle, nuclear chemistry is not yet fully supported."
            )
