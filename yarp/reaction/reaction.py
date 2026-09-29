"""
Definition of the reaction object class.
"""
import warnings
from copy import deepcopy

import numpy as np

from yarp.reaction.state import state
from yarp.yarpecule.hashes import bmat_hash, reaction_hash

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
        self.hash = reaction_hash(self)
        
        self.outcome_label = dict()
        self.network_meta = dict()

    ######################
    # Internal Functions #
    ######################

    def _validate_reaction(self):
        """
        Validate atom balance and maps, then align product indices if needed.

        Element-inconsistent mappings are reported but accepted because atom
        maps are treated as user-supplied correspondence labels.
        """
        reactant = self.reactant.graph
        product = self.product.graph

        # Check that the reactant and product adjacency matrices have the same shape.
        if reactant.adj_mat.shape != product.adj_mat.shape:
            raise ValueError(
                "Reactant and product adjacency matrices must have the same shape."
            )

        # Enforce that the reactant and product have the same element composition.
        if sorted(reactant.elements) != sorted(product.elements):
            raise ValueError(
                "Reactant and product must contain the same element composition. "
                "While in principle we find nuclear chemistry exciting, it is not yet fully supported in YARP."
            )

        # Check that the reactant and product have the same unique atom-map sets.
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
            warnings.warn(
                "Element-inconsistent atom maps detected "
                f"({mismatch_text}). Check the maps for this reaction; "
                "while exciting in principle, nuclear chemistry is not yet fully supported.",
                RuntimeWarning,
                stacklevel=2,
            )

        order = [product_by_map[atom_map] for atom_map in reactant_maps]
        if order != list(range(len(order))):
            self._align_product(order)

    def _align_product(self, order):
        """Apply one atom permutation to the product state and its cached data."""
        graph = self.product.graph
        old_to_new = {old: new for new, old in enumerate(order)}
        matrix_order = np.ix_(order, order)
        graph._adj_mat = graph.adj_mat[matrix_order]
        graph._elements = [graph.elements[i] for i in order]
        for attr in ("_geo", "_masses", "_atom_hashes"):
            setattr(graph, attr, getattr(graph, attr)[order])
        graph._atom_info = {i: deepcopy(graph.atom_info[j]) for i, j in enumerate(order)}
        for info in graph._atom_info.values():
            bonds = info["stereo"]["bonds"]
            info["stereo"]["bonds"] = {old_to_new[j]: value for j, value in bonds.items()}

        lewis = graph.lewis
        lewis._adj_mat = graph._adj_mat
        lewis._elements = graph._elements
        lewis._bond_mats = [mat[matrix_order] for mat in lewis._bond_mats]
        lewis._rings = [[old_to_new[i] for i in ring] for ring in lewis._rings]
        for attr in ("_e_acceptors", "_e_donors", "_formal_charge"):
            setattr(lewis, attr, getattr(lewis, attr)[order])
        lewis._atom_neighbors = [
            {old_to_new[j] for j in lewis._atom_neighbors[i]} for i in order
        ]
        graph._bond_order_dict = {
            old_to_new[i]: {old_to_new[j]: value for j, value in neighbors.items()}
            for i, neighbors in graph._bond_order_dict.items()
        }
        bem = np.zeros_like(lewis._bond_mats[0])
        for mat in lewis._bond_mats:
            bem += mat
        graph._bem_sum_hash = bmat_hash(bem)
        initial = self.product.conformers["initial_geom"]
        initial.geo = graph.geo
        initial.elements = graph.elements
