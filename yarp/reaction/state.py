"""
Definition of the state object class.
"""
from copy import deepcopy

import numpy as np

from yarp.reaction.conformer import conformer

class state:
    """
    Attributes:
    -----------

    graph : yarpecule object
        Yarpecule object that contains the molecular graph of all molecules in state.
        This is provided upon initialization, and is typically not modified.

    conformers : dict of conformer objects
        Conformers correspond to all molecules contained in state (for now...)
        Keys indicate where in the process a given conformer was generated.
        Doesn't need to indicate the specific level-of-theory used,
        as that should be stored within the conformer object itself.
        The conformer stored under the key "final" will be used in downstream
        reaction characterization processes

    paired_bem : 2D numpy array
        Bond electron matrix to be used in the generation of joint optimized conformers

    species : list of yarpecule objects
        Separated molecular graphs of each species contained in state.

    conc : dict
        Concentrations of species predicted from microkinetic models and the like.
        Keys correspond to canonical SMILES of species in state.
        Values are initialized to zero, and the intent is that
        these values can be modified as needed for network analysis.        
    """

    def __init__(self, yp, canon=False):

        self._graph = deepcopy(yp)
        self._graph.get_smiles()
        self._graph.get_inchi()

        # This duplication of data seems worth it for the downstream clarity
        self.conformers = {
            "initial_geom": conformer(calc_type="yarpecule", calc_data=self._graph)
        }
        self.paired_bem = None

        self._species = self._graph.separate(canon=canon)

        # ERM: We probably want to rethink this...
        # Can this be integrated with rxn.network_meta somehow?
        self.conc = dict()
        for _ in self.species:
            _.get_smiles()
            self.conc[_.canon_smi] = 0.0


    def set_graph_geometry(self, geo):
        """
        Replace the coordinates held by this state's yarpecule.

        The pre-optimization writes back a geometry that sits on the graph's
        own connectivity, so a later enumeration cycle -- which builds its
        parents from `rxn.product.graph` -- starts from relaxed, on-graph
        coordinates rather than the ones a product inherited from its parent.

        The array is replaced, not written into. `conformers["initial_geom"]`
        holds the *same* array object as the graph when the state is built, so
        an in-place write would silently rewrite the starting geometry too.
        """
        geo = np.array(geo, dtype=float, copy=True)
        expected = (len(self._graph.elements), 3)
        if geo.shape != expected:
            raise ValueError(f"Geometry has shape {geo.shape}; this state's graph needs {expected}.")
        self._graph._geo = geo

    ###############
    # Properties  #
    ###############
    @property
    def graph(self):
        return self._graph

    @property
    def inchi(self):
        return self._graph.inchi

    @property
    def canon_smi(self):
        return self._graph.canon_smi

    @property
    def map_smi(self):
        return self._graph.map_smi

    @property
    def hash(self):
        return self._graph.hash

    @property
    def bem_sum_hash(self):
        return self._graph.bem_sum_hash

    @property
    def identity(self):
        """
        Key identifying this state as a specific molecule with a specific atom
        indexing.

        `hash` alone is mapping-independent, so two states that are the same
        molecule under different atom mappings share it. Anything keyed on
        `hash` that then exchanges index-ordered data -- geometries, paired
        BEMs, bond_changes -- will silently mix the two. Measured over the KHP
        cycle-2 collection: 1110 products span 712 distinct (graph, mapping)
        pairs but only 268 distinct yarpecule hashes.
        """
        return (self._graph.hash, self._graph.bem_sum_hash)

    @property
    def bond_mats(self):
        return self._graph.bond_mats

    @property
    def species(self):
        return self._species

