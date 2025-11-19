"""
Definition of the state object class.
"""
from copy import deepcopy


class state:
    """
    Attributes:
    -----------

    graph : yarpecule object
        Yarpecule object that contains the molecular graph of all molecules in state.
        This is provided upon initialization, and is typically not modified.

    conformers : list of conformer objects
        Conformers correspond to all molecules contained in state?

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

        self.conformers = []

        self._species = self._graph.separate(canon=canon)
        self.conc = dict()
        for _ in self.species:
            _.get_smiles()
            self.conc[_.canon_smi] = 0.0


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
    def bond_mats(self):
        return self._graph.bond_mats

    @property
    def species(self):
        return self._species

    def gen_conformers(self, input):

        # Initialize a calculator object to make a call out to CREST or RDKit
        # Placeholder for actual conformer generation
        self.conformers = ["conf1", "conf2", "conf3"]

    def refine(self, input):
        # Initialize a calculator object to perform a geometry optimization (local minima)
        # This will probably just be a light wrapper to do this in the conformer class, actually
        pass
