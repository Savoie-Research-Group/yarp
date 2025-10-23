"""
Definition of the node object class.
"""
from copy import deepcopy


class node:
    """
    Attributes:
    -----------

    graph : yarpecule object
        Yarpecule object that contains the molecular graph.
        This is provided upon initialization, and is typically not modified.

    conformers : list of conformer objects
    """

    def __init__(self, yp, canon=False):

        self.graph = deepcopy(yp)
        self.graph.get_smiles()
        self.graph.get_inchi()

        self.species = self.graph.separate(canon=canon)
        self.conc = dict()
        for _ in self.species:
            _.get_smiles()
            self.conc[_.canon_smi] = 0.0


        self.conformers = []

    ###############
    # Properties  #
    ###############
    @property
    def inchi(self):
        return self.graph.inchi
    
    @property
    def canon_smi(self):
        return self.graph.canon_smi
    
    @property
    def map_smi(self):
        return self.graph.map_smi

    def gen_conformers(self, input):

        # Initialize a calculator object to make a call out to CREST or RDKit
        # Placeholder for actual conformer generation
        self.conformers = ["conf1", "conf2", "conf3"]

    def refine_node(self, input):
        # Initialize a calculator object to perform a geometry optimization (local minima)
        # This will probably just be a light wrapper to do this in the conformer class, actually
        pass
