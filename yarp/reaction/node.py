"""
Definition of the node object class.
"""


class node:
    """
    Attributes:
    -----------

    graph : yarpecule object
        Yarpecule object that contains the molecular graph.
        This is provided upon initialization, and is typically not modified.

    conformers : list of conformer objects
    """

    def __init__(self, yp):

        self.graph = yp

        # Ehhh, now that I'm thinking about it, why not just make these yarpecule attributes?
        # Come back to this later, it might make more sense here, but we'll see
        self.smiles = self.graph.get_smiles()
        self.inchi = self.graph.get_inchi()

        self.conformers = []

    def gen_conformers(self, input):

        # Initialize a calculator object to make a call out to CREST or RDKit
        # Placeholder for actual conformer generation
        self.conformers = ["conf1", "conf2", "conf3"]

    def refine_node(self, input):
        # Initialize a calculator object to perform a geometry optimization (local minima)
        # This will probably just be a light wrapper to do this in the conformer class, actually
        pass
