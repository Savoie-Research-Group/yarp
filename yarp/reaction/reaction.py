"""
Definition of the reaction object class.
"""

from yarp.reaction.node import node
from yarp.reaction.edge import edge
from yarp.reaction.conformer import select_conformer_pair


class reaction:

    def __init__(self, reactant, product):

        self.reactant = node(reactant)
        self.product = node(product)

        self.edge = edge()

        self.id = self.reactant.inchi + "_to_" + self.product.inchi

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
