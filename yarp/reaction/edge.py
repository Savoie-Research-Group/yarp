"""
Definition of the edge object class.
"""


class edge:

    def __init__(self, reactant, product, input):

        self.ts = None
        self.path_coords = []

    def refine_edge(self, input):

        # If IRC is turned off, only update the self.ts attribute
        # If IRC is turned on, update the self.ts and the self.path_coords attributes

        pass
