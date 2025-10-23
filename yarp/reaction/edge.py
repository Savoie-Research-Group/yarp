"""
Definition of the edge object class.
"""
from copy import deepcopy

class edge:

    def __init__(self, r_node, p_node):

        self.start = deepcopy(r_node)
        self.end = deepcopy(p_node)

        self.barrier = dict() # key: level of theory, value: dG kcal/mol
        self.reverse_barrier = dict()
        self.flux = None

        self.ts = None
        self.path_coords = []

    def refine_edge(self):

        # If IRC is turned off, only update the self.ts attribute
        # If IRC is turned on, update the self.ts and the self.path_coords attributes

        pass
