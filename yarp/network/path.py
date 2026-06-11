"""
Definition of the path object class.
"""
from copy import deepcopy

class path:

    def __init__(self, initial_species, catalysts=[]):
        self.start = deepcopy(initial_species)
        self.end = None
        self.catalysts = catalysts

        self.steps = []
    