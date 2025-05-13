"""
Definition of the conformer class and related helper functions.
"""


class conformer:
    """
    Attributes:
    -----------
    geo : numpy array
        3D cartesian coordinates of the conformer.

    elements : list of str
        List of elements in the conformer.

    properties : dict
        Computed properties of the conformer, such as Gibbs free energy, enthalpy, etc.
        I think it makes sense to group them together in a dictionary, rather than having them individually added
        as attributes. This way, we can easily add new properties without modifying the class.

    lot : str
        Some label that indicates the level of theory used to generate the conformer.
        Not sure if this should be put here, or in the higher level classes that contain conformers.
    """

    def __init__(self):

        self.geo = None
        self.elements = []
        self.properties = dict()
        self.lot = ""


def select_conformer_pair(r, p, inp):
    pass
