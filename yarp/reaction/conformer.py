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
        Some label that indicates the level of theory used to generate the conformer's data.

    type : str
        Some label that indicates the sort of structure this conformer is.
        Current thoughts: "local minimum" for reactant/product, "saddle point" for transition states,
        and "black magic" for conical intersections
    """

    def __init__(self, calc_type=None, calc_data=None):
        self.lot = ""
        self.type = ""
        self.geo = None
        self.elements = []
        self.properties = {
            "internal_energy_Eh": 0.0,
            "Gibbs_free_energy_kcal_per_mol": 0.0,
            "heat_of_formation_0K_kcal_per_mol": 0.0,
            "heat_of_formation_298K_kcal_per_mol": 0.0,
            "standard_entropy_kcal_per_mol": 0.0,
            "heat_capacity_joule_per_K": 0.0 # assume starting temp/pressure of 25C and 1 atm?
        }

        if calc_type is not None and calc_data is not None:
            self.update_from_calc(calc_type, calc_data)
    
    def update_from_calc(self, calc_type, calc_data):
        # read in data from a completed calculation, and scrape all the important stuff out
        # Have a distinct routine to scrape data from a yarpecule class!
        pass


