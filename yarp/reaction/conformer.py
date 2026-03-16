"""
Definition of the conformer class and related helper functions.
"""

class conformer:
    """
    Base class representing a single 3D geometry of a state or transition state.

    Attributes:
    -----------
    geo : numpy.ndarray
        3D cartesian coordinates of the conformer (N x 3).

    elements : list of str
        List of atomic symbols in the conformer.

    properties : dict
        Computed thermodynamic and electronic properties. 
        Stores energies (internal, Gibbs, enthalpy), entropies, and frequencies.

    lot : str
        Level of theory used to generate this specific geometry and its properties 
        (e.g., 'xtb', 'b3lyp/def2svp').

    software : str
        The software package used to compute the conformer (e.g., 'crest', 'gaussian').

    type : str
        Categorical label of the structure type.
        Examples: 'reactant-minimum', 'product-minimum', 'ts-guess', 'ts-optimized', 'irc-endpoint'.
        
    imaginary_freqs : list of float
        List of imaginary frequencies (if any). Crucial for validating transition states.
    """

    def __init__(self, calc_type=None, calc_data=None):
        self.lot = ""
        self.software = ""
        self.type = ""
        
        self.geo = None
        self.elements = []
        self.imaginary_freqs = []

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
        """
        General parser router. Depending on the calc_type (e.g., 'xtb_opt', 'g16_freq'),
        it routes the calc_data (raw text or parsed dict) to the appropriate internal updater.
        """
        pass

    def is_valid_minimum(self):
        """
        Checks if the structure is a valid local minimum (i.e., exactly zero imaginary frequencies).
        Returns bool.
        """
        pass

    def is_valid_ts(self):
        """
        Checks if the structure is a valid saddle point (i.e., exactly one imaginary frequency).
        Returns bool.
        """
        pass
    
    def to_xyz_string(self):
        """
        Returns the geometry formatted as a standard XYZ string, useful for 
        passing to external scripts or writing scratch files.
        """
        pass


