"""
Definition of the conformer object class.
"""
import numpy as np
from yarp.util.write_files import xyz_generate_string

class conformer:
    """
    Base class representing a single 3D geometry of a state or transition state.

    Attributes:
    -----------
    geo : numpy.ndarray
        An (N_atom, 3) array containing the cartesian coordinates of each atom in the molecule.
        Units are in Angstroms.

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
        self.geo = None
        self.elements = []

        self.lot = ""
        self.software = ""
        self.type = ""
        
        self.vibrational_freqs = None
        self.hessian = None
        self.imaginary_freq_mode = None

        self.properties = {
            "internal_energy_Eh": None,
            "gibbs_free_energy_kcal_per_mol": None,
            "enthalpy_kcal_per_mol": None,
            "entropy_temp_kcal_per_mol": None
        }

        if calc_type is not None and calc_data is not None:
            self.update_from_calc(calc_type, calc_data)
    
    def update_from_calc(self, calc_type, calc_data):
        """
        General parser router. Depending on the calc_type (e.g., 'xtb_opt', 'g16_freq'),
        it routes the calc_data (raw text or parsed dict) to the appropriate internal updater.
        """

        if calc_type == "yarpecule":
            self.geo = calc_data.geo
            self.elements = calc_data.elements
            self.type = "initial_mol_graph"
        elif calc_type == 'conf_gen':
            self.geo = calc_data.get('geometry')
            self.elements = calc_data.get('elements')
            self.lot = calc_data.get('lot')
            self.software = calc_data.get('software')

            rank = calc_data.get('conf_rank')

            self.type = f'conf_gen_rank{rank}_{self.software}_{self.lot}'
        else:
            pass

    def is_valid_minimum(self, tolerance=-1e-2):
        """
        Checks if the structure is a valid local minimum (i.e., exactly zero imaginary frequencies).
        Returns bool.
        """
        if self.vibrational_freqs is None: return False

        n_imaginary = np.sum(self.vibrational_freqs < tolerance)
        return n_imaginary == 0

    def is_valid_ts(self, tolerance=-1e-2):
        """
        Checks if the structure is a valid 1st order saddle point (i.e., exactly one imaginary frequency).
        Returns bool.
        """
        if self.vibrational_freqs is None: return False

        n_imaginary = np.sum(self.vibrational_freqs < tolerance)
        return n_imaginary == 1
    
    def to_xyz_string(self):
        """
        Returns the geometry formatted as a standard XYZ string, useful for 
        passing to external scripts or writing scratch files.
        """
        if self.geo is not None and len(self.elements) > 0:
            return xyz_generate_string(elements=self.elements, geo=self.geo)
        else:
            return None


