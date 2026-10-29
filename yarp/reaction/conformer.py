"""
Definition of the conformer object class.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from yarp.util.constants import Constants
from yarp.util.write_files import xyz_generate_string, xyz_write

# Duplicate TS criteria (conformer.is_duplicate_of). Fixed constants, not user
# settings: in the ts_handoff survey, duplicates had an H-matched RMSD <= 0.018 A
# and |dE| <= 0.011 kcal/mol; the closest distinct TSs (H rotamers) 0.40 A and
# 0.057 kcal/mol.
DUP_RMSD_MAX = 0.1  # Angstrom, H-matched all-atom RMSD
DUP_DE_MAX = 0.03   # kcal/mol, electronic energy
# Mirror images count as duplicates. They have equal energies and lead to
# mirror-image endpoints, which reaction hashes can't tell apart yet. Switch this
# off once they see stereochemistry.
DUP_MATCH_MIRRORS = True

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

            self.type = f'conf_gen_rank{rank}_{self.lot}_{self.software}'
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

    def is_duplicate_of(self, other):
        """
        Checks if two TS conformers of one reaction (same atom order) are the same
        structure: electronic energies within DUP_DE_MAX and an H-matched RMSD
        within DUP_RMSD_MAX, mirror images included if DUP_MATCH_MIRRORS. A
        conformer without an energy is never a duplicate.
        Returns bool.
        """
        e_self = self.properties.get("internal_energy_Eh")
        e_other = other.properties.get("internal_energy_Eh")
        if e_self is None or e_other is None or list(self.elements) != list(other.elements):
            return False
        if abs(e_self - e_other) * Constants.ha_to_kcalmol > DUP_DE_MAX:
            return False

        geo_self, geo_other = np.asarray(self.geo, float), np.asarray(other.geo, float)
        candidates = [geo_other, geo_other * [-1, 1, 1]] if DUP_MATCH_MIRRORS else [geo_other]
        return min(_h_matched_rmsd(self.elements, geo_self, g) for g in candidates) <= DUP_RMSD_MAX

    def to_xyz_string(self):
        """
        Returns the geometry formatted as a standard XYZ string, useful for 
        passing to external scripts or writing scratch files.
        """
        if self.geo is not None and len(self.elements) > 0:
            return xyz_generate_string(elements=self.elements, geo=self.geo)
        else:
            return None

    def export_geometry(self, filename):
        """
        Export the conformer's geometry to an xyz file.
        This shouldn't ever change any of the attributes of the conformer.

        Parameters
        ----------
        filename : str
            The name of the file to export the geometry to.
        """
        xyz_write(filename, self.elements, self.geo, comment=self.type)


def unique_ts(confs):
    """
    Drops duplicate TS conformers (conformer.is_duplicate_of) from a list of
    (key, conformer) pairs, keeping the order. Of each set of duplicates the
    lowest-energy one is kept (ties: the first in order); each dropped one gets
    properties['duplicate_of'] = the kept one's key.
    Returns the kept conformers.
    """
    def energy(kc):
        e = kc[1].properties.get("internal_energy_Eh")
        return (e is None, 0.0 if e is None else e)

    kept = {}
    for k, conf in sorted(confs, key=energy):
        match = next((kk for kk, kc in kept.items() if conf.is_duplicate_of(kc)), None)
        if match is None:
            kept[k] = conf
        elif conf.properties.get("duplicate_of") != match:
            conf.properties["duplicate_of"] = match
            print(f"     * TS {k} is a duplicate of TS {match}: skipped.")
    return [conf for k, conf in confs if k in kept]


def _fit(a, b):
    """Proper rotation r and centroids ca, cb that best map a onto b: (a - ca) @ r + cb."""
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1, 1, d]) @ vt, ca, cb


def _rmsd(a, b):
    r, ca, cb = _fit(a, b)
    return float(np.sqrt(np.mean(np.sum(((a - ca) @ r + cb - b) ** 2, axis=1))))


def _h_matched_rmsd(elements, a, b):
    """
    All-atom RMSD of geometries a and b after pairing each H of a with the
    nearest free H of b, once a is fitted onto b by its heavy atoms. Equivalent
    H atoms (as on a CH2) can swap labels between otherwise identical TSs. With
    fewer than 3 heavy atoms there is no fit to pair them in, so this is the
    plain all-atom RMSD.
    """
    heavy = np.array([str(e).lower() != "h" for e in elements])
    h = np.where(~heavy)[0]
    if heavy.sum() < 3 or len(h) == 0:
        return _rmsd(a, b)
    r, ca, cb = _fit(a[heavy], b[heavy])
    a_fit = (a - ca) @ r + cb
    rows, cols = linear_sum_assignment(np.linalg.norm(a_fit[h][:, None] - b[h][None], axis=2))
    order = np.arange(len(elements))
    order[h[rows]] = h[cols]
    return _rmsd(a, b[order])
