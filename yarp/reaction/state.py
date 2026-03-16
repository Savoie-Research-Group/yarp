"""
Definition of the state object class.
"""
from copy import deepcopy

from yarp.reaction.conformer import conformer
from yarp.reaction.external.calculator import CrestCalculator


class state:
    """
    Attributes:
    -----------

    graph : yarpecule object
        Yarpecule object that contains the molecular graph of all molecules in state.
        This is provided upon initialization, and is typically not modified.

    conformers : dict of conformer objects
        Conformers correspond to all molecules contained in state (for now...)
        Keys indicate where in the process a given conformer was generated.
        Doesn't need to indicate the specific level-of-theory used,
        as that should be stored within the conformer object itself.
        The conformer stored under the key "final" will be used in downstream
        reaction characterization processes

    paired_bem : 2D numpy array
        Bond electron matrix to be used in the generation of joint optimized conformers

    species : list of yarpecule objects
        Separated molecular graphs of each species contained in state.

    conc : dict
        Concentrations of species predicted from microkinetic models and the like.
        Keys correspond to canonical SMILES of species in state.
        Values are initialized to zero, and the intent is that
        these values can be modified as needed for network analysis.        
    """

    def __init__(self, yp, canon=False):

        self._graph = deepcopy(yp)
        self._graph.get_smiles()
        self._graph.get_inchi()

        # This duplication of data seems worth it for the downstream clarity
        self.conformers = {
            "initial_geom": conformer(calc_type="yarpecule", calc_data=self._graph)
        }
        self.paired_bem = None

        self._species = self._graph.separate(canon=canon)
        # We probably want to rethink this...
        # Can this be integrated with rxn.network_meta somehow?
        self.conc = dict()
        for _ in self.species:
            _.get_smiles()
            self.conc[_.canon_smi] = 0.0


    ###############
    # Properties  #
    ###############
    @property
    def graph(self):
        return self._graph

    @property
    def inchi(self):
        return self._graph.inchi
    
    @property
    def canon_smi(self):
        return self._graph.canon_smi
    
    @property
    def map_smi(self):
        return self._graph.map_smi

    @property
    def hash(self):
        return self._graph.hash

    @property
    def bond_mats(self):
        return self._graph.bond_mats

    @property
    def species(self):
        return self._species

    # ==========================================
    # STAGE 2: Initial Guess Generation Helpers
    # ==========================================

    def generate_conformers(self, lot='gfn2', software='crest', source="initial_geom", n_conformers=1):
        """
        Step 2.A: Generate an ensemble of conformers for this state.
        Calls external tools (like CREST) and populates self.conformers 
        with the resulting structures.
        """
        if source not in self.conformers.keys():
            print(f"Inital geometry from {source} not found in State object! Can't generate conformers")
            return
        
        xyz_str = self.conformers[source].to_xyz_string()
        job_id = f'{self._graph.inchi}'
        if str(software).lower() == 'crest':
            calc = CrestCalculator(init_xyz=xyz_str, job_id=job_id, lot=lot)
            results = calc.execute()
        else:
            raise ValueError(f"Software {software} is not available to generate conformers with!")

        return

    def bias_conformers_to_target(self, target_state_bem, lot, software):
        """
        Step 2.B: Perform joint optimization by applying constraints (using a paired BEM).
        Biases the geometry of this state's conformers towards a target state 
        (e.g., biasing reactants towards products).
        """
        pass

    def evaluate_and_rank_conformers(self, strategy="ml-rich"):
        """
        Step 2.C: Rank conformers based on energy, RMSD, or ML predictions.
        Tags the best candidate(s) to be passed into the GSM/TS-guess stage.
        """
        pass

    # ==========================================
    # STAGE 3: Refinement Helpers
    # ==========================================

    def optimize_conformers(self, conformer_keys, lot, software):
        """
        Step 3.A: Run high-level geometry optimization on specific conformers.
        Reads the output and saves the refined conformers under new keys 
        (e.g., 'rpopt-{lot}-{software}').
        """
        pass

    # ==========================================
    # Utility / I-O Helpers
    # ==========================================

    def extract_lowest_energy_conformer(self, lot):
        """
        Returns the conformer object with the lowest Gibbs Free Energy or 
        electronic energy for a given level of theory.
        """
        pass



