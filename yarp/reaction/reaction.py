"""
Definition of the reaction object class.
"""

from yarp.reaction.state import state
from yarp.yarpecule.hashes import reaction_hash

class reaction:
    """
    Base class for describing a reaction in YARP

    Parameters:
    -----------
    reactant : yarpecule
        Molecular graph of all species involved in the reactant-side state of the reaction

    product : yarpecule
        Molecular graph of all species involved in the product-side state of the reaction

    Attributes:
    -----------
    reactant : state object
        The reactant-side state of the reaction.

    product : state object
        The product-side state of the reaction.

    ts_geom : dict
        Transition state geometry of the reaction.
        Keys correspond to '{lot}-{software}' or specific stage tags (e.g., 'tsguess').

    barrier : dict
        Energy of activation barrier (dG) of the reaction R --> P (kcal/mol).

    reverse_barrier : dict
        Energy of activation barrier (dG) of the reaction P --> R (kcal/mol).

    heat_of_rxn : dict
        Heat of reaction (dH) of the reaction R --> P (kcal/mol).

    id : str
        Human-readable name of reaction used to generate folders/files.

    hash : str/float
        Unique identifier for a reaction object.
        
    outcome_label : dict
        Intended/unintended logic labels for specific levels of theory.
        
    network_meta : dict
        Placeholder for storing metadata related to network generation.
    """

    def __init__(self, reactant, product):
        self.reactant = state(reactant)
        self.product = state(product)

        # Set up bond change information
        self.bond_changes = self.reactant.graph.adj_mat - self.product.graph.adj_mat
        self.reactant.paired_bem = self.product.graph.bond_mats[0]
        self.product.paired_bem = self.reactant.graph.bond_mats[0]

        # Geometries
        self.ts_geom = dict() 

        # Energetics
        self.barrier = dict()
        self.reverse_barrier = dict()
        self.heat_of_rxn = dict()

        # Identifiers & Metadata
        self.id = self.reactant.inchi + "_to_" + self.product.inchi
        self.hash = reaction_hash(self)
        
        self.outcome_label = dict() 
        self.network_meta = dict()

    # ==========================================
    # STAGE 1: First Pass Characterization
    # ==========================================
    
    def characterize_egat(self, egat_barrier, egat_rev_barrier, egat_heat):
        """
        Stage 1: Update reaction attributes based on EGAT predictions.
        """
        self.barrier['egat'] = egat_barrier
        self.reverse_barrier['egat'] = egat_rev_barrier
        self.heat_of_rxn['egat'] = egat_heat

    # ==========================================
    # STAGE 2: Initial Guess of Transition State
    # ==========================================

    def generate_rp_conformers(self, lot, software, n_conformers=1):
        """
        Step 2.A: Generate reactant-product conformers.
        Populates self.reactant.conformers and self.product.conformers 
        with 'rpopt-{lot}-{software}' and 'rpconf-{conf_id}-{lot}-{software}'.
        """
        pass

    def perform_joint_optimization(self, lot, software, strategy="both"):
        """
        Step 2.B: Perform joint optimization to bias reactant-product 
        conformers towards the transition state structure.
        Updates self.reactant.paired_bem and conformer dictionaries.
        """
        pass

    def select_conformer_pair(self, selection_strategy="ml-rich"):
        """
        Step 2.C: ML selection of R-P conformers to be used in downstream tasks.
        Assigns the chosen conformer key to 'selected-{strategy}'.
        """
        pass

    def generate_ts_guess_gsm(self, lot, software):
        """
        Step 2.D: Growing string method generation of initial transition state guess.
        Updates self.ts_geom['tsguess-{lot}-{software}'] with a conformer object.
        """
        pass

    # ==========================================
    # STAGE 3: Low-Level (High-Level) Refinement
    # ==========================================

    def optimize_rp_geometries(self, lot, software):
        """
        Step 3.A: Geometry optimization of reactant-product conformers via external batch job.
        Updates self.reactant.conformers['rpopt-{lot}-{software}'] and 
        self.product.conformers['rpopt-{lot}-{software}'].
        """
        pass

    def optimize_transition_state(self, lot, software):
        """
        Step 3.B: Transition state optimization of prior TS guess structure.
        Updates self.ts_geom['tsopt-{lot}-{software}'].
        """
        pass

    def validate_irc(self, lot, software):
        """
        Step 3.C: IRC validation of transition state structure.
        Updates self.reactant.conformers['irc-{lot}-{software}'] and 
        self.product.conformers['irc-{lot}-{software}'].
        """
        pass

    def compile_reaction_properties(self, lot, software, outcome_label):
        """
        Step 3.D: Compilation of reaction property information from the generated conformers.
        Updates self.outcome_label, self.barrier, self.reverse_barrier, and self.heat_of_rxn.
        """
        pass