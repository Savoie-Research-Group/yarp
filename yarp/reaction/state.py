"""
Definition of the state object class.
"""
from copy import deepcopy

from yarp.reaction.conformer import conformer


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

    def opt_geom(self, target_geom, starting_geom="initial_geom", lot="uff"):
        # Select a starting point 3D geometry (from self.conformers) and optimize it using requested LOT
        # It will be saved using the label provided in target_geom (required input!!!)
        pass

    def gen_conformers(self, starting_geom="initial_geom", lot="uff", engine="crest", n_conf=1):
        # Select a starting point 3D geometry (from self.conformers)
        # Use selected software engine to generate the requested number of conformers
        # with the requested level of theory
        # Do I make a way to control the label it will be stored under self.conformers here?
        # Yeah... I think that probably makes sense to do!
        pass

    def joint_opt(self, bem, starting_geom=["initial_geom"], lot="uff"):
        # Generate conformers based on provided bem
        # Not sure how to downselect yet only the desired conformers... maybe a list of keys?
        # Provided BEM will be stored under self.paired_bem
        pass

    def select_conformer(self):
        # Apply ML prediction model to select the best conformer candidate for use in
        # downstream reaction path characterization processes
        pass

    def _update_conf_from_calc(self):
        # This will be used to read in data from completed calcs
        # Hopefully, I can make this general enough to use whenever we are
        # writing input files to / reading output files from disk
        pass



