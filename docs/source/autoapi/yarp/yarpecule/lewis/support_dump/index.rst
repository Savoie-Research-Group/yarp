yarp.yarpecule.lewis.support_dump
=================================

.. py:module:: yarp.yarpecule.lewis.support_dump

.. autoapi-nested-parse::

   Functions that are needed for finding Lewis structures.
   But I don't know where they should live.
   So they are here for now! -ERM



Exceptions
----------

.. autoapisummary::

   yarp.yarpecule.lewis.support_dump.LewisStructureError


Functions
---------

.. autoapisummary::

   yarp.yarpecule.lewis.support_dump.delta_aromatic
   yarp.yarpecule.lewis.support_dump.gen_all_lstructs
   yarp.yarpecule.lewis.support_dump.gen_init
   yarp.yarpecule.lewis.support_dump.valid_bonds
   yarp.yarpecule.lewis.support_dump.valid_moves


Module Contents
---------------

.. py:exception:: LewisStructureError(message='An error occured in a find_lewis() call.')

   Bases: :py:obj:`Exception`


   Common base class for all non-exit exceptions.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: message
      :value: 'An error occured in a find_lewis() call.'



.. py:function:: delta_aromatic(bond_mat, rings, move)

   Helper function for valid moves that determines if a proposed move will results in a change in aromaticity

   :param bond_mat: The bond electron matrix that the bond/electron rearrangments are calculated for.
   :type bond_mat: array
   :param rings: List of lists holding the atom indices in each ring. Used to determine (anti) aromaticity.
   :type rings: list
   :param move: (int, i, j) where int is the value to be added to the ij position of the bond-electron matrix.
   :type move: tuple

   :returns: **change** -- True indicates that the move will result in an increase in aromaticity, False that it will not.
   :rtype: boolean


.. py:function:: gen_all_lstructs(obj_fun, bond_mats, scores, hashes, elements, reactive, rings, ring_atoms, bridgeheads, seps, min_score, ind=0, counter=100, N_score=1000, N_max=10000, min_opt=False, min_win=False)

   A generator for find_lewis() that recursively applies a set of valid bond-electron moves to find all relevant resonance structures.

   :param obj_fun: A function that accepts a bond electron matrix and returns a score.
                   This assumes that the elements and objective function weights have already been supplied
                   (e.g., by defining an anonymous function to pass to this function).
   :type obj_fun: function
   :param bond_mats: Contains the bond-electron matrices that have already been discovered and scored.
                     Used by the algorithm to avoid back-tracking.
   :type bond_mats: list of bond_mat arrays
   :param scores: Contains the scores for all bond-electron matrices that have been enumerated.
   :type scores: list of floats
   :param hashes: Contains a set of bond-electron matrix hash values used to accelerate the check for duplication.
   :type hashes: set of floats
   :param elements: Contains elemental information indexed to the supplied adjacency matrix.
   :type elements: list of lower-case elemental symbols
   :param reactive: Contains the indices of the atoms in the bond-electron matrix that are candidates for the rearrangement moves.
   :type reactive: list of integers
   :param rings: List of lists holding the atom indices in each ring.
   :type rings: list
   :param ring_atoms: Contains the indices of of atoms in rings.
                      These are used to determine the possibility of forming double bonds,
                      if multiple double bonds and alkynes are allowed on an atom when enumerating resonance structures.
   :type ring_atoms: list of integers
   :param bridgeheads: Contains the indices of the atoms serving as ring bridgeheads.
                       These are used to enforce Bredt's rules during the resonance structure search.
   :type bridgeheads: list of integers
   :param seps: Contains the number of bonds separating each pair of atoms at the ij-th position.
   :type seps: array
   :param min_score: Contains the current best score out of all enumerated Lewis structures.
   :type min_score: float
   :param ind: Contains the index of the bond_mat within bond_mats that the function is supposed to act on.
   :type ind: int, default=0
   :param counter: Keeps track of the number of iterations that have passed without finding a better Lewis structure.
                   Used to determine the `N_score` break condition.
   :type counter: int, default=0
   :param N_score: The function will break if this number of steps pass without finding an improved Lewis structure.
   :type N_score: int, default=100
   :param N_max: The function will break if this number of bond electron matrices have been generated.
   :type N_max: int, default=10000
   :param min_opt: If set to `True` then the search is run in a greedy mode
                   where Lewis structures are only accepted if they are as good or better than the structure discovered up to that point.
                   This option is used as part of the base algorithm
                   to initially find a reasonable structure before a more fine-grained comprehensive search.
   :type min_opt: boolean, default=False
   :param min_win: When set, a Lewis structure is only accepted if its score is within this value of the best structure found up to that point.
                   This allows the algorithm to explore intermediate structures that may be less ideal
                   but that eventually lead to an overall relaxation of the structure.
   :type min_win: float, default=False

   :Yields: **iterator** (*tuple*) -- This function yields a set of initial guesses for the find_lewis algorithm via iteration.
            Each iteration returns a tuple, (score, bond_mat, reactive_indices),
            containing the score of the initial guess, the bond-electron matrix, and the list of reactive indices.


.. py:function:: gen_init(obj_fun, adj_mat, elements, rings, q)

   A helper-generator for initial guesses for the final_lewis algorithm.

   :param obj_fun: A function that accepts a bond electron matrix and returns a score.
                   This assumes that the elements and objective function weights have already been supplied
                   (e.g., by defining an anonymous function to pass to this function).
   :type obj_fun: function
   :param adj_mat: Contains the bonding information of the molecule of interest, indexed to the elements list.
   :type adj_mat: array of integers
   :param elements: Contains elemental information indexed to the supplied adjacency matrix.
   :type elements: list of lower-case elemental symbols
   :param rings: List of lists holding the atom indices in each ring.
   :type rings: list,
   :param q: Sets the overall charge for the molecule.
   :type q: int

   :Yields: **iterator** (*tuple*) -- This function yields all a set of initial guesses for the find_lewis algorithm via iteration.
            Each iteration returns a tuple (score, bmat, inds)
            containing the score of the initial guess, the bond-electron matrix, and the list of reactive indices.


.. py:function:: valid_bonds(ind, bond_mat, elements, reactive, ring_atoms)

   This is a simple version of `valid_moves()` that only returns valid bond-formation moves with some
   quality checks (e.g., octet violations and allenes in rings). This function is used to generate the initial guesses for the Lewis Structure.

   :param ind:
   :type ind: int
   :param bond_mat: The bond electron matrix that the bond/electron rearrangments are calculated for.
   :type bond_mat: array
   :param elements: list of elements indexed to the bond_mat.
   :type elements: list
   :param reactive: List of integers corresponding to the indices of bond_mat where atoms capable of undergoing bond-elctron rearrangments reside.
   :type reactive: list
   :param ring_atoms: List of integers corresponding to the indices of bond_mat where the atoms reside in a ring. Used to avoid forming allenes and alkynes within rings.
   :type ring_atoms: list

   :returns: **move** -- Each tuple in the list is composed of (int, i, j) where int is the value to be added to the ij position of the bond-electron matrix.
   :rtype: list of tuples,


.. py:function:: valid_moves(bond_mat, elements, reactive, rings, ring_atoms, bridgeheads, seps)

   Generator that returns all valid moves that can be performed on a given bond-electron matrix.
   Used as a helper function for gen_all_lstructs to loop over potential lewis structures.

   :param bond_mat: The bond electron matrix that the bond/electron rearrangments are calculated for.
   :type bond_mat: array
   :param elements: list of elements indexed to the bond_mat
   :type elements: list
   :param reactive: List of integers corresponding to the indices of bond_mat where atoms capable of undergoing bond-elctron rearrangments reside.
   :type reactive: list
   :param rings: List of lists holding the atom indices in each ring. Used to determine (anti) aromaticity.
   :type rings: list,
   :param ring_atoms: List of integers corresponding to the indices of bond_mat where the atoms reside in a ring. Used to avoid forming allenes and alkynes within rings.
   :type ring_atoms: list
   :param bridgeheads: List of integers corresponding to the indices of bond_mat where the atoms reside at bridgeheads. Used for respecting Bredt's rule.
   :type bridgeheads: list
   :param seps: Array holding the graphical separations of each pair of atoms. Used to determine valid charge transfers based on proximity.
   :type seps: array

   :Yields: **move** (*list of tuples,*) -- Each tuple in the list is composed of (int, i, j) where int is the value to be added to the ij position of the bond-electron matrix.

   .. admonition:: Notes

      Attempted moves on each reactive atom (i) include (in this order):
      (1) shifting a pi-bond between a neighbor (j) and next-nearest neighbor (k) of a 2-electron deficient atom (i) to one between i and j.
      (2) shifting a pi-bond between a neighbor (j) and next-nearest neighbor (k) of a radical 1-electron deficient atom (i) to one between i an j.
      (3) shifting a pi-bond between a neighbor (j) and next-nearest neighbor (k) of a lone-pair containing atom (i) to a lone-pair on k and a new pi-bond between i and j.
      (4) forming a pi-bond between a radical containing atom (i) and a neighbor (j) with unbound electron(s). This might be accompanied by a charge transfer from j to another atom if required.
      (5) forming a pi-bond between an atom with a long pair (i) and a neighbor (j) capable of accepting a pi-bond.
      (6) turn a pi-bond between i and its neighbor j into a lone pair on i if favored by electronegativity or aromaticity.
      (7) transfer an electron to i from its neighbor j, if i is electron deficient and has a greater electronegativity.
      (8) transfer a charge from i to another atom if i has an expanded octet and unbound electrons.
      (9) shuffle aromatic and anti-aromatic bonds (i.e., change bond alteration along the cycle).
      (10) forming a pi-bond between two radicals
      All of these moves are contingent on the ability of atoms to expand octet, whether they are electron deficient, and whether the move would lead to unphysical ring-strain.


