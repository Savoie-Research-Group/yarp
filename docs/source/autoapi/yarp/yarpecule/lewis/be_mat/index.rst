yarp.yarpecule.lewis.be_mat
===========================

.. py:module:: yarp.yarpecule.lewis.be_mat

.. autoapi-nested-parse::

   Helper functions for bond-electron matrices.



Functions
---------

.. autoapisummary::

   yarp.yarpecule.lewis.be_mat.adjust_metals
   yarp.yarpecule.lewis.be_mat.all_zeros
   yarp.yarpecule.lewis.be_mat.bmat_score
   yarp.yarpecule.lewis.be_mat.bmat_unique
   yarp.yarpecule.lewis.be_mat.is_aromatic
   yarp.yarpecule.lewis.be_mat.return_bo_dict
   yarp.yarpecule.lewis.be_mat.return_connections
   yarp.yarpecule.lewis.be_mat.return_def
   yarp.yarpecule.lewis.be_mat.return_e
   yarp.yarpecule.lewis.be_mat.return_expanded
   yarp.yarpecule.lewis.be_mat.return_formals
   yarp.yarpecule.lewis.be_mat.return_n_e_accept
   yarp.yarpecule.lewis.be_mat.return_n_e_donate


Module Contents
---------------

.. py:function:: adjust_metals(bond_mats, adj_mat, elements)

   Accepts a list of bond mats and will adjust the bonding about the transition metals following the covalent bond
   classification (CBC) scheme. The adjacency matrix is used to determine where potential bonds exists. In short,
   if adjacency matrix indicates a potential bond between the metal and an electron-decicient atom with a radical,
   then a covalent bond  is formed using an electron from the metal, if the atom is electron deficient without a
   radical, then a bond is formed using two electrons from the metal, if the atom is not electron deficient (pi or
   lone pair containing) then no bond is formed (e.g., if the atom has a full octet then the bond is considered
   dative).

   :param bond_mats: Contains the bond_matrices that are being adjusted for the metal centers.
   :type bond_mats: list of arrays
   :param adj_mat: Contains the connectivity of the molecular graph.
   :type adj_mat: array
   :param elements: Contains the element labels for the atoms in the graph.
   :type elements: list

   :returns: **bond_mats** -- Contains the bond-electron matrices that have been updated to account for the nature of the ligands
             about the metal center.
   :rtype: list of arrays


.. py:function:: all_zeros(m)

   Helper function for `bmat_unique()` that checks is a numpy array is all zeroes
   (uses short-circuit logic to speed things up in contrast to np.any)


.. py:function:: bmat_score(bond_mat, elements, rings, cat_en, an_en, rad_env, e_def, e_exp, w_def=-1, w_exp=0.1, w_formal=0.1, w_aro=-24, w_rad=0.1, factor=0.0, verbose=False)

   Score function used to rank candidate Lewis Structures during and after the exploration. The `find_lewis()` algorithm uses a few
   different sets of weights at the start vs later parts of the algortihm by defining different versions via anonymous functions.

   bmat_score is the objective function that is minimized by the "best" lewis structures. The explanation of terms is as follows:
       1. Every electron deficiency (less than octet) is strongly penalized.
          Electron deficiencies on more electronegative atoms are penalized more strongly.
       2. Expanded octets are penalized at 0.1 per violation by default
       3. Formal charges are penalized based on their sign and the electronegativity of the atom they occur on
       4. (anti)aromaticity is incentivized (penalized) depending on the size of the ring.

   :param bond_mat: A numpy array containing bond-orders in off-diagonal positions and unbound electrons along the diagonal.
                    This array is indexed to the elements list.
   :type bond_mat: array
   :param elements: Contains elemental information indexed to the supplied adjacency matrix.
                    Expects a list of lower-case elemental symbols.
   :type elements: list
   :param rings: List of lists holding the atom indices in each ring. If none, then the rings are calculated.
   :type rings: list, default=None
   :param cat_en: Holds the cation electronegativity for each atom to determine the penalty for formal charges.
   :type cat_en: array
   :param an_en: Holds the anion electronegativity for each atom to determine the penalty for formal charges.
                 This is currently not used! - ERM
   :type an_en: array
   :param rad_env: Holds the radical environment term for each atom to determine the relative stability of hosting a radical.
   :type rad_env: array
   :param e_tet: Holds the number of electrons each atom needs to avoid a deficiency penalty (e.g., 8 for most organics,
                 2 for hydrogen).
   :type e_tet: array
   :param w_def: The weight of the electron deficiency term in the objective function for scoring bond-electron matrices.
   :type w_def: float, default=-1
   :param w_exp: The weight of the term for penalizing octet expansions in the objective function for scoring bond-electon matrices.
   :type w_exp: float, default=0.1
   :param w_formal: The weight of the formal charge term in the objective function for scoring bond-electon matrices.
   :type w_formal: float, default=0.1
   :param w_aro: The weight of the aromatic term in the objective function for scoring bond-electron matrices.
   :type w_aro: float, default=-24
   :param w_rad: The weight of the radical term in the objective function for scoring bond-electron matrices.
   :type w_rad: float, default=0.1
   :param factor: An optional value that can be added to the score. Useful for normalizing with respect to something (e.g., the ionization potential of the molecule).
   :type factor: float, default=0
   :param verbose: Controls whether the individual components of the score are printed to standard out during evaluation.
   :type verbose: bool, default=False

   :returns: **score** -- The score for the supplied bond-electron matrix.
   :rtype: float


.. py:function:: bmat_unique(new_bond_mat, old_bond_mats)

   Helper function for `gen_all_lstructs()` that checks whether an array already exists in a set of arrays.
   Deprecated because it was expensive. Now a hash is used in place of this in the comparison routine.


.. py:function:: is_aromatic(bond_mat, ring)

   Returns 1,0,-1 for aromatic, non-aromatic, and anti-aromatic respectively

   :param bond_mat: A numpy array containing bond-orders in off-diagonal positions and unbound electrons along the diagonal.
                    This array is indexed to the elements list.
   :type bond_mat: array
   :param ring: The atom indices of the ring being checking for aromaticity within bond_mat.
   :type ring: list

   :returns: **aromaticity** -- A value indicating aromaticity. 1,0,-1 for aromatic, non-aromatic, and anti-aromatic respectively.
   :rtype: int


.. py:function:: return_bo_dict(y, score_thresh=0.0)

   Returns a dictionary of dictionaries containing the set of bond orders observed across all bond-electron matrices
   available to the yarpecule. For example, if atoms 1 and 2 have a double bond in one resonance structure but a single
   bond in another, this dictionary will hold set({1,2}) in the bo_dict[1][2] and bo_dict[2][1] positions.

   :param y: Contains the bond_mats and bond_mat_scores needed for evaluation as attributesset.
   :type y: yarpecule
   :param score_thresh: Only bond_mats with a score below this threshold are used for determining bond orders. If none of the
                        bond_mats satisfy this threshold, then only the lowest scoring bond-electron matrix is used.
   :type score_thresh: float

   :returns: **bo_dict** -- Contains the set of observable bond-orders across all bond-electron matrices between atoms i and j at
             each element. The keys of the dictionary are the atom indices. For example, to query the bond-order of
             the bond between atoms 4 and 6 you can use bo_dict[4][6] or bo_dict[6][4]. By default, unbonded atoms
             have `None` as their bond-order.
   :rtype: dictionary of dictionaries


.. py:function:: return_connections(ind, bond_mat, inds=None, min_order=1)

   Returns indices of atoms bonded to the atom at `ind` according to the bond-electron matrix.

   :param ind: The index of the bond-electron matrix that the connections are being returned for.
   :type ind: int
   :param bond_mat: A numpy array containing bond-orders in off-diagonal positions and unbound electrons along the diagonal.
                    This array is indexed to the elements list.
   :type bond_mat: array
   :param inds: Optional list of indices of atoms that the user wants to restrict the return to. Useful for avoiding the return
                some trivial atoms that aren't relevant to resolving the Lewis structure.
   :type inds: list, default=None
   :param min_order: Optional argument that sets the threshold for determining a connection. If the user wishes to only find
                     doubly-bonded connections, then this would be set to 2 (default: 1).
   :type min_order: int, default=1

   :returns: **connections** -- Contains the indices of the bonded atoms subject to the `inds` and `min_order` arguments.
   :rtype: list


.. py:function:: return_def(bond_mat, elements, e_def)

   Returns returns the electron deficiencies of each atom (based on octet goal supplied via `e_tet`).

   :param bond_mat: A numpy array containing bond-orders in off-diagonal positions and unbound electrons along the diagonal.
                    This array is indexed to the elements list.
   :type bond_mat: array
   :param elements: Contains elemental information indexed to the supplied adjacency matrix.
                    Expects a list of lower-case elemental symbols.
   :type elements: list
   :param e_def: Holds the number of electrons each atom needs to avoid a deficiency penalty (e.g., 8 for most organics,
                 2 for hydrogen).
   :type e_def: array

   :returns: **deficiencies** -- Contains the electron deficiencies of each atom. This array is indexed to the bond-electron matrix.
   :rtype: array

   .. admonition:: Notes

      Atoms with expanded octets return 0 not a negative value.


.. py:function:: return_e(bond_mat)

   Returns the valence electrons possessed by each atom (half of each bond)

   :param bond_mat: A numpy array containing bond-orders in off-diagonal positions and unbound electrons along the diagonal.
                    This array is indexed to the elements list.
   :type bond_mat: array

   :returns: **valencies** -- Contains the valence electrons possessed by each atom. This array is indexed to the bond-electron matrix.
   :rtype: array


.. py:function:: return_expanded(bond_mat, elements, e_exp)

   Returns returns the number of surplus electrons beyond the target for each atom (based on octet goal
   supplied via `e_tet`).

   :param bond_mat: A numpy array containing bond-orders in off-diagonal positions and unbound electrons along the diagonal.
                    This array is indexed to the elements list.
   :type bond_mat: array
   :param elements: Contains elemental information indexed to the supplied adjacency matrix.
                    Expects a list of lower-case elemental symbols.
   :type elements: list
   :param e_exp: Holds the number of electrons each atom can have until incurring an expanded octect penalty (e.g., 8 for most organics,
                 2 for hydrogen).
   :type e_exp: array

   :returns: **surplus** -- Contains the excess electrons for each atom. This array is indexed to the bond-electron matrix.
   :rtype: array

   .. admonition:: Notes

      Atoms with electron deficiencies return 0 not a negative value.


.. py:function:: return_formals(bond_mat, elements)

   Returns returns the formal charge on each atom.

   :param bond_mat: A numpy array containing bond-orders in off-diagonal positions and unbound electrons along the diagonal.
                    This array is indexed to the elements list.
   :type bond_mat: array
   :param elements: Contains elemental information indexed to the supplied adjacency matrix.
                    Expects a list of lower-case elemental symbols.
   :type elements: list

   :returns: **formals** -- Contains the formal charge for each atom. This array is indexed to the bond-electron matrix.
   :rtype: array


.. py:function:: return_n_e_accept(bond_mat, elements)

   Returns returns the number of electrons each atom can accept without violating orbital constraints or breaking sigma bonds.
   Atoms that can expand their octets are treated as permitting two additional electrons beyond their orbital constraint (e.g.,
   sulfur can accept up to 10 electrons). Atoms participating in a double bonds are assumed to be able to accept at least two
   electrons since the double-bond can in principle be converted into a lone pair on the neighboring atom.

   :param bond_mat: A numpy array containing bond-orders in off-diagonal positions and unbound electrons along the diagonal.
                    This array is indexed to the elements list.
   :type bond_mat: array
   :param elements: Contains elemental information indexed to the supplied adjacency matrix.
                    Expects a list of lower-case elemental symbols.
   :type elements: list

   :returns: **na** -- contains the number of electrons that each atom can accept.
   :rtype: array


.. py:function:: return_n_e_donate(bond_mat, elements)

   Returns returns the number of electrons each atom can donate without breaking sigma bonds. This total basically comes to the
   sum of non-sigma-bonded electrons associated with each atom. Atoms participating in a double bonds are assumed to be able to
   donote at least two electrons since the double-bond can in principle be converted into a lone pair on the atom.

   :param bond_mat: A numpy array containing bond-orders in off-diagonal positions and unbound electrons along the diagonal.
                    This array is indexed to the elements list.
   :type bond_mat: array
   :param elements:
                    `   Contains elemental information indexed to the supplied adjacency matrix.
                       Expects a list of lower-case elemental symbols.
   :type elements: list

   :returns: **na** -- contains the number of electrons that each atom can accept.
   :rtype: array


