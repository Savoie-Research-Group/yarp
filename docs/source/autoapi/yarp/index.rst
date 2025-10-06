yarp
====

.. py:module:: yarp


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/yarp/main_yarp/index
   /autoapi/yarp/reaction/index
   /autoapi/yarp/util/index
   /autoapi/yarp/yarpecule/index


Classes
-------

.. autoapisummary::

   yarp.lewis_struct
   yarp.yarpecule


Package Contents
----------------

.. py:class:: lewis_struct(adj_mat, elements, q)

   Base class for generating Lewis structures of molecules

   Parameters:
   -----------

   adj_mat : numpy.ndarray
           The adjacency matrix of the graphical representation of the molecular structure.
           Array is indexed to atoms in the `yarpecule`. If atom_i and atom_j are
           bonded, matrix elements M_ij and M_ji are equal to 1. Otherwise,
           all elements are 0.

   elements : list (str)
           A list of lower-case element labels indexed to the atomic ordering of the `yarpecule`.

   q : int
           The total charge on the `yarpecule`.

   Attributes:
   -----------

   rings: list, default=None
          List of lists holding the atom indices in each ring. If none, then the rings are calculated.

   bond_mats : list
           A list of arrays containing up to `mats_max` bond-electron matrices.
           Sorted by score in ascending order (lower score = better structure).

   scores : list
           A list of scores for each bond-electron matrix within `bond_mats`.

   e_acceptors : numpy.ndarray
           Lewis acidic atoms. Used for enumeration.
           Contains the number of electrons that each atom can accept.
           Currently only computed for the highest scoring bond-electron matrix.

   e_donors : numpy.ndarray
           Lewis basic atoms. Used for enumeration.
           Contains the number of electrons that each atom can donate.
           Currently only computed for the highest scoring bond-electron matrix.

   formal_charge : numpy.ndarray
           Formal charge of each atom.
           Currently only computed for the highest scoring bond-electron matrix.

   atom_neighbors : list of sets
           Each entry is a set of the atom's own index and the indices of its bonded neighbors,
           as determined by the adjacency matrix.

   bo_dict : dictionary of dictionaries
           Contains the set of observable bond-orders across all bond-electron matrices between atoms i and j at
           each element. The keys of the dictionary are the atom indices. For example, to query the bond-order of
           the bond between atoms 4 and 6 you can use bo_dict[4][6] or bo_dict[6][4]. By default, unbonded atoms
           have `None` as their bond-order.


   .. py:method:: __len__()


   .. py:method:: _find_rings(adj_mat)

      Make a call out to the return_rings function



   .. py:method:: _gen_bond_el_mat(adj_mat, elements, q=0, mats_max=10, mats_thresh=10.0, w_def=-1, w_exp=0.1, w_formal=0.1, w_aro=-24, w_rad=0.1, local_opt=True)

      Accesses self._rings, but shouldn't modify it at all...

      This will basically do everything in find_lewis()
      Should find_lewis() be chunked up more in order to have more refined
      unit testing? - ERM

      Algorithm for finding relevant Lewis Structures of a molecular graph given an overall charge.

      :param elements: Contains elemental information indexed to the supplied adjacency matrix.
                       Expects a list of lower-case elemental symbols.
      :type elements: list
      :param adj_mat: Contains the bonding information of the molecule of interest, indexed to the elements list.
      :type adj_mat: array of integers
      :param q: Sets the overall charge for the molecule.
      :type q: int, default=0
      :param rings: List of lists holding the atom indices in each ring. If none, then the rings are calculated.
      :type rings: list, default=None
      :param mats_max: The maximum number of bond electron matrices to return.
      :type mats_max: int, default=10
      :param mats_thresh: The value used to determine if a bond electron matrix is worth returning to the user.
                          Any matrix with a score within this value of the minimum structure will be returned as a
                          potentially relevant resonance structure (up to mats_max).
      :type mats_thresh: float, default=0.5
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
      :param local_opt: This controls whether non-local charge transfers are allowed (False). This can be expensive.
      :type local_opt: boolean, default=True
      :param Updates:
      :param -------:
      :param self._bond_mats: A list of arrays containing up to `mats_max` bond-electron matrices.
                              Sorted by score in ascending order (lower is better).
      :type self._bond_mats: list
      :param self._scores: A list of scores for each bond-electon matrix within bond_mats.
      :type self._scores: list



   .. py:method:: _get_properties(adj_mat, elements)

      Throw all these functions together?



   .. py:method:: _return_formals(bond_mat, elements)

      Returns returns the formal charge on each atom.

      :param bond_mat: A numpy array containing bond-orders in off-diagonal positions and unbound electrons along the diagonal.
                       This array is indexed to the elements list.
      :type bond_mat: array
      :param elements: Contains elemental information indexed to the supplied adjacency matrix.
                       Expects a list of lower-case elemental symbols.
      :type elements: list

      :returns: **formals** -- Contains the formal charge for each atom. This array is indexed to the bond-electron matrix.
      :rtype: array



   .. py:method:: _return_n_e_accept(bond_mat, elements)

      Returns the number of electrons each atom can accept without violating orbital constraints or breaking sigma bonds.
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



   .. py:method:: _return_n_e_donate(bond_mat, elements)

      Returns the number of electrons each atom can donate without breaking sigma bonds. This total basically comes to the
      sum of non-sigma-bonded electrons associated with each atom. Atoms participating in a double bonds are assumed to be able to
      donote at least two electrons since the double-bond can in principle be converted into a lone pair on the atom.

      :param bond_mat: A numpy array containing bond-orders in off-diagonal positions and unbound electrons along the diagonal.
                       This array is indexed to the elements list.
      :type bond_mat: array
      :param elements: `   Contains elemental information indexed to the supplied adjacency matrix.
                       Expects a list of lower-case elemental symbols.
                       Not used!!! - ERM
      :type elements: list

      :returns: **na** -- contains the number of electrons that each atom can donate.
      :rtype: array



   .. py:attribute:: _atom_neighbors
      :value: None



   .. py:attribute:: _bo_dict
      :value: None



   .. py:attribute:: _bond_mats
      :value: None



   .. py:attribute:: _e_acceptors
      :value: None



   .. py:attribute:: _e_donors
      :value: None



   .. py:attribute:: _elements


   .. py:attribute:: _formal_charge
      :value: None



   .. py:attribute:: _rings
      :value: None



   .. py:attribute:: _scores
      :value: None



   .. py:property:: bond_mats


.. py:class:: yarpecule(mol, canon=True, mode='rdkit')

   Base class for describing a molecule in YARP

   MISSING: update_masses() <-- ERM: I see this defined, but never used in classy YARP. Do we need it?

   Parameters:
   -----------

   mol : var
         The input that supplies the molecular graph information. This can either be a smiles string, a tuple holding
         (adj_mat, elements, charge),  or one or more filenames. (<-- ERM: CAN it handle multiple files?)
         For strings the extension is used to determine which parser to use (e.g., .xyz etc),
         otherwise the constructor will attempt to parse the input as a smiles string using rdkit.

   canon : bool, default=True
           Controls whether the atoms are indexed based on a canonicalization routine. Default is `True`.

   mode : str, default=rdkit
           When parsing SMILES this controls whether RDKIT is used or the in-house parser. By default rdkit is used
           but setting this to 'yarp' will use the in-house parser. This variable is unused if the molecular info
           is passed through another method besides SMILES.
           Thoughts on renaming this to smi_mode? - ERM

   Attributes:
   -----------
   geo : numpy.ndarray
           An (N_atom, 3) array containing the cartesian coordinates of each atom in the molecule.
           Units are in Angstroms.
           Array is indexed based on atomic ordering of the `yarpecule`.

   elements : list (str)
           A list of lower-case element labels indexed to the atomic ordering of the `yarpecule`.

   q : int
           The total charge on the `yarpecule`.

   masses : numpy.array
           A list of the atomic masses in the yarpecule. These masses are used in the determination of uniqueness,
           such that isotopomers will be considered unique.

   adj_mat : numpy.ndarray
           The adjacency matrix of the graphical representation of the molecular structure.
           Array is indexed to atoms in the `yarpecule`. If atom_i and atom_j are
           bonded, matrix elements M_ij and M_ji are equal to 1. Otherwise,
           all elements are 0.

   atom_hashes : array
           A list of hash values for each atom, based on graph connectivity and the masses of the atoms.

   mapping : ???
           Oh dang, what is this friend?????

   lewis_struct : list of `lewis_struct` object(s)
           Lewis structure(s) of the yarpecule. Multiple structures are generated for cases involving resonance.

   yarpecule_hash : float
           A unique identifier for the yarpecule based on atom hashes and bond-electron matrices
           generated from the Lewis structure(s) of the yarpecule.


   .. py:method:: __len__()


   .. py:method:: _gen_lewis_struct()

      Compute Lewis structure(s) for the yarpecule.
      Also update the yarpecule hash while we're at it?

      Updated Attributes:
      ------------------
      self.lewis_struct



   .. py:method:: _order_atoms(canon=False, mapping=None)

      Either canonically order the atoms or apply a user defined mapping.
      Not sure if the adjacency matrix is updated here, but I think it should be. - ERM

      Parameters:
      -----------
      canon : bool
              If True, the atoms are ordered based on a canonicalization routine.
              If False, the atoms are ordered based on the order they are provided.

      mapping : TBD! - ERM

      Updated Attributes:
      ------------------
      self._atom_hashes
              If canon is True, atom hashes are updated according to the `canon_order()` function.
              If canon is False, atom hashes are calculated directly from the `atom_hash()` function.

      self._mapping
              I don't know what is currently/should be done with this yet. - ERM



   .. py:method:: _read_structure(mol, mode)

      Read in an externally provided molecular structure and update
      core attributes of the yarpecule object.

      Parameters:
      -----------
      mol : str or tuple
              Input structure

      mode : str
              Mode to control SMILES parsing.

      Updated Attributes:
      ------------------
      self._adj_mat : numpy.ndarray
              Set to reflect input structure.

      self._geo : numpy.ndarray
              Set to reflect input structure.

      self._elements : list
              Set to reflect input structure.

      self._q : int
              Set to reflect input structure.

      self._masses : numpy.ndarray
              Atomic masses are computed from `elements`
              according to `el_mass` from `yarp.util.properties.py`



   .. py:method:: draw_lewis_struct()

      Draw the Lewis structure of the yarpecule.
      This shouldn't ever change any of the attributes of the yarpecule.



   .. py:method:: export_geometry(filename, format='xyz')

      Export the geometry of the yarpecule to a file.
      This shouldn't ever change any of the attributes of the yarpecule.

      :param filename: The name of the file to export the geometry to.
      :type filename: str
      :param format: The format of the file to export the geometry to.
      :type format: str, default='xyz'



   .. py:method:: export_smiles(mode='canonical')

      Export the SMILES representation of the yarpecule.
      This shouldn't ever change any of the attributes of the yarpecule.
      Option to export SMILES with explicit atom mappings.
      Maybe also make it so we can optionally map the H atoms, but default to only reporting heavy atoms?

      :param mode: The mode of the SMILES representation to export.
                   Options are 'canonical' or 'non-canonical'.
      :type mode: str, default='canonical'



   .. py:method:: join(other_yps, canon=True, mode='rdkit')

      Join two yarpecules together to form a new yarpecule.



   .. py:method:: update_atom_order(atom_index=None, canon=True)

      Update the atom order of the yarpecule.
      And then update all the other attributes that depend on the atom order.

      User can just ask to canonicalize the yarpecule,
      or they can provide a magic little list to tell us how to reorder the atoms.
      Not sure what exactly this should look like yet. - ERM



   .. py:attribute:: _adj_mat
      :value: None



   .. py:attribute:: _atom_hashes
      :value: None



   .. py:attribute:: _elements
      :value: None



   .. py:attribute:: _geo
      :value: None



   .. py:attribute:: _lewis_struct
      :value: None



   .. py:attribute:: _mapping
      :value: None



   .. py:attribute:: _masses
      :value: None



   .. py:attribute:: _q
      :value: 0



   .. py:attribute:: _yarpecule_hash
      :value: None



   .. py:property:: adj_mat


   .. py:property:: elements


   .. py:property:: geo


   .. py:property:: lewis


