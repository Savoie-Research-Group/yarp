yarp.yarpecule.yarpecule
========================

.. py:module:: yarp.yarpecule.yarpecule

.. autoapi-nested-parse::

   Definition of yarpecule object class



Classes
-------

.. autoapisummary::

   yarp.yarpecule.yarpecule.yarpecule


Module Contents
---------------

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


