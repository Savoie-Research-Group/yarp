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

.. py:class:: yarpecule(mol, mode='yarp', canon=True)

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

   mode : str, default=yarp
           When parsing SMILES this controls whether RDKIT is used or the in-house parser. By default
           the in-house parser is used. This variable is unused if the molecular info
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



   .. py:method:: draw_bmats(outfile='be_mats.pdf', show_inline=False)


   .. py:method:: export_geometry(filename, format='xyz')

      Export the geometry of the yarpecule to a file.
      This shouldn't ever change any of the attributes of the yarpecule.

      :param filename: The name of the file to export the geometry to.
      :type filename: str
      :param format: The format of the file to export the geometry to.
      :type format: str, default='xyz'



   .. py:method:: get_inchi()

      Generate the InChIKey for a given yarpecule using RDKit.
      Requires the yarpecule to already have SMILES
      Each separable group within the yarpecule will have an independently
      generated InChIKey, with dashes connecting them together.

      Modifies
      --------
      self._inchi : str



   .. py:method:: get_smiles()

      Generate a SMILES representation of the yarpecule.
      This shouldn't ever change any of the attributes of the yarpecule.
      Option to export SMILES with explicit atom mappings.
      Maybe also make it so we can optionally map the H atoms, but default to only reporting heavy atoms?


      Modifies
      --------
      self._canon_smi : str
      self._map_smi : str



   .. py:method:: join(yarpecules, canon=True)

      Method for creating a new yarpecule containing the union of the current yarpecule and all supplied yarpecules.

      :param yarpecules: A list of the yarpecules that the user wants to merge with this yarpecule.
                         Can also handle a single yarpecule being submitted.
      :type yarpecules: list of yarpecules
      :param canon: Controls whether or not the resulting yarpecule is subjected to the canonicalization ordering procedure.
      :type canon: bool, default=True

      :returns: **yarpecule** -- A new yarpecule containing the union of the chemical graphs contained in the supplied yarpecules.
      :rtype: yarpecule

      .. admonition:: Notes

         The resulting yarpecule will not retain any of the bond-electron matrix information of the parent yarpecules.



   .. py:method:: separate(canon=True)

      Method for separating discrete molecules into their own standalone yarpecule objects.
      Returns a copy of itself if there is only one discrete molecule.

      :param canon: Controls whether or not the resulting yarpecules are subjected to the canonicalization ordering procedure.
      :type canon: bool, default=True

      :returns: **mols** -- If there are no distinct molecules, returns a single yarpecule object as a list of length 1.
      :rtype: list of yarpecules



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



   .. py:attribute:: _bond_order_dict
      :value: None



   .. py:attribute:: _canon_smi
      :value: None



   .. py:attribute:: _elements
      :value: None



   .. py:attribute:: _geo
      :value: None



   .. py:attribute:: _inchi
      :value: None



   .. py:attribute:: _lewis_struct
      :value: None



   .. py:attribute:: _map_smi
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


   .. py:property:: atom_hashes


   .. py:property:: atom_neighbors


   .. py:property:: bo_dict


   .. py:property:: bond_mat_scores


   .. py:property:: bond_mats


   .. py:property:: canon_smi


   .. py:property:: elements


   .. py:property:: fc


   .. py:property:: geo


   .. py:property:: hash


   .. py:property:: inchi


   .. py:property:: lewis


   .. py:property:: map_smi


   .. py:property:: n_e_accept


   .. py:property:: n_e_donate


   .. py:property:: q


   .. py:property:: rings


