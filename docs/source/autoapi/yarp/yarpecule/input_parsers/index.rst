yarp.yarpecule.input_parsers
============================

.. py:module:: yarp.yarpecule.input_parsers

.. autoapi-nested-parse::

   Helper functions for parsing molecular information from a variety of input formats.
   Consider moving this to util if anything outside of yarpecule needs to access it.



Functions
---------

.. autoapisummary::

   yarp.yarpecule.input_parsers.mol_parse
   yarp.yarpecule.input_parsers.xyz_from_smiles
   yarp.yarpecule.input_parsers.xyz_parse
   yarp.yarpecule.input_parsers.xyz_q_parse


Module Contents
---------------

.. py:function:: mol_parse(mol)

   A simple wrapper for rdkit function to read a mol file.

   :param mol: The mol file that is being to convert into a geometry, adjacency matrix, list of elements, and charge.
   :type mol: str

   :returns: **(elements, geo, adj_mat, q)** -- `elements` is a list with the element labels, `geo` is an nx3 numpy array holding the rdkit
             generated geometry, `adj_mat` is an nxn array holding the adjacency matrix, `q` is an `int`
             holding the charge (based on the sum of formal charges).
   :rtype: tuple


.. py:function:: xyz_from_smiles(smiles, mode='rdkit')

   A simple wrapper to generate a 3D geometry, adj_mat, and elements from a SMILES string.
   Two modes for parsing SMILES strings are available: an in-house option [`smiles2adjmat()`]
   and an rdkit option. In either case, the generation of 3D geometries is handled by rdkit.

   :param smiles: The SMILES string that is being converted into a geometry, adjacency matrix, list of elements, and charge.
   :type smiles: str
   :param mode: This variable controls whether to use the yarp SMILES parser or the rdkit parser.
                The default is to use rdkit.
                The in-house `smiles2adjmat()` parser is used if 'yarp' is supplied to the argument.
   :type mode: str

   :returns: **(elements, geo, adj_mat, q)** -- `elements` is a list with the element labels,
             `geo` is an nx3 numpy array holding the rdkit generated geometry,
             `adj_mat` is an nxn array holding the adjacency matrix,
             `q` is an `int` holding the molecular charge (based on the sum of formal charges).
   :rtype: tuple


.. py:function:: xyz_parse(xyz, read_types=False, multiple=False)

   Simple wrapper function for grabbing the coordinates and elements from an xyz file.

   :param xyz: This is the xyz file being parsed.
   :type xyz: filename
   :param read_types: If this is set to `True` then the function will try and grab optional data from a fourth column of
                      the file (i.e., a column after the x y and z information).
   :type read_types: bool, default=False
   :param multiple: Allows for multiple coordinates/elements to be read from the same XYZ file.
                    If set to False, only the final set of coordinates and elements will be returned.
   :type multiple: bool, default=False

   :returns: * **elements** (*list*) -- A list with the element labels indexed to the geometry.
             * **geo** (*array*) -- An nx3 numpy array holding the cartesian coordinates for the user supplied geometry.
             * **atom_types** (*list (optional)*) -- If the `read_types=True` option is supplied then an optional third list is returned.


.. py:function:: xyz_q_parse(xyz)

   This function grabs charge information from the comment line of an xyz file. The charge information is
   interpreted as the first field following the `q` keyword. If no charge information is specified the function
   returns neutral as a the default behavior.

   :param xyz: This is the xyz file that is read by the function.
   :type xyz: filename

   :returns: **q** -- The charge information.
   :rtype: int


