yarp.yarpecule.graph.smiles
===========================

.. py:module:: yarp.yarpecule.graph.smiles

.. autoapi-nested-parse::

   Helper functions for converting SMILES strings into molecular graphs



Exceptions
----------

.. autoapisummary::

   yarp.yarpecule.graph.smiles.OctetError


Functions
---------

.. autoapisummary::

   yarp.yarpecule.graph.smiles.add_hydrogens
   yarp.yarpecule.graph.smiles.reorder_by_mappings
   yarp.yarpecule.graph.smiles.smiles2adjmat


Module Contents
---------------

.. py:exception:: OctetError(atom_indices, message='Atom indices {} has an octet violation.')

   Bases: :py:obj:`ValueError`


   Exception raised when the number of electrons exceeds the allowed limit.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: message
      :value: ''



.. py:function:: add_hydrogens(adjmat, atom_info)

   This is a helper function for the smiles2adjmat() function that adds hydrogens to atoms based on either
   the explicit number of hydrogens designation or the using an inference algorithm based on the formal charge
   and number of bonds. If an explicit number of hydrogens is supplied it will overrule the hydrogen inference
   routine based on formal charge. Formal charge inference is based on the neutral full octet protonation state
   with protons/hydrides removed or added to meet the formal charge requirements.

   :param adjmat: This is numpy array holding the graph defined by the smiles string.
   :type adjmat: array
   :param atom_info: This list is indexed to the adjacency matrix and contains a tuple for each atom. Each tuple has
                     the element token, formal charge, explicit hydrogens, isotope, atom_mapping, and should_infer_hydrogens
                     as its respective elements.
   :type atom_info: list of tuples

   :returns: * **adjmat** (*array*) -- This is numpy array holding the graph defined by the smiles string. This array is expanded relative
               to the inputted array to accomodate the additional hydrogens.
             * **atom_info** (*list of tuples*) -- This list is indexed to the adjacency matrix and contains a tuple for each atom. Each tuple has
               the element token, formal charge, explicit hydrogens, isotope, atom_mapping, and should_infer_hydrogens
               as its respective elements. This list is expanded relative to the inputted list to reflect the additional hydrogens.


.. py:function:: reorder_by_mappings(adjmat, atom_info)

   Reorder atoms in the graph based on their atom mappings if present.
   If no mappings are present, returns the original ordering.

   :param adjmat: The adjacency matrix of the molecular graph
   :type adjmat: array
   :param atom_info: List of tuples containing (element, charge, hydrogens, isotope, mapping)
   :type atom_info: list

   :returns: (reordered_adjmat, reordered_atom_info) if mappings present
             (original_adjmat, original_atom_info) if no mappings
   :rtype: tuple


.. py:function:: smiles2adjmat(smiles, verbose=False)

   In-house Savoie group SMILES parser. Written in python and transparent to debug. The main motivation
   was to consistently handle protonation of radicals and atoms with formal charges. The usual SMILES
   syntax rules apply, except that square brace annotations are handled specially. Square braces are
   reserved to annotate the isotope, formal charge, or number of hydrogens that should be added to an
   atom. The isotope number must preceed the element label. The charge and number of hydrogens must be
   after the element label. The formal charge can be specified as +d, ++++, -d, ---, where d is an integer.
   The number of hydrogens to be added can be specified as Hd or HHH, where d is an integer.

   :param smiles: The smiles string that the user wants to parse.
   :type smiles: str

   :returns: * **adjmat** (*array*) -- This is numpy array holding the graph defined by the smiles string.
             * **atom_info** (*list of tuples*) -- This list is indexed to the adjacency matrix and contains a tuple for each atom. Each tuple has
               the element token, formal charge, explicit hydrogens, isotope, atom_mapping, and should_infer_hydrogens
               as its respective elements.


