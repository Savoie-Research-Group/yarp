yarp.yarpecule.atom_mapping
===========================

.. py:module:: yarp.yarpecule.atom_mapping

.. autoapi-nested-parse::

   Helper functions related to atom mappings of molecules



Functions
---------

.. autoapisummary::

   yarp.yarpecule.atom_mapping.canon_order
   yarp.yarpecule.atom_mapping.gen_subgraphs
   yarp.yarpecule.atom_mapping.graph_seps


Module Contents
---------------

.. py:function:: canon_order(elements, adj_mat, masses=None, hash_list=None, things_to_order=[], change_mol_seq=True, return_index=True)

   Canonicalizes the ordering of atoms in a graph based on a hash function.
   Atoms that hash to equivalent values retain their relative order from the inputted graph.

   :param elements: Contains elemental information indexed to the supplied adjacency matrix.
                    Expects a list of lower-case elemental symbols.
   :type elements: list
   :param adj_mat: nxn array containing indicated bonds between positions i and j by a 1 in that position.
   :type adj_mat: array
   :param masses: The atomic masses are used for calculating the atom hashes and in turn sorting the atoms. By default the average
                  atomic masses are used. The user can override this behavior by supplying their own masses via this parameter.
   :type masses: array, default=None
   :param things_to_order: objects that are supplied to this list should have the same length as `elements`. After
                           canonicalizing the ordering of atoms in the graph, objects supplied to this argument will also
                           be ordered in the same way. For example, if the user wants to order the geometry in the way
                           as the elements, then it can be supplied to this argument.
   :type things_to_order: list of objects, default=[]
   :param change_mol_seq: When set to True the function will rearrange separate molecules in the supplied adjacency matrix
                          (if more than one are present) based on the largest hash value in the molecule. Default behavior
                          is to reorder the molecules in this fashion (True).
   :type change_mol_seq: bool, default=True
   :param return_index: When set to True, the function will return a list corresponding to the ordered atoms with their
                        indices in the original graph. Objects ordered by the old ordering can be sorted by a simple
                        `a = a[idx]` call, where `idx` is the list of indices returned when this option is set to `True`.
   :type return_index: bool, default=True

   :returns: * **elements** (*list*) -- Contains the ordered elemental labels after applying the canonicalization procedure.
             * **adj_mat** (*array*) -- Contains the ordered adjacency matrix after applying the canonicalization procedure.
             * **hash_list** (*list*) -- A list of hash values for each atom, ordered according to the canonicalization procedure.
             * **idx** (*list*) -- A list containing the indices of the ordered atoms in the original adjacency matrix. For example, [1,0,2]
               would mean that the first atom in the canonicalized ordering was the second atom in the original ordering,
               and so forth. This list is useful for sorting objects that were indexed to the old ordering (e.g., using a
               `obj = obj[idx]` call).
             * **ordered_things** (*tuple*) -- Contains the canonically ordered versions of the objects supplied to the optional
               `things_to_order` parameter. If n items were supplied, then that number will be returned.

   .. admonition:: Notes

      The minimal return of this function is the tuple of ordered `elements`, `adj_mat`, and `hash_values`. The `idx` object
      is only returned when `return_index=True` and `ordered_things` are only returned when objects are supplied to the
      optional `things_to_order` argument.


.. py:function:: gen_subgraphs(adj_mat, gs=None)

   A function for calculating the connected subgraphs of an adjacency matrix. The algorithm uses the graphical
   separations between atoms to determine the connections and has a cost of n matrix multiplications, where n
   is the size of the adjacency matrix.

   :param adj_mat: The adjacency matrix that the subgraphs are being calculated for
   :type adj_mat: array
   :param gs: An array of the same dimensions as adj_mat, that holds the separations between each pair of nodes in the
              off-diagonal positions. By default these are calculated, but if the user already has them on hand then they
              can be passed directly.
   :type gs: array, default=None

   :returns: **subgraphs** -- A list of lists, where each subgraph holds the indices of the nodes in the subgraph. The ordering of
             the lists and ordering of atoms in each subgraph should not be relied on.
   :rtype: list of lists


.. py:function:: graph_seps(adj_mat_0)

   Returns a matrix of graphical separations for all nodes in a graph defined by the inputted adjacency matrix

   :param adj_mat_0: This array is indexed to the atoms in the `yarpecule` and has a one at row i and column j if there is
                     a bond (of any kind) between the i-th and j-th atoms.
   :type adj_mat_0: array

   :returns: **seps** -- What is the final shape of this matrix? (ERM)
   :rtype: NDArray


