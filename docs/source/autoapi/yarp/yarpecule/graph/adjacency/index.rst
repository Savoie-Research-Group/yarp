yarp.yarpecule.graph.adjacency
==============================

.. py:module:: yarp.yarpecule.graph.adjacency

.. autoapi-nested-parse::

   Helper functions related to adjacency matrices



Functions
---------

.. autoapisummary::

   yarp.yarpecule.graph.adjacency.adjmat_to_adjlist
   yarp.yarpecule.graph.adjacency.graph_seps
   yarp.yarpecule.graph.adjacency.table_generator


Module Contents
---------------

.. py:function:: adjmat_to_adjlist(adj_mat)

   Convenience function for converting between adjacency matrix
   and adjacency list (actually a list of sets for convenience)


.. py:function:: graph_seps(adj_mat_0)

   Returns a matrix of graphical separations for all nodes in a graph defined by the inputted adjacency matrix

   :param adj_mat_0: This array is indexed to the atoms in the `yarpecule` and has a one at row i and column j if there is
                     a bond (of any kind) between the i-th and j-th atoms.
   :type adj_mat_0: NDArray (N x N)

   :returns: **seps** -- Default is to assign zeros along the diagonal, and -1 for all off-diagonal elements.
             If a connection to a neighboring atom is found, off-diagonal elements are assigned an
             integer value of 1 or greater, depending on how many connections are found.
   :rtype: NDArray (N x N)


.. py:function:: table_generator(elements, geometry, scale_factor=1.2, filename=None)

   Algorithm for finding the adjacency matrix of a geometry based on atomic separations.

   :param elements: Contains elemental information indexed to the supplied adjacency matrix.
                    Expects a list of lower-case elemental symbols.
   :type elements: list
   :param geo: nx3 array of atomic coordinates (cartesian) in angstroms.
   :type geo: array
   :param scale_factor: Used to scale the atomic radii to determine if a bond exists.
   :type scale_factor: float, default=1.2

   :returns: **adj_mat** -- An nxn array indexed to elements containing ones bonds occur.
   :rtype: array


