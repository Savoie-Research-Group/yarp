yarp.yarpecule.hashes
=====================

.. py:module:: yarp.yarpecule.hashes

.. autoapi-nested-parse::

   Helper functions related to hash objects associated with determining unique atoms and yarpecules



Functions
---------

.. autoapisummary::

   yarp.yarpecule.hashes.atom_hash
   yarp.yarpecule.hashes.bmat_hash
   yarp.yarpecule.hashes.rec_sum
   yarp.yarpecule.hashes.yarpecule_hash


Module Contents
---------------

.. py:function:: atom_hash(ind, adj_mat, masses, alpha=100.0, beta=0.1, gens=10)

   Creates a unique hash value for each atom based on its location in the molecular graph (out to a depth of `gens`).
   The algorithm for calculating this performs a walk of the subgraph about `ind` without back-tracking. At each step,
   of the walk, `s`, the masses of the visited atoms are summed and weighted by `beta`*0.1**(`s`), where `beta` is a
   user-supplied parameter. The recursive walk is performed by the `rec_sum()` helper function.

   :param ind: The index of the adjacency matrix of the atom that the hash is being calculated for.
   :type ind: int
   :param adj_mat: nxn array containing indicated bonds between positions i and j by a 1 in that position.
   :type adj_mat: array
   :param masses: an n-length array-like that holds the masses of each atom indexed to adj_mat.
   :type masses: array
   :param alpha: This is used to scale the contribution to the hash of the number of bonded neighbors to the atom at `ind`.
   :type alpha: float, default=100.0
   :param beta: This is the base value for weighting the sum of masses of the bonded neighbors at each level of the graph.
   :type beta: float, default=0.1
   :param gens: This is the depth of the recursion for determining graphical uniqueness. It the subgraphs of two atoms out
                to `gens` bonds away are identical, then the atoms will hash to the same value. The default value (10) is
                meant to be a conservative value.
   :type gens: int, default=10

   :returns: **hash** -- The hash value associated with the atom.
   :rtype: float


.. py:function:: bmat_hash(bond_mat)

   Creates a unique hash value for each bond-electron matrix that is used to speed uniqueness checks.

   :param bond_mat: The bond electron matrix that the hash is calculated for.
   :type bond_mat: array

   :returns: **hash_value**
   :rtype: float

   .. admonition:: Notes

      The hash is calculated as bond_mat * an ascending array (1,2,... counting up through all elements and rows) summed over rows,
      then those values are multiplied by 10**(-i/100) where i is the column, and summed.


.. py:function:: rec_sum(ind, adj_mat, masses, beta, gens, avoid_list=[])

   This is a helper function for `atom_hash()` that performs a non-backtracking walk of the adjacency matrix and sums
   the masses of atoms at each step with a weighting factor based on the number of steps that have been taken and the
   user-supplied base of `beta`.

   :param ind: The index of the adjacency matrix of the atom that the hash is being calculated for.
   :type ind: int
   :param adj_mat: nxn array containing indicated bonds between positions i and j by a 1 in that position.
   :type adj_mat: array
   :param masses: an n-length array-like that holds the masses of each atom indexed to adj_mat.
   :type masses: array
   :param beta: This is the base value for weighting the sum of masses of the bonded neighbors at each level of the graph.
   :type beta: float, default=0.1
   :param gens: This is the depth of the recursion. This value counts down during the recursion.
   :type gens: int, default=10
   :param avoid_list: This list holds the indices of atoms that have already been visited during the walk. This list is
                      checked at each step of the recursion to avoid backtracking and retracing cycles.
   :type avoid_list: list

   :returns: **sum** -- The recursive sum of depth-weighted masses.
   :rtype: float


.. py:function:: yarpecule_hash(y)

   Creates a unique hash value for the yarpecule object based on the sum of all bond-electron matrices and the atom hashes.
   Since the atom hashes are sensistive to the masses used for the atoms, the hash of isotopomers will be unique.

   :param y: This is the yarpecule instance that the hash is being calculated for.
   :type y: yarpecule

   :returns: **hash_value**
   :rtype: float

   .. admonition:: Notes

      Any method affecting the `bond_mats` or `masses` attributes of the yarpecule instance should also recalculate this hash.
      The hash is calculated as a 128-bit number. For use in sets and comparisons this number is hashed by python's hash function.


