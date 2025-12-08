yarp.reaction.generate_rxns
===========================

.. py:module:: yarp.reaction.generate_rxns


Functions
---------

.. autoapisummary::

   yarp.reaction.generate_rxns.enumerate_products
   yarp.reaction.generate_rxns.generate_rxns
   yarp.reaction.generate_rxns.quick_geom_opt


Module Contents
---------------

.. py:function:: enumerate_products(r_yp, n_break, n_form, react=[], mode='sequential', lewis_filter=True, cutoff=0.0, ring_filter=False)

   r_yp : yarpecule object
       The reactant from which all products are enumerated

   n_break : int
       Number of bonds to break

   n_form : int
       Number of bonds to form

   react : set (default = None)
       When supplied this is used to restrict bond formations only to those atoms in this set.
       If supplied, then `react` must have a searchable list or set
       (i.e., the function uses an `in` call, so sets are better) per `yarpecule`.
       An empty list is interpreted as all atoms being available to react.

   mode : string
       Toggle between the two available product enumeration modes:
       Concerted and sequential (default) enumeration.

   lewis_filter : bool (default = True)
       Filter out enumerated products based on bond-electron matrix scores and formal charges.

   cutoff : float (default = 0.0)
       Threshold used in sequential enumeration to discard unphysical Lewis structures
       with bond-electron matrix scores above this value.

   ring_filter : bool (default = False)
       Filter out 3 and 4 member rings from enumerated products.


.. py:function:: generate_rxns(inp)

   init : dict
       literally the stuff contained under "initialize" in the input YAML file

   Returns a dictionary of reaction objects

   Should this be a class?


.. py:function:: quick_geom_opt(molecule, lot='uff')

   Perform low-level level geometry optimization on yarpecule using openbabel.

   ERM: Can we just change the forcefield from UFF if we want?

   Parameters:
   ----------
   molecule : yarpecule object
       molecule to be optimized

   lot : string
       Level of theory used for quick optimization

   :returns: **molecule** -- optimized molecule
   :rtype: yarpecule object


