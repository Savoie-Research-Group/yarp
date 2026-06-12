yarp.reaction.conformer
=======================

.. py:module:: yarp.reaction.conformer

.. autoapi-nested-parse::

   Definition of the conformer class and related helper functions.



Classes
-------

.. autoapisummary::

   yarp.reaction.conformer.conformer


Functions
---------

.. autoapisummary::

   yarp.reaction.conformer.select_conformer_pair


Module Contents
---------------

.. py:class:: conformer

   Attributes:
   -----------
   geo : numpy array
       3D cartesian coordinates of the conformer.

   elements : list of str
       List of elements in the conformer.

   properties : dict
       Computed properties of the conformer, such as Gibbs free energy, enthalpy, etc.
       I think it makes sense to group them together in a dictionary, rather than having them individually added
       as attributes. This way, we can easily add new properties without modifying the class.

   lot : str
       Some label that indicates the level of theory used to generate the conformer.
       Not sure if this should be put here, or in the higher level classes that contain conformers.


   .. py:attribute:: elements
      :value: []



   .. py:attribute:: geo
      :value: None



   .. py:attribute:: lot
      :value: ''



   .. py:attribute:: properties


.. py:function:: select_conformer_pair(r, p, inp)

