yarp.reaction.node
==================

.. py:module:: yarp.reaction.node

.. autoapi-nested-parse::

   Definition of the node object class.



Classes
-------

.. autoapisummary::

   yarp.reaction.node.node


Module Contents
---------------

.. py:class:: node(yp)

   Attributes:
   -----------

   graph : yarpecule object
       Yarpecule object that contains the molecular graph.
       This is provided upon initialization, and is typically not modified.

   conformers : list of conformer objects


   .. py:method:: gen_conformers(input)


   .. py:method:: refine_node(input)


   .. py:property:: canon_smi


   .. py:attribute:: conformers
      :value: []



   .. py:attribute:: graph


   .. py:property:: inchi


   .. py:property:: map_smi


