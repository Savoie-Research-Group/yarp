Welcome to yarp documentation!
======================================

`yarp` is a python package with classes and functions that implement the yarp reaction exploration
methodology. The core classes are built for workflows that involve enumerating potential reactions,
performing pattern matching on molecules, conducting transition state searches, managing calculations,
analyzing the outcomes of transition state searches, analyzing networks, and digesting large amounts
of reaction data. 


Given the generality of these capabilities, we have refactored the original `yarp` workflows into the
object-oriented package that you are reading the docs for, so that other researchers can leverage these
algorithms in their own custom chemical workflows. 

**This software is still work in progress. Use at your own risk. Also take a look at the _license**

.. _license: https://github.com/bsavoie/yarp/blob/master/LICENSE
.. _issue: https://github.com/bsavoie/yarp/issues

.. toctree::
   :maxdepth: 4
   :numbered:
   :caption: Contents:

   overview.rst
   installation.rst
   yarpecule.rst
   ts.rst
   net.rst
   sieve.rst   
   worked_examples.rst
   api/modules.rst

Indices and tables
==================

.. * :ref:`genindex`
* :ref:`search`
