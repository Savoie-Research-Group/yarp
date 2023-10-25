.. Yarp documentation master file, created by
   sphinx-quickstart on Thu Aug  3 08:41:32 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


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


Check out the :doc:`usage` section for further information, including how to
:ref:`install <installation>` the project.

**This software is still work in progress. Use at your own risk.**

Take a look at the `license`_ or open an `issue`_.

.. _license: https://github.com/Savoie-Research-Group/yarp/blob/master/LICENSE.md
.. _issue: https://github.com/Savoie-Research-Group/yarp/issues

Modules
=======
.. toctree::
   installation
   usage
   yarpecule
   find_lewis
   enum
   hashes
   sieve
   input_parsers
   properties
   taffi_functions   
   :maxdepth: 1
   :numbered:
   :caption: Contents:
