<<<<<<< HEAD
0. SPECIAL THING ABOUT THIS BRANCH (in ``pyTEST_Example/``)
===========================================================

- ``class_main_dft.py`` and ``class_refinement.py`` as a replacement for
  ``main_dft.py`` and ``TS_refinement.py``
- These two ``class*`` files uses **DFT_Class** for handling rxns and TS
  conformers and send them to DFT processes (in ``DFT_Class.py``)
- DFT processes are written in classes: ``tsopt.py`` for **TSOPT**,
  ``irc.py`` for **IRC**, ``opt.py`` for geometry optimization
- ``DFT_Class.py`` also contains information about job scheduling and
  restarting.

YARP: Yet Another Reaction Program for Automated Reaction Exploration
=====================================================================

This is an object-oriented refactoring of yarp methodology to be
compatible with general reaction-discovery workflows. YARP is a
repository that explore the reaction network for a given set of
molecules (reactants). YARP consists of construction of bond-electron
matrix, product enumeration, reaction conformational sampling,
transition state (TS) localization by growing string method (GSM), and
TS characterization by Berny optimzation and IRC calculation. More
details could be found on:

1. https://www.nature.com/articles/s43588-021-00101-3
2. https://pubs.acs.org/doi/full/10.1021/acs.jctc.2c00081

1. Installation:
================

- 1.1 Anaconda is recommended and you can get YARP repository by:

::

   git clone https://github.com/Savoie-Research-Group/yarp.git

- 1.2 Build the YARP environment and install required packages by:

::

   cd yarp
   conda env create -f env_linux.yaml

- 1.3 Install yarp package by:

::

   conda activate classy-yarp
   pip install .

- 1.4 Create pysis cmd by:

::

   vi ~/.pysisyphusrc

and in ``.pysisyphusrc`` file add:

::

   [xtb]
   cmd=xtb

Finally, save and quit this file.

2. Test Your Installation
=========================

- 2.1 First install pytest via:

::

   pip install pytest

- 2.2 Do a simple test (few seconds) that reads the structure of Fe(CO)5
  and returns the **bond-electron matrix**:

::

   cd examples/
   pytest -s

- 2.3 Then do a more complicated test (10 mins) that runs **YARP-xTB**
  that includes **geometry optimization**, **conformational sampling**,
  **joint optimization**, **Grow-String Method (GSM)**, **Berny
  optimization**, and **IRC**

::

   cd pyTEST_Example/
   pytest -s

3. How YARP Works:
==================

- 3.1 YARP class:

  - In yarp/yarp folder, several functions are shown
  - the bond-electron matrix and reaction enumeration are generated via
    these functions. More details are shown in
    ``yarp/examples/reaction.py``

- 3.2 Reaction class:

  - All calculations, including conformational sampling, GSM, and TS
    characterization, are In ``yarp/reaction`` folder.
  - In ``yarp/reaction/main_xtb.py``, the workflow are shown. To run
    ``main_xtb.py`` we need a ``parameters.yaml`` file.
  - To build a ``parameters.yaml`` file, we will need:

  ::

       input: [reaction folder or reaction file] # it could be a folder (store several xyz, mol, or smiles files) or a xyz, mol, or smiles file.
       scratch: [a path you want to store result]
       reaction_data: [a pickle file] # a pickle to store and load the result
       n_break: [an integer, n] # define a break-n-bonds-form-n-bonds reaction
       form_all: [True/False] # If true, it means you will try all break-n-bonds-form-infinity-bonds reactions.
       lewis_criteria: [a float] # the criteria to remove unfavorable products by its Lewis structure (0.0 is recommended).
       ff: [force field name] # the force field to initialize geometry (uff/mmff94 are all available)
       crest: [cmd for crest] # the command for CREST. (usually it would be crest)
       lot: [gfn2, gfn1, gfn-ff] # the level of theory in xTB.
       method: [crest/rdkit] # for conformational sampling, CREST and rdkit (by uff) are available options.
       enumeration: [True/False] # If it's true, it means that product enumeration will be applied.
       n_conf: [an integer, n] # define n conformers would be considered for further calculations.
       nconf_dft: [an integer, n] # define n conformers for DFT calculations.
       c_nprocs: [an integer] # define how many cpus you are going to use for CREST. (if you run on pc, set this as 1)
       mem: [integer] # in GB
       opt_level: [tight, vtight, and etc.] # define the convergence for xTB.
       crest_quick: [True/False] # if true, the crest with quick mode will be performed.
       low_solvation: [gbsa, alpb, False] # define the solvation models in xTB.
       solvent: [solvent name] # define the solvent.
       pysis_wt: [integer] # the walltime for xTB calculation (in second).
       charge: [integer] # charge of your system.
       multiplicity: [integer] # multiplicity for your system.
       model_path: [your yarp path/reaction/bin] # models we need for yarp.
       gsm_inp: [your yarp path/reaction/bin/inpfileq] # growing string method bin file.

  More details could be found in ``parameters.yaml``.

- 3.3 As the ``parameter.yaml`` file is created, you can run yarp by:

::

   python main_xtb.py [your parameter yaml file]
=======
Welcome to yarp-again
=====================

Come on in, the water’s (probably) fine!

Installation notes
------------------

First, get yourself a conda environment up and running by executing the
command ``conda env create -f environment.yml`` from the root directory.
Then do a good old ``conda activate yarp-again``

Also from the root directory of ``yarp-again``, run ``pip install -e .``
- Or if you’re here to use YARP, not develop YARP, run ``pip install .``
for a non-editable installation

How do you know everything’s working correctly? Run the test suite via
the command ``pytest test/ -v`` from the root directory. You should see
that all tests passed

Here’s a working list of all the packages I’ve installed, **and why**
(-ERM) - ``conda install python``: When I ran this, it pulled in version
3.13.2 - ``conda install pyaml``: For reading YAML input files from
command-line - ``conda install numpy``: Because numpy is a friend to all
- ``conda install pytest``: For automated unit/regression testing! -
``conda install scipy``: Adjacency matrix routines need this. Other
stuff might also - ``pip install rdkit``: RDKit is a friend to… many
(i.e. reading .mol files and processing SMILES)

How to add tests cases to the pytest suite
------------------------------------------

All tests should be contained in the folder ``test/``. The structure of
``test/`` should mirror the structure of the source code contained in
``yarp/``. - For example, the source code file
``yarp/yarpecule/input_parsers.py`` as a corresponding testing file
``test/yarpecule/test_input_parsers.py`` - It is important that all
testing files start with ``test_`` so that pytest will detect them.

The file ``test/conftest.py`` contains pytest fixtures, which are
objects that all tests within ``test/`` have access to and can use in
their tests. - Example: ``test_input_parsers.py`` uses the fixture
``ethene_xyz`` as an input for its tests related to parsing XYZ files -
Feel free to add additional XYZ and MOL files to the ``test/molecules/``
folder, and use them to set up additional fixtures in ``conftest.py``

Local Documentation
-------------------

If you’ve cloned the repo and want to browse the documentation locally:

1. Activate the ``yarp_env`` Conda environment

2. Run the build script:

   .. code:: bash

      cd docs
      ./update_docs.sh

After building the docs with ``./update_docs.sh``, view the docs by
running: ./view_docs.sh

| The HTML files are located in:
| 📁 ``docs/build/html/index.html``

.. _installation-notes-1:

Installation notes
------------------

First, get yourself a conda environment up and running by executing the
command ``conda env create -f environment.yml`` from the root directory.
Then do a good old ``conda activate yarp-again``

Also from the root directory of ``yarp-again``, run ``pip install -e .``
- Or if you’re here to use YARP, not develop YARP, run ``pip install .``
for a non-editable installation

How do you know everything’s working correctly? Run the test suite via
the command ``pytest test/ -v`` from the root directory. You should see
that all tests passed

Here’s a working list of all the packages I’ve installed, **and why**
(-ERM) - ``conda install python``: When I ran this, it pulled in version
3.13.2 - ``conda install pyaml``: For reading YAML input files from
command-line - ``conda install numpy``: Because numpy is a friend to all
- ``conda install pytest``: For automated unit/regression testing! -
``conda install scipy``: Adjacency matrix routines need this. Other
stuff might also - ``pip install rdkit``: RDKit is a friend to… many
(i.e. reading .mol files and processing SMILES)

.. _how-to-add-tests-cases-to-the-pytest-suite-1:

How to add tests cases to the pytest suite
------------------------------------------

All tests should be contained in the folder ``test/``. The structure of
``test/`` should mirror the structure of the source code contained in
``yarp/``. - For example, the source code file
``yarp/yarpecule/input_parsers.py`` as a corresponding testing file
``test/yarpecule/test_input_parsers.py`` - It is important that all
testing files start with ``test_`` so that pytest will detect them.

The file ``test/conftest.py`` contains pytest fixtures, which are
objects that all tests within ``test/`` have access to and can use in
their tests. - Example: ``test_input_parsers.py`` uses the fixture
``ethene_xyz`` as an input for its tests related to parsing XYZ files -
Feel free to add additional XYZ and MOL files to the ``test/molecules/``
folder, and use them to set up additional fixtures in ``conftest.py``
>>>>>>> yarp-again/main
