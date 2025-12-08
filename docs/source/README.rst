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
