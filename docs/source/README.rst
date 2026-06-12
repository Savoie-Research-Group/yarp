Welcome to Yet Another Reaction Program (YARP)
==============================================

High throughput reaction characterization engine for the exploration of
chemical reaction networks and beyond!

Here is the available functionality for YARP version 3.0.0: - Visualize
and manipulate molecules with yarpecules - Perform multiple cycles of
break-n, form-m product enumeration to generate chemical reaction
networks - Estimate forward and reverse reaction barriers with the
Savoie Group’s EGAT model trained on the RGD1 database - Generate an
initial guess at a TS structure using CREST conformer generation +
double-ended growing string method at the xTB level of theory - Refine
reaction paths (reactant - transition state - product) at both the xTB
and DFT levels of theory, using Pysisyphus and ORCA, respectively -
Perform basic network analysis on chemical reaction networks

Have questions or find a bug? Please hop over to our Issues tab, and let
us know!

Getting Started with YARP
-------------------------

Detailed examples and user instructions can be found in ``tutorials/``!

Installation Notes
~~~~~~~~~~~~~~~~~~

To access the latest (possibly experimental) version of the code:

.. code:: bash

   git clone git@github.com:Savoie-Research-Group/yarp.git

For the latest stable release, you can access the following tag:

.. code:: bash

   git checkout v3.0.0

Once you have your desired source code version, get yourself a conda
environment up and running by executing the command
``conda env create -f environment.yml`` from the root directory. Then do
a good old ``conda activate yarp``

Also from the root directory of this git repository, run
``pip install -e .`` - This will give you access to the ``yarp-init``,
``yarp-progress``, and ``yarp-loop`` executables - If you’re here to use
YARP, not develop YARP, feel free to run ``pip install .`` instead for a
non-editable installation

**How do you know everything’s working correctly?**

Run the test suite via the command ``pytest test/`` from the base-level
of this git repository. You should see that all tests passed.

Quick Start - Basic Usage
~~~~~~~~~~~~~~~~~~~~~~~~~

There are three core YARP execution scripts: 1. ``yarp-init``: (source
code –> ``yarp/initialize_yarp.py``) - Parses user input configuration,
initializes reaction objects, and sets up task scheduler - If product
enumeration is requested by user, that logic is executed here - How to
use from command-line:
``bash     cd /path/to/your/working/directory     yarp-init input.yaml``
2. ``yarp-progress``: (source code –> ``yarp/progress_yarp.py``) -
Manages the directed acyclic graph workflow for individual YARP tasks -
Checks active jobs, submits additional ones, analyzes completed jobs -
Requires user to re-execute in order to complete all tasks specified by
``yarp-init`` - How to use from command-line:
``bash     yarp-progress /path/to/your/working/directory`` 3.
``yarp-loop``: (source code –> ``yarp/run_yarp_loop.py``) - This is a
convenience wrapper for ``yarp-progress``, which will re-execute that
script at set intervals - All YARP outputs will be written to a
``yarp_loop.out`` file, generated in the provided working directory -
How to use from command-line (via nohup background processes):
``bash     nohup yarp-loop -w /path/to/your/working/directory -i <interval_length_in_minutes> -d <total_duration_in_minutes> &``

YARP provides quite a lot of options for reaction exploration and
characterization. These are all accessible through the YAML input file
provided to ``yarp-init`` Checkout the ``tutorials/`` folder for more
details, but here is a minimal example YARP input:

.. code:: yaml

   # PART 1: Initialization of reactions and task manager settings
   initialize:
     initial_structure: # block to control starting point structure(s) for YARP
       source: O=CCOO
       type: smiles
       mode: species
     output: YARP_RXNS.pkl       # where YARP reaction objects are written to (default = YARP_RXNS.pkl)
     status: STATUS.json         # keeps track of all reaction characterization tasks (default = STATUS.json)
     job manager:                # block to define how reaction characterization tasks are executed
        scheduler: local
        container: docker
        max_active_jobs: 4       # upper limit of how many jobs can be submitted (default = 100)
     enumeration:                # block to control product enumeration; if absent, no product enum will be performed
       mode: concerted
       n_break: 2
       n_form: 2

   # PART 2: Characterization of reactions
   stages:
    - egat                       # these can be set to anything, so long as they match downstream header blocks

   egat:
     method: ml_rxn_prop         # these are pre-set options that determine what tasks will be put in the pipeline for a given reaction!
     model: egat_rgd1
     n_cpus: 8
     mem_per_cpu: 1000           # in MB

Heads up about containers!
--------------------------

YARP relies extensively on containers to manage software dependencies
necessary for reaction path characterization. YARP is set up to
interface with both Docker and Apptainer container services. Typically,
we set up Docker when running YARP on personal computers and use
Apptainer for HPC systems, where root-level access is restricted. Please
make sure these are installed/available prior to executing YARP jobs!

Most of YARP’s containers have been published to DockerHub, and will be
automatically pulled down the first time you run a YARP job which uses
said containers. Expect a few minutes of delay in these cases. A few
notable exceptions to this exist, and we provide guidance on manual
container set up steps below.

Setting up ORCA container
^^^^^^^^^^^^^^^^^^^^^^^^^

As per ORCA’s EULA license, we are not able to distribute ORCA’s
software binaries. However, the software is freely available for
download at the `ORCA
Forum <https://orcaforum.kofo.mpg.de/app.php/portal>`__

To build an ORCA container, take the following steps: 1. Download
``orca_6_0_1_linux_x86-64_shared_openmpi416.tar.xz`` from the ORCA Forum
portal 2. Place the tar file into ``containers/orca`` 3. For a Docker
container, ``cd containers/orca`` and execute
``docker build -t orca:6.0.1 .`` from the command-line 4. For an
Apptainer container, ``cd containers/orca`` and execute
``apptainer build ../orca_6.0.1.sif orca_6.0.1.def`` - This will place
the Apptainer container in the default location that YARP looks for
``.sif`` files! - However, you can place the ``.sif`` file anywhere and
provide the absolute file path as an input to ``initialize_yarp.py``
when executing YARP jobs via Apptainer

Need to use a different version of ORCA? Feel free to adjust the
relevant sections of our provided Docker and Apptainer files.

Relevant portion of ``containers/orca/Dockerfile``:

.. code:: bash

   # Create a micromamba environment containing exactly OpenMPI 4.1.6, xTB, and CREST
   RUN micromamba create -y -p /home/${USER}/env -c conda-forge openmpi=4.1.6 xtb crest && \
       micromamba clean --all --yes

   # Copy the ORCA tarball into the image and extract it
   # Assuming it is named 'orca_6_0_1_linux_x86-64_shared_openmpi416.tar.xz'
   COPY --chown=${USER}:${GROUP} orca_6_0_1_linux_x86-64_shared_openmpi416.tar.xz /home/${USER}/
   RUN mkdir /home/${USER}/orca && \
       tar -xf /home/${USER}/orca_6_0_1_linux_x86-64_shared_openmpi416.tar.xz -C /home/${USER}/orca --strip-components=1 && \
       rm /home/${USER}/orca_6_0_1_linux_x86-64_shared_openmpi416.tar.xz

Relevant portion of ``containers/orca/orca_6.0.1.def``:

.. code:: bash

   %files
       # Copy the ORCA tarball from your host machine into the container's /opt directory
       orca_6_0_1_linux_x86-64_shared_openmpi416.tar.xz /opt/orca_installer.tar.xz

   ...

   %post

   ...

       # Create a micromamba environment containing exactly OpenMPI 4.1.6, xTB, and CREST
       # We install this into /opt/env instead of a user's home directory
       micromamba create -y -p /opt/env -c conda-forge openmpi=4.1.6 xtb crest
       micromamba clean --all --yes

       # Extract the ORCA tarball we copied in the %files section into /opt/orca
       mkdir -p /opt/orca
       tar -xf /opt/orca_installer.tar.xz -C /opt/orca --strip-components=1

**WARNING: Make sure you use the correct OpenMPI version associated with
your specific version of ORCA!!!!** Many performance issues arise from
having the wrong OpenMPI version, most notably, parallelization. Please
see ``containers/orca/example`` for a minimal functionality example you
can run to ensure your ORCA container has been properly configured.
