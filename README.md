# Welcome to Yet Another Reaction Program (YARP)
Come on in, the water's (probably) fine!

Here is the available functionality for version 0.3.0 of the code:
- Visualize and manipulate molecules with yarpecules
- Perform break-n, form-m product enumeration to generate chemical reaction networks
- Estimate forward and reverse reaction barriers with the Savoie Group's EGAT model trained on the RGD1 database
- Generate an initial guess at a TS structure using CREST conformer generation + double-ended growing string method at the xTB level of theory
- Refine reaction paths (reactant - transition state - product) at both the xTB and DFT levels of theory, using Pysisyphus and ORCA, respectively
- Perform basic network analysis on chemical reaction networks

To access this version, checkout the associated tag:
```
git clone git@github.com:Savoie-Research-Group/yarp-again.git
git checkout v0.3.0
```

## Getting Started with YARP

### Basic Usage

There are three core YARP execution scripts:
1. `yarp-init`: (source code --> `yarp/initialize_yarp.py`)
    - Parses user input configuration, initializes reaction objects, and sets up task scheduler
    - If product enumeration is requested by user, that logic is executed here
    - How to use from command-line:
    ```
    cd /path/to/your/working/directory
    yarp-init input.yaml
    ```
2. `yarp-progress`: (source code --> `yarp/progress_yarp.py`)
    - Manages the directed acyclic graph workflow for individual YARP tasks
    - Checks active jobs, submits additional ones, analyzes completed jobs
    - Requires user to re-execute in order to complete all tasks specified by `yarp-init`
    - How to use from command-line:
    ```
    yarp-progress /path/to/your/working/directory
    ```
3. `yarp-loop`: (source code --> `yarp/run_yarp_loop.py`)
    - This is a convenience wrapper for `yarp-progress`, which will re-execute that script at set intervals
    - How to use from command-line (via nohup background processes):
    ```
    nohup yarp-loop -w /path/to/your/working/directory -i <interval_length_in_minutes> -d <total_duration_in_minutes> > yarp_loop.out &
    ```

YARP provides quite a lot of options for reaction exploration and characterization.
These are all accessible through the YAML input file provided to `yarp-init`
Checkout the `tutorials/` folder for more details, but here is a minimal example YARP input:
```
# PART 1: Initialization of reactions and task manager settings
initialize:
  initial species: OCC=O      # can also be a path to an XYZ or MOL file, or a YARP pickle file
  output: YARP_RXNS.pkl       # where YARP reaction objects are written to (default = YARP_RXNS.pkl)
  status: STATUS.json         # keeps track of all reaction characterization tasks (default = STATUS.json)
  job manager:                # block to define how reaction characterization tasks are executed
     scheduler: local
     container: docker
     max_active_jobs: 4       # upper limit of how many jobs can be submitted (default = 100)
  enumeration:                # block to control product enumeration
     enumerate: True

# PART 2: Characterization of reactions
stages:
 - egat                       # these can be set to anything, so long as they match downstream header blocks

egat:
  method: ml_rxn_prop         # these are pre-set options that determine what tasks will be put in the pipeline for a given reaction!
  model: egat_rgd1
  n_cpus: 1
  mem_per_cpu: 1000           # in MB
```

### Enumeration notes

The `reactive atoms` enumeration option restricts bond breaking/forming to the
selected atom-map IDs. These are zero-based YARP atom maps, not necessarily the
current local atom indices of a candidate molecule. For unmapped input SMILES,
XYZ, or MOL files, YARP generates zero-based atom maps during yarpecule
construction; use the mapped SMILES printed by YARP to audit the selection.

Example:

```yaml
initialize:
  enumeration:
     enumerate: True
     mode: concerted
     bonds to break: 2
     bonds to form: 2
     reactive atoms: [0, 1, 2, 3] # atom-map IDs to include in enumeration
```

When restarting from a YARP pickle and using `separate products`, YARP validates
the reactive atom-map list against the unseparated product before separation.
Separated fragments that contain none of the selected maps are skipped rather
than enumerated without the restriction.

### Installation Notes

First, get yourself a conda environment up and running by executing the command `conda env create -f environment.yml`
from the root directory. Then do a good old `conda activate yarp`

Also from the root directory of this git repository, run `pip install -e .`
- This will give you access to the `yarp-init`, `yarp-progress`, and `yarp-loop` executables
- If you're here to use YARP, not develop YARP, feel free to run `pip install .` instead for a non-editable installation

**How do you know everything's working correctly?**

Run the test suite via the command `pytest -v test/` from the base-level of this git repository.
You should see that all tests passed.

**Heads up about containers!**

YARP relies extensively on containers to manage software dependencies necessary for reaction path characterization.
YARP is set up to interface with both Docker and Apptainer container services.
Typically, we set up Docker when running YARP on personal computers and use Apptainer for HPC systems, where root-level access is restricted.
Please make sure these are installed/available prior to executing YARP jobs!

Most of YARP's containers have been published to DockerHub, and will be automatically pulled down the first time you
run a YARP job which uses said containers.
Expect a few minutes of delay in these cases.
A few notable exceptions to this exist, and we provide guidance on manual container set up steps below.

#### Setting up ORCA container

As per ORCA's EULA license, we are not able to distribute ORCA's software binaries.
However, the software is freely available for download at the [ORCA Forum](https://orcaforum.kofo.mpg.de/app.php/portal)

To build an ORCA container, take the following steps:
1. Download `orca_6_0_1_linux_x86-64_shared_openmpi416.tar.xz` from the ORCA Forum portal
2. Place the tar file into `containers/orca`
3. For a Docker container, `cd containers/orca` and execute `docker build -t orca:6.0.1 .` from the command-line
4. For an Apptainer container, `cd containers/orca` and execute `apptainer build ../orca_6.0.1.sif orca_6.0.1.def`
    - This will place the Apptainer container in the default location that YARP looks for `.sif` files!
    - However, you can place the `.sif` file anywhere and provide the absolute file path as an input to `initialize_yarp.py` when executing YARP jobs via Apptainer

The trickiest part of an ORCA installation is ensuring the parallelization features are working properly.
If you use our provided Docker/Apptainer files, you should be good to go, but it's always good to double-check these things.
Please see `containers/orca/example` for a minimal functionality example you can run to ensure your ORCA container has been properly configured.

## YARP Developer FYIs!

### How to add tests cases to the pytest suite

All tests should be contained in the folder `test/`.
The structure of `test/` should mirror the structure of the source code contained in `yarp/`.
- For example, the source code file `yarp/yarpecule/input_parsers.py` as a corresponding testing
    file `test/yarpecule/test_input_parsers.py`
- It is important that all testing files start with `test_` so that pytest will detect them.

The file `test/conftest.py` contains pytest fixtures, which are objects that all tests within `test/`
have access to and can use in their tests.
- Example: `test_input_parsers.py` uses the fixture `ethene_xyz` as an input for its
    tests related to parsing XYZ files
- Feel free to add additional XYZ and MOL files to the `test/molecules/` folder,
    and use them to set up additional fixtures in `conftest.py`
    
### Local Sphinx Documentation

If you've cloned the repo and want to browse the documentation locally:

1. Activate the `yarp_env` Conda environment
2. Run the build script:

    ```bash
    cd docs
    ./update_docs.sh
    ```

After building the docs with `./update_docs.sh`, view the docs by running:
./view_docs.sh

The HTML files are located in:  
📁 `docs/build/html/index.html`
