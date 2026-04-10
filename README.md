# Welcome to Yet Another Reaction Program (YARP)
Come on in, the water's (probably) fine!

Here is the available functionality for version 0.3.0 of the code
- Visualize and manipulate molecules with yarpecules
- Perform break-n, form-m product enumeration to generate chemical reaction networks
- Estimate forward and reverse reaction barriers with the Savoie Group's EGAT model trained on the RGD1 database
- Perform basic network analysis on chemical reaction networks

To access this version, checkout the associated tag:
```
git clone git@github.com:Savoie-Research-Group/yarp-again.git
git checkout v0.3.0
```

## Installation notes

First, get yourself a conda environment up and running by executing the command `conda env create -f environment.yml`
from the root directory. Then do a good old `conda activate yarp-again`

Also from the root directory of `yarp-again`, run `pip install -e .`
- Or if you're here to use YARP, not develop YARP, run `pip install .` for a non-editable installation 

How do you know everything's working correctly?
Run the test suite via the command `pytest -v test/` from the root directory.
You should see that all tests passed.

YARP relies extensively on containers to manage software dependencies necessary for reaction path characterization.
Most of these containers have been published to DockerHub, and will be automatically pulled down the first time you
run a YARP job which uses said containers.
A few notable exceptions to this exist, and we provide guidance on manually container set up steps below.

YARP is set up to interface with both Docker and Apptainer container services.
Please make sure these are installed/available prior to executing YARP jobs attempting to leverage these types of containers.

### Setting up ORCA container

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
Please see `containers/orca/example` for a minimal functionality example you can run to ensure your ORCA container has been properly configured.

## Getting started using yarp-again

Three modules are available in the `tutorials/` folder:
1. Introduction to the yarpecule object
2. Introduction to bnfm product enumeration
3. Introduction to the full main_yarp.py user workflow

## How to add tests cases to the pytest suite

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
    
## Local Documentation

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
