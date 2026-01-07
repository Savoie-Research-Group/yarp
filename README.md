# Welcome to yarp-again
Come on in, the water's (probably) fine!

Here is the available functionality for version 0.1.0 of the code
- Visualize and manipulate molecules with yarpecules
- Perform bnfm product enumeration to generate chemical reaction networks
- Estimate reaction barriers with the Savoie Group's EGAT model trained on the RGD1 database
- Perform basic network analysis on chemical reaction networks

To access this version, checkout the associated tag:
```
git clone git@github.com:Savoie-Research-Group/yarp-again.git
git checkout v0.1.0
```

## Installation notes

First, get yourself a conda environment up and running by executing the command `conda env create -f environment.yml`
from the root directory. Then do a good old `conda activate yarp-again`

Also from the root directory of `yarp-again`, run `pip install -e .`
- Or if you're here to use YARP, not develop YARP, run `pip install .` for a non-editable installation 

How do you know everything's working correctly?
Run the test suite via the command `pytest -v test/` from the root directory.
You should see that all tests passed.

**NOTE:** As of version 0.1.0, the conda environment has been found to work on Linux OS, but not on Mac OS.

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
