# Welcome to yarp-again
Come on in, the water's (probably) fine!

## Installation notes

First, get yourself a conda environment up and running by executing the command `conda env create -f environment.yml`
from the root directory.

Also from the root directory of `yarp-again`, run `pip install -e .`
- Or if you're here to use YARP, not develop YARP, run `pip install .` for a non-editable installation 

Here's a working list of all the packages I've installed, **and why** (-ERM)
- `conda install python`: When I ran this, it pulled in version 3.13.2
- `conda install pyaml`: For reading YAML input files from command-line
- `conda install numpy`: Because numpy is a friend to all
- `conda install pytest`: For automated unit/regression testing!
