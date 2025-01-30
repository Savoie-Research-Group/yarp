"""
This program aims to refine the transition states (TS) of a collection of XYZ files
using ORCA to perform DFT level TS optimizations.

To use this script:

```
conda activate classy-yarp
python TS_refine.py parameters.yaml
```

Tags to include in parameters.yaml:
-----------------------------------

input : str
    Path to either an XYZ file or a folder filled with XYZ files

dft_lot : str
    Level of theory (lot) desired to run density functional theory (DFT) at.
    TO-DO: what are acceptable inputs here????

"""

import sys
import os
import yaml
import fnmatch

# Add the parent directory to the system path
# Now python should be able to find other python module files,
# even if they live in the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from yarp.input_parsers import xyz_parse
from wrappers.orca import ORCA
from job_submission import *


def main(args):

    # Read energies and atomic cartesian coordinates (geometries) of TS inputs into a dictionary
    TS_dict=dict()
    
    if os.path.isfile(args["input"]):
        # For a single XYZ file
        energy, geom = xyz_parse(args["input"])
        TS_dict[args["input"].split("/")[-1].split(".")[0]] = dict()
        TS_dict[args["input"].split("/")[-1].split(".")[0]]["E"] = energy
        TS_dict[args["input"].split("/")[-1].split(".")[0]]["TSG"] = geom
    else:
        # For a directory of multiple XYZ files
        xyz_files=[args["input"]+"/"+i for i in os.listdir(args["input"]) if fnmatch.fnmatch(i, "*.xyz")]
        for i in xyz_files:
            energy, geom = xyz_parse(i)
            TS_dict[i.split("/")[-1].split(".")[0]]=dict()
            TS_dict[i.split("/")[-1].split(".")[0]]["E"] = energy
            TS_dict[i.split("/")[-1].split(".")[0]]["TSG"] = geom
    
    # Read in DFT level of theory (lot) from input file
    # Join phrases together if multiple words provided
    if len(args["dft_lot"].split()) > 1: dft_lot="/".join(args["dft_lot"].split())
    else: dft_lot=args["dft_lot"]
    
    return

if __name__=="__main__":
    parameters = sys.argv[1]
    parameters = yaml.load(open(parameters, "r"), Loader=yaml.FullLoader)
    main(parameters)