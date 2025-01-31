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
    TO-DO: add error if missing from input

dft_lot : str
    Level of theory (lot) desired to run density functional theory (DFT) at.
    TO-DO: what are acceptable inputs here????
    dft_lot: wB97X D4 def2-TZVP # the functional and basis set for DFT
    TO-DO: add default if missing from input

scratch : str
    Folder that output files will be placed
    TO-DO: add default if missing from input

package : str
    Software package (ORCA, Gaussian) to perform DFT with.
    Currently only ORCA is available!

dft_nprocs : int
    Number of CPUs available for each DFT calculation.

mem : int
    Memory (in GB) available for each DFT calculation.

charge : int
    Overall charge of molecule(s).
    This will be applied to all molecules in a batch submission.

multiplicity : int
    Spin multiplicity of molecule(s).
    This will be applied to all molecules in a batch submission.

solvent : ?????
    Turn implicit solvation on/off
    TO-DO: figure out what this should look like!

solvation_model : str
    Keyword for ORCA solvent model.
    TO-DO: describe how to find eligible inputs here.

dielectric : float
    Dielectric constant for implicit solvation.
    Modify this to match the solvent of choice.

dft_njobs : int
    Number of DFT jobs to be submitted.
    TO-DO: figure out the nuance of this interaction with "running_jobs"
    This will not always be the actual number of jobs submitted?
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
from utils import xyz_write
from wrappers.orca import ORCA
from job_submission import QSE_jobs

def main(args):

    # Read energies and atomic cartesian coordinates (geometries) of input XYZ files into a dictionary
    TS_dict=dict()
    
    if os.path.isfile(args["input"]):
        # For a single XYZ file
        energy, geom = xyz_parse(args["input"])
        TS_dict[args["input"].split("/")[-1].split(".")[0]] = dict() # store XYZ filename
        TS_dict[args["input"].split("/")[-1].split(".")[0]]["E"] = energy
        TS_dict[args["input"].split("/")[-1].split(".")[0]]["TSG"] = geom
    else:
        # For a directory of multiple XYZ files
        xyz_files = [args["input"]+"/"+i for i in os.listdir(args["input"]) if fnmatch.fnmatch(i, "*.xyz")]
        for i in xyz_files:
            energy, geom = xyz_parse(i)
            TS_dict[i.split("/")[-1].split(".")[0]] = dict() # store XYZ filename
            TS_dict[i.split("/")[-1].split(".")[0]]["E"] = energy
            TS_dict[i.split("/")[-1].split(".")[0]]["TSG"] = geom
    
    # Set up location to put output files
    scratch = args["scratch"]
    os.makedirs(scratch, exist_ok=True)

    # Read in DFT level of theory (lot) from input file
    # Join phrases together if multiple words are provided
    # ERM: only needed for Gaussian???? comment out for now :(
    # if len(args["dft_lot"].split()) > 1: dft_lot="/".join(args["dft_lot"].split())
    # else: dft_lot=args["dft_lot"]

    # Set up DFT input files for each TS object
    job_list = dict()
    running_jobs = []

    for i in TS_dict.keys():
        # Create a folder associated with each input file name
        wf = os.path.join(scratch, i)
        os.makedirs(wf, exist_ok=True)

        # Generate an XYZ file for each input
        xyz_file = os.path.join(wf, f"{i}.xyz")
        xyz_write(xyz_file, TS_dict[i]["E"], TS_dict[i]["TSG"])

        # Conditional tree for different DFT packages
        if args["package"] == "ORCA":
            # Initialize ORCA class object
            dft_job = ORCA(
                input_geo = xyz_file,
                work_folder = wf,
                nproc = int(args["dft_nprocs"]),
                mem = int(args["mem"])*1000,
                jobname = f"{i}-TSOPT",
                jobtype = "OptTS Freq",
                lot = args["dft_lot"],
                charge = args["charge"],
                multiplicity = args["multiplicity"],
                solvent = args["solvent"],
                solvation_model = args["solvation_model"],
                dielectric = args["dielectric"],
                writedown_xyz = True
            )

            # Generate input file
            dft_job.generate_geometry_settings(hess = True, hess_step = int(args["hess_recalc"]))
            dft_job.generate_input()

            # Save ORCA object for use later
            # ERM: Do I need to use this later, actually????
            job_list[i]=dft_job

            # Check if a completed job is present before adding to the list of jobs to run
            if dft_job.calculation_terminated_normally() is False: running_jobs.append(i)
        else :
            raise RuntimeError("It's ORCA or bust right now my friend, sorry!")

    # Conditional tree to submit DFT jobs after checking for already completed jobs
    if len(running_jobs) > 1 :
        # Do something clever to determine how many jobs to submit
        # TO-DO: figure out this black magic!
        n_jobs = len(running_jobs) // int(args["dft_njobs"])
        if len(running_jobs) % int(args["dft_njobs"]) > 0 :
            n_jobs += 1

        # Generate a single QSE job array submission script for all DFT jobs to be submitted
        all_dft_jobs = QSE_jobs(
            jobname = args["package"],
            module = "module load orca", # TO-DO: make this a user input
            submit_path = scratch, # QSE script is at same level as all XYZ repos!
            queue = "long",
            ncpus = int(args["dft_nprocs"]),
            ntasks = n_jobs,
        )
        all_dft_jobs.prepare_submission_script()

        # Submit stuff to the queue! (ERM: turn on after testing out whether things generate properly)
        # all_dft_jobs.submit()
    else:
        print("Looks like you've already done all the TS optimizations!")

    return

if __name__ == "__main__":
    parameters = sys.argv[1]
    with open(parameters, "r") as file:
        parameters = yaml.safe_load(file)

    main(parameters)