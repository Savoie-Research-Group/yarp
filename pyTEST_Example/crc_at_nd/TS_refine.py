"""
This program aims to refine the transition states (TS) of a collection of XYZ files
using ORCA to perform DFT level TS optimizations.

To use this script:

```
conda activate classy-yarp
python TS_refine.py parameters.yaml
```

Relevant options to include in parameters.yaml:
-----------------------------------------------

input : str
    Path to either an XYZ file or a folder filled with XYZ files.
    This must be provided, or else the code will not run!

scratch : str (default scratch)
    Folder that output files will be placed in.

dft_lot : str (default wB97X D4 def2-TZVP)
    Level of theory (lot) desired to run density functional theory (DFT) at.
    For ORCA, please provide the following format:
    "[functional] [dispersion correction] [basis set]"

package : str (default ORCA)
    Software package (ORCA, Gaussian) to perform DFT with.
    Currently only ORCA is available!

dft_nprocs : int (default 1)
    Number of CPUs available for each DFT calculation.

mem : int (default 2)
    Memory (in GB) available for each DFT calculation.
    NOTE: this is not incorporated into QSE submission scripts!

hess_recalc : int (default 1)
    Controls how often ORCA recalculates the Hessian?

solvent : bool (default False)
    Turn implicit solvation on/off

solvation_model : str (default CPCM)
    Keyword for ORCA solvent model.

dielectric : float (default 0.0)
    Dielectric constant for implicit solvation.
    Modify this to match the solvent of choice.

charge : int (default 0)
    Overall charge of molecule(s).
    This will be applied to all molecules in a batch submission.

multiplicity : int (default 1)
    Spin multiplicity of molecule(s).
    This will be applied to all molecules in a batch submission.
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

    print("Hi there! Let's optimize some transition states using DFT in YARP!")

    input = str(args.get("input"))
    if input is None or input == "None":
        raise RuntimeError(
            "The 'input' field is missing from the supplied YAML file. Can't do much without some molecules!")
    print(f"  * Molecule files will be read from {input}")

    # Read energies and atomic cartesian coordinates (geometries) of input XYZ files into a dictionary
    TS_dict = dict()

    if os.path.isfile(input):
        # For a single XYZ file
        energy, geom = xyz_parse(input)
        # store XYZ filename
        TS_dict[input.split("/")[-1].split(".")[0]] = dict()
        TS_dict[input.split("/")[-1].split(".")[0]]["E"] = energy
        TS_dict[input.split("/")[-1].split(".")[0]]["TSG"] = geom
    else:
        # For a directory of multiple XYZ files
        xyz_files = [input+"/" +
                     i for i in os.listdir(input) if fnmatch.fnmatch(i, "*.xyz")]
        for i in xyz_files:
            energy, geom = xyz_parse(i)
            # store XYZ filename
            TS_dict[i.split("/")[-1].split(".")[0]] = dict()
            TS_dict[i.split("/")[-1].split(".")[0]]["E"] = energy
            TS_dict[i.split("/")[-1].split(".")[0]]["TSG"] = geom

    # Set up location to put output files
    scratch = args.get("scratch", "scratch")
    os.makedirs(scratch, exist_ok=True)
    print(f"  * Output files will be placed in {scratch}")
    print("-***-")

    # Set up DFT input files for each TS object
    print("Initializing DFT job files:")
    job_list = dict()
    running_jobs = []

    package = str(args. get("package", "ORCA"))
    job_file_name = f"TSOPT"
    nproc = int(args.get("dft_nprocs", 1))
    mem = int(args.get("mem", 2)) * 1000
    job_mode = str(args.get("job_mode", "OptTS Freq"))
    level_of_theory = str(args.get("dft_lot", "wB97X D4 def2-TZVP"))
    charge = int(args.get("charge", 0))
    multiplicity = int(args.get("multiplicity", 1))
    solvent = bool(args.get("solvent", False))
    solv_model = str(args.get("solvation_model", "CPCM"))
    dielectric = float(args.get("dielectric", 0.0))

    hess_recalc = int(args.get("hess_recalc", 1))

    for i in TS_dict.keys():
        print(f"  - Processing {i}")

        # Create a folder associated with each input file name
        wf = os.path.join(scratch, i)
        os.makedirs(wf, exist_ok=True)

        # Generate an XYZ file for each input
        xyz_file = os.path.join(wf, f"{i}.xyz")
        xyz_write(xyz_file, TS_dict[i]["E"], TS_dict[i]["TSG"])

        # Conditional tree for different DFT packages
        if package == "ORCA":
            # Initialize ORCA class object
            dft_job = ORCA(
                input_geo=xyz_file,
                work_folder=wf,
                nproc=nproc,
                mem=mem,
                jobname=job_file_name,
                jobtype=job_mode,
                lot=level_of_theory,
                charge=charge,
                multiplicity=multiplicity,
                solvent=solvent,
                solvation_model=solv_model,
                dielectric=dielectric,
                writedown_xyz=False
            )

            # Generate input file
            dft_job.generate_geometry_settings(
                hess=True, hess_step=hess_recalc)
            dft_job.generate_input()

            # Save ORCA object for use later
            # ERM: Do I need to use this later, actually????
            job_list[i] = dft_job

            # Check if a completed job is present before adding to the list of jobs to run
            if dft_job.calculation_terminated_normally() is False:
                print(f"    * Adding job {i} to the queue to be run!")
                running_jobs.append(i)
            else:
                print(
                    f"    * Job {i} has already been run! Excluding from the queue.")
        else:
            raise RuntimeError("It's ORCA or bust right now my friend, sorry!")

    n_jobs = len(running_jobs)

    print("-***-")
    print(f"By my calculations, we've got {n_jobs} to run!")

    scheduler = str(args.get("scheduler", "QSE"))
    if scheduler == "QSE":
        # Generate a single QSE job array submission script for all DFT jobs to be submitted
        all_dft_jobs = QSE_jobs(
            package=args["package"],
            jobname=job_file_name,
            module="module load orca\n",  # TO-DO: make this a user input
            submit_path=scratch,  # QSE script is at same level as all XYZ repos!
            queue="long",
            ncpus=int(args["dft_nprocs"]),
            ntasks=n_jobs
        )
    elif scheduler == "condor":
        all_dft_jobs = "bob"
        raise RuntimeError("Condor is a work in progress!")
    else:
        raise RuntimeError(
            "Sorry, only QSE or condor are valid entries in scheduler!")

    print("  - Preparing submission script(s)")
    all_dft_jobs.prepare_submission_script()

    # Submit stuff to the queue! (ERM: turn on after testing out whether things generate properly)
    print("  - Submitting jobs to the queue!")
    all_dft_jobs.submit()

    print("-***-")
    print("Alrighty, I am signing off now! Hope all goes well with the jobs that are running!")

    return


if __name__ == "__main__":
    parameters = sys.argv[1]
    with open(parameters, "r") as file:
        parameters = yaml.safe_load(file)

    main(parameters)
