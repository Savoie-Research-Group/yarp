#!/bin/env python                                                                                                                                                             
# Author: Qiyuan Zhao (zhaoqy1996@gmail.com), Zhao Li

import subprocess
import os
import time

# for parallel parallel jobs (e.g., multiple jobs in parallel, each asks for multiple cpus), check the below contents to see if it works
'''
# Run the crest job in parallel with the desired number of threads
OMP_NUM_THREADS=4 crest X &
OMP_NUM_THREADS=4 crest Y &
OMP_NUM_THREADS=4 crest Z &

wait
'''
# compared with directly call crest in each line

class SLURM_Job:
    def __init__(self, submit_path='.', partition='standby', time=4, jobname='JobSubmission', node=1, ppn=4, mem_per_cpu=1000, specify_array=False, email = ""):
        """
        Initialize slurm job parameters
        Time needs to be specify in hours
        """
        self.time = time
        self.jobname = jobname
        self.partition = partition
        self.node = node
        self.ppn = ppn
        self.mem = mem_per_cpu
        self.submit_path = submit_path
        self.specify_array = specify_array
        self.script_file = os.path.join(submit_path, jobname+'.submit')

        #Zhao's note: add Email notification
        self.email = email

    def submit(self):
        """
        Submit a SLURM job using the specified script file and return the job id
        """
        current_dir = os.getcwd()
        # go into the job.submit folder to submit the job
        os.chdir('/'.join(self.script_file.split('/')[:-1]))
        command = f"sbatch {self.script_file}"
        output = subprocess.run(command, shell=True, capture_output=True, text=True)
        # go back to current dir
        os.chdir(current_dir)
        self.job_id = output.stdout.split()[-1]
        
    def status(self):
        """
        Check the status of the SLURM job.
        """
        if hasattr(self, "job_id") is False:
            print("Haven't submitted this job yet, can not check the status of this job...")
            return "UNSUBMITTED"

        try:
            command = f"squeue -j {self.job_id} --noheader --format %T"
            output = subprocess.run(command, shell=True, capture_output=True, text=True)
            job_status = output.stdout.strip()

            if job_status == "":
                # Job ID not found, indicating the job has completed
                return "FINISHED"
            else:
                # Common status: RUNNING and PENDING
                return job_status
        except:
            return "UNKNOWN"

    def create_job_head(self):
        """
        Create a slurm job script for given settings
        """
        with open(self.script_file, "w") as f:
            f.write("#!/bin/bash\n")
            if self.specify_array: f.write(f"#SBATCH --array={self.specify_array}\n")
            f.write(f"#SBATCH --job-name={self.jobname}\n")
            f.write(f"#SBATCH --output={self.jobname}.out\n")
            f.write(f"#SBATCH --error={self.jobname}.err\n")
            f.write(f"#SBATCH -A {self.partition}\n")
            f.write(f"#SBATCH --nodes={self.node}\n")
            f.write(f"#SBATCH --ntasks-per-node={self.ppn}\n")
            f.write(f"#SBATCH --mem {self.mem*self.ppn}MB\n")

            if len(self.email) > 0:
                f.write(f"#SBATCH --mail-user={self.email}\n")
                f.write(f"#SBATCH --mail-type=BEGIN,END,FAIL\n")
            f.write(f"#SBATCH --time {self.time}:00:00\n\n")
            f.write("echo Running on host `hostname`\n")
            f.write("echo Start Time is `date`\n\n")


    def create_job_bottom(self):
        """
        Print job finishing time
        """
        with open(self.script_file, "a") as f:
            f.write("\necho End Time is `date`\n\n")
            
    def setup_orca_script(self):
        """
        Load in ORCA and OPENMPI
        """
        with open(self.script_file, "a") as f:
            f.write("\n# Load Orca\n")
            f.write("# set up env for Orca\n")
            f.write("module unload openmpi\n")
            f.write("module load intel-mkl\n")
            # f.write("module purge\n")
            f.write('export PATH="/depot/bsavoie/apps/orca_5_0_1_openmpi411:$PATH"\n')
            f.write('export LD_LIBRARY_PATH="/depot/bsavoie/apps/orca_5_0_1_openmpi411:$LD_LIBRARY_PATH"\n')
            f.write('export PATH="/depot/bsavoie/apps/openmpi_4_1_1/bin:$PATH"\n')
            f.write('export LD_LIBRARY_PATH="/depot/bsavoie/apps/openmpi_4_1_1/lib:$LD_LIBRARY_PATH"\n\n')

    def setup_qchem_script(self):
        """
        Load in QChem
        set for athena
        """
        with open(self.script_file, "a") as f:
            f.write("\n# Load QChem\n")
            f.write("source /home/paulzim/qchem/trunk2022/paul.set.local0\n")

    def create_orca_jobs(self, orca_job_list, parallel=False, orcapath=None):
        """
        Generate orca jobs
        NOTE:  a list of orca job objects needs to be provided
        """
        self.create_job_head()
        self.setup_orca_script()
        
        with open(self.script_file, "a") as f:
            for orcajob in orca_job_list:
                f.write("\n# cd into the submission directory\n")
                f.write(f"cd {orcajob.work_folder}\n\n")
                if orcapath is None: orcapath = '/depot/bsavoie/apps/orca_5_0_1_openmpi411/orca'
                if parallel: f.write(f"{orcapath} {orcajob.orca_input} > {orcajob.output} &\n\n")
                else: f.write(f"{orcapath} {orcajob.orca_input} > {orcajob.output} \n\n")
                
            f.write("wait\n")

        self.create_job_bottom()

    def create_qchem_jobs(self, qchem_job_list, parallel=False):
        """
        Generate QChem jobs
        NOTE: qchem_input_file needs to be provided with FULL path
        """
        self.create_job_head()
        self.setup_qchem_script()

        with open(self.script_file, "a") as f:
            for qchemjob in qchem_job_list:
                f.write("\n# cd into the submission directory\n")
                f.write(f"cd {qchemjob.work_folder}\n\n")
                if parallel: f.write(f"qchem -nt {qchemjob.nproc} {qchemjob.qchem_input} > {qchemjob.output} &\n\n")
                else: f.write(f"qchem -nt {qchemjob.nproc} {qchemjob.qchem_input} > {qchemjob.output}\n\n")

            f.write("wait\n")

        self.create_job_bottom()

    def create_gaussian_jobs(self, job_list, parallel=False):
        """
        Generate Gaussian16 jobs
        Note: a list of gaussian job objects needs to be provided.
        """
        self.create_job_head()
        with open(self.script_file, "a") as f:
            for job in job_list:
                f.write("# cd into the submission directory\n")
                f.write(f"cd {job.work_folder}\n\n")
                f.write(f"module load gaussian16/B.01\n")
                if parallel: f.write(f"g16 < {job.gjf} > {job.output}.{job.nprocs}.out &\n")
                else:
                    f.write(f"g16 < {job.gjf} > {job.output} &\n")
                f.write("\nwait\n")
        # stop here

    def create_gsm_jobs(self, gsm_job_list, gsm_thread=1):
        """
        Create a GSM job script using the specified script file.
        Avaiable methods: xTB, Orca, QChem
        """

        # check input
        if self.ppn % gsm_thread != 0:
            print(f"Make sure your total number of cpu (ppn={self.ppn}) is divisible by gsm thread")
            quit()

        self.create_job_head()

        # write head of GSM submission file
        with open(self.script_file, "a") as f:
            # specify job array
            f.write("item=$SLURM_ARRAY_TASK_ID\n")
            f.write(f'ID=`printf "%0*d\\n" {gsm_thread} ${{item}}`\n')

            # set up orca/qchem path
            if gsm_job_list[0].method.lower() =='orca': 
                self.setup_orca_script()
                gsm_exe = 'gsm.orca'
            elif gsm_job_list[0].method.lower() =='qchem': 
                self.setup_qchem_script()
                gsm_exe = 'gsm.qchem'
            else:
                # GFM-xTB use orca interface but is using xTB to generate pseudo-Orca files
                gsm_exe = 'gsm.orca'
                
            # set thread and load packages
            f.write(f"export OMP_NUM_THREADS={gsm_thread}\n")
            f.write("module load intel-mkl\n\n") # a specific setting for bell 

            for gsmjob in gsm_job_list:
                f.write(f'cd {gsmjob.work_folder}\n')
                f.write(f"./{gsm_exe} ${{item}} {self.ppn//gsm_thread} > {gsmjob.output}\nwait\n\n")

        self.create_job_bottom()

    def create_pysis_jobs(self, pysis_job_list, parallel=False):
        """
        Create a pysis job script using the pysis_job
        """
        self.create_job_head()

        with open(self.script_file, "a") as f:

            # set thread and load packages
            f.write('export PATH="/depot/bsavoie/apps/anaconda3/envs/yarp/bin:$PATH"\n')

            # set up GSM commands (only supports doing each task in sequential)
            for pysisjob in pysis_job_list:
                f.write("\n# cd into the submission directory\n")
                f.write(f"cd {pysisjob.work_folder}\n\n")
                if parallel: f.write(f"pysis {pysisjob.pysis_input} > {pysisjob.output} &\n\n")
                else: f.write(f"pysis {pysisjob.pysis_input} > {pysisjob.output}\n\n")

        self.create_job_bottom()

    def create_crest_jobs(self, crest_job_list):
        """
        Create a crest job script for crest jobs
        """
        self.create_job_head()

        with open(self.script_file, "a") as f:

            #Zhao's note: try fixing crest env
            #add module load anaconda 
            #conda activate <your current conda env>
            #This is just a fix on Purdue's machine, you can turn it off#
            #Zhao's note: purdue changed its default anaconda, switch!
            #f.write(f"module load anaconda\n")
            f.write(f"module load anaconda/2022.10-py39\n")
            f.write(f"conda activate copy-classy-yarp\n")

            f.write(f"export OMP_STACKSIZE={crest_job_list[0].mem}M\n")
            f.write(f"export OMP_NUM_THREADS={crest_job_list[0].nproc}\n")

            # set up GSM commands (only supports doing each task in sequential)
            for crestjob in crest_job_list:
                f.write("\n# cd into the submission directory\n")
                f.write(f"cd {crestjob.work_folder}\n\n")
                f.write("# Running crest jobs for the input file\n")
                f.write(f"{crestjob.command} > {crestjob.output}\nwait\n\n")

        self.create_job_bottom()

    def create_auto3d_jobs(self, auto3d_job_list):
        """
        Create a slurm script to run auto3d jobs
        """
        self.create_job_head()

        with open(self.script_file, "a") as f:
            for auto3djob in auto3d_job_list:                
                f.write("\n# cd into the submission directory\n")
                f.write(f"cd {auto3djob.work_folder}\n\n")
                f.write(f"{auto3djob.command}\nwait\n\n")

        self.create_job_bottom()

    def create_python_jobs(self, python_commands, anaconda_env_name, work_folders=None, thread=1):
        """
        Create a slurm script to run python jobs
        NOTE: input_file needs to be provided with FULL path
        """
        self.create_job_head()

        if work_folders is None: work_folders = ['.'] * len(python_commands)
        with open(self.script_file, "a") as f:
            for jobid,python_command in enumerate(python_commands):
                # first activate anaconda env
                f.write(f"source activate {anaconda_env_name}\n\n")
                f.write("\n# cd into the submission directory\n")
                f.write(f"cd {work_folders[jobid]}\n\n")
                f.write(f"{python_command}\nwait\n\n")

        self.create_job_bottom()

    '''
    def cleanup_files(self, file_paths):
        """
        Remove the specified files.
        """
        for file_path in file_paths:
            try:
                os.remove(file_path)
            except OSError:
                pass
    '''

class QSE_jobs:
    """
    Base class to manage submission of external jobs to Univa Grid Engine (QSE) resource manager.

    Attributes
    ----------
    jobname : str
        Flag to control whether QSE submission script is generated to run ORCA, CREST, or
        some other package. Currently only accepts "ORCA" or "CREST".

    module : str
        Command line input to load external software.
        For example, if ORCA job is wanted, this should be "module load orca".
        Syntax will depend on the cluster system running on.
        So we need to make this as flexible as possible.

    submit_path : str
        Directory from which submission script will be executed.
        Defaults to current working directory.

    queue : str
        Queue to submit job requests to. Default is general CPU queue at ND-CRC (long).
        Eventually, this should probably be the Savoie group's specific queue at ND-CRC.
        There's also a GPU queue, but YARP is for CPUs right now, not GPUs.

    ncpus : int
        Number of CPUs used to parallelize across for each QSE job instance.
        Defaults to 1 CPU.

    ntasks : int
        Number of tasks contained in the job array.
        Defaults to 1 task.

    email : str
        Email to send "abort, begin, end" updates to for submitted jobs.
        Default is to not include this in submission script.

    script_file : str
        Path to QSE submission script.

    job_id : ???
        Holder for QSE job ID generated after submission.

    Planning Notes
    --------------
    IMPORTANT: Need to improve robustness of the ORCA input/output file formatting!!!

    How do I documment the class functions?

    Have most things figured out for ORCA. Not yet set up for CREST/Gaussian.
    CREST is a priority. Gaussian will happen when it happens.

    Submission of jobs must be done on a frontend login node at ND-CRC. Therefore,
    we are limited to only execute the job submission script for 1 hour wall time.
    Need to add a routine for limiting walltime, so we don't get shutdown by the CRC staff.
    UPDATE: Turns out the 1 hour wall time is a soft limit set by the CRC staff.
    This is therefore an optional "nice to have", rather than a necessity.

    """

    # Constructor
    def __init__(self, package = "ORCA", jobname = "JobSubmission", module = "module load orca", submit_path = os.getcwd(), queue = "long", ncpus = 1, ntasks = 1, email= ""):

        # Required inputs (based on Notre Dame's Center for Research Computing requirements!)
        self.ncpus = ncpus
        self.queue = queue
        self.ntasks = ntasks

        self.jobname = jobname
        self.package = package
        self.module = module # line needed to load the necessary software
        self.submit_path = submit_path # assumes input files have already been generated in this location!

        self.email = email

        # Derived attributes
        self.script_file = os.path.join(submit_path, jobname+'.submit')
        self.job_id = None

    def prepare_submission_script(self):
        """
        Create a QSE submission script based on inputs from class initialization

        To-Do's:
        - Figure out what orca input files will be named
        - Figure out how crest executable line will read
        """

        with open(self.script_file, "w") as f:
            # Make script header with QSE-style resource requests
            f.write("#!/bin/bash\n")
            
            # Specify CPUs and compute queue
            f.write(f"#$ -pe smp {self.ncpus}\n") # ERM: for now, it's SMP or bust
            f.write(f"#$ -q {self.queue}\n")

            # Set up job array for multi-job submissions (default is only 1 job submission)
            if self.ntasks == 1 :
                f.write("#$ -t 1\n")
            else :
                f.write(f"#$ -t 1-{self.ntasks}\n")

            f.write(f"#$ -N {self.jobname}\n")

            if len(self.email) > 0:
                f.write(f"#$ -M {self.email}\n")
                f.write(f"#$ -m abe\n")

            # Discard standard output/error to prevent garbage file accumulation
            # This will be collected later for each job task
            f.write(f"#$ -o /dev/null\n")
            f.write(f"#$ -e /dev/null\n")

            # Collect info on compute resources
            f.write("echo Running on host `hostname`\n")
            f.write("echo Start Time is `date`\n\n")
            f.write("base_dir=$(PWD)\n")

            # Collect a list of folders to iterate through
            f.write("mapfile -t folder_array < <(find . -maxdepth 1 -type d ! -name '.' -exec basename {} \;)\n")

            # Open a "does this directory exist?" bash conditional
            f.write('if [ -d "${folder_array[$SGE_TASK_ID-1]}" ]; then\n')

            # Redirect output and error to the appropriate folder
            f.write('    exec > "${folder_array[$SGE_TASK_ID-1]}/output.log" 2> "${folder_array[$SGE_TASK_ID-1]}/error.log"\n')

            # Change to folder of interest
            f.write("cd ${folder_array[$SGE_TASK_ID-1]}\n")

            # Put in script body according to jobname input
            if self.package == "ORCA" :
                f.write("echo Loading ORCA\n")
                # Put in module load commands
                f.write(f"{self.module}")
                # Set up full path to ORCA for paralleliztion runs
                f.write("orca=$(which orca)\n")
                # Execute ORCA input file
                # ERM: Need to figure out what these files will be named!!!
                f.write(f"echo Executing ORCA job from $PWD\n")
                f.write(f"$orca {self.jobname}.in > {self.jobname}.out\n")
            elif self.package == "CREST" :
                # ERM : Not available from module load, but should work by activating classy YARP conda environment!?
                f.write("CREST stuff\n")
            else:
                # Throw a runtime error
                raise RuntimeError("QSE class currently only supports CREST and ORCA job submissions!")

            # Return to previous folder
            f.write('cd "$base_dir"\n')

            # Close the "does this directory exist?" bash conditional
            f.write('else\n')
            f.write('    echo "Error: Directory ${folder_array[$SGE_TASK_ID-1]} does not exist" > "error_job_${JOB_ID}_task_${SGE_TASK_ID}.log"\n')
            f.write('fi\n')

            # Make script footer
            f.write("\necho End Time is `date`\n\n")

    def submit(self):
        """
        Submit a QSE job array using the previously built script file 

        Save job IDs to class variable for access later

        To-do's:
        - update jobID parsing to distinguish between base job number and task ID
        - figure out a good default to use for self.job_id initialization
        """
        os.chdir(self.submit_path)
        current_dir = os.getcwd()
        print(f"Submitting jobs from {current_dir}")
        
        # Execute job submission via qsub
        command = f"qsub {self.script_file}"
        output = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        # save job ID somehow....
        # self.job_id = output.stdout.split()[-1] # need to modify this!??

