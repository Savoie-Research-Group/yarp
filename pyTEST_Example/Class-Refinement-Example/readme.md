# Class-based refactor of DFT
* This script will replace `TS_refinement.py` using a class approach
## DFT processes are divided into **classes**
  * For example, `class_refinement.py` has `TSOPT` and `IRC` classes
## Files and usage
  * Needs: `job_test.submit`, `SINGLE_RXN.p`, `parameters.yaml`
  * run `./job_test.submit`
  * :memo: NOTE: Also need to prepare a folder of xyz files somewhere (defined in `parameters.yaml`)
  * Usage can be divided into two modes:
    1. First time usage: files are copied into the current folder
      * reaction classes will be created by reading rxn template from `SINGLE_RXN.p`
      * RxnDFT, ConformerDFT classes will be created
      * Each Conformer class will have TSOPT and IRC class
      * ConformerDFT will record the current stage: `TSOPT` or `IRC`
      * TSOPT and IRC classes will record job status: Running, submitted, Initialized, ...
      * At the end of the first run, everything will be saved into `REFINE.p` for picking up the jobs later (with record of all jobs submitted/running)
    2. Second time and later usage: 
      * It will pickup `REFINE.p`
      * It will check the job status recorded, see if they are done or still pending
      * if dead, it will restart the job
      * If the process is dead and beyond saving, this Conformer will be ignored and there is no further action
        * :memo: NOTE: for example, if a TS conformer has multiple imag frequencies, this TS conformer will not do IRC
