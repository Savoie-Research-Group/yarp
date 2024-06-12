# **Classy-YARP for Transition Metal Complex Reaction: Example**
  * In this example, you will find:
    * [A folder](Metal-Example/input_files) that contains clean input files
      * It contains:
      * [a yaml file](Metal-Example/input_files/parameters.yaml) for input keywords
      * job submission files for [xtb](Metal-Example/input_files/job_xtb.submit) and for [DFT](Metal-Example/input_files/job_dft.submit)
      * [a folder with xyz file](Metal-Example/input_files/reaction_xyz) for the input reaction
    * A folder that contains result files
    * A folder with wrapper functions to call xtb/DFT calculators

# Reference:
  * Please see **Figure 4** in [Kim et al.](https://doi.org/10.1021/jacs.3c00500)
  * This example focuses on reaction of **Oxidative Addition with Cu**.

# How to use:
  1. Install classy-YARP
     * Please see [here](https://github.com/Savoie-Research-Group/yarp/blob/master/README.md) about installing classy-YARP 
  3. Download the current folder
  4. Change the directories in the job submission files ([xtb](Metal-Example/input_files/job_xtb.submit) and [DFT](Metal-Example/input_files/job_dft.submit)) to the location of your [wrapper_functions/](Metal-Example/wrapper_functions)
  5. Prepare your input xyz file, see [here](Metal-Example/input_files/reaction_xyz/4-5-inp.xyz) for example.
     * Up = reactant, bottom = product
  6. Setup your [parameters.yaml](Metal-Example/input_files/parameters.yaml) file 
