# **Classy-YARP for Transition Metal Complex Reaction: Example**
  * In this example, you will find:
    * [A folder](https://github.com/Savoie-Research-Group/yarp/tree/metal/Metal-Example/input_files) that contains clean input files
      * It contains:
      * [a yaml file](https://github.com/Savoie-Research-Group/yarp/blob/metal/Metal-Example/input_files/parameters.yaml) for input keywords
      * job submission files for [xtb](https://github.com/Savoie-Research-Group/yarp/blob/metal/Metal-Example/input_files/job_xtb.submit) and for [DFT](https://github.com/Savoie-Research-Group/yarp/blob/metal/Metal-Example/input_files/job_dft.submit)
      * [a folder with xyz file](https://github.com/Savoie-Research-Group/yarp/tree/metal/Metal-Example/input_files/reaction_xyz) for the input reaction
    * Another folder that contains result files

# Reference:
  * Please see **Figure 4** in [Kim et al.](https://doi.org/10.1021/jacs.3c00500)

# How to use:
  1. Install classy-YARP
     * Please see [here](https://github.com/Savoie-Research-Group/yarp/blob/master/README.md) about installing classy-YARP 
  3. Download the current folder
  4. Change the directories in the job submission files ([xtb](https://github.com/Savoie-Research-Group/yarp/blob/metal/Metal-Example/input_files/job_xtb.submit) and [DFT](https://github.com/Savoie-Research-Group/yarp/blob/metal/Metal-Example/input_files/job_dft.submit)) to the location of your [wrapper_functions/](https://github.com/Savoie-Research-Group/yarp/tree/metal/Metal-Example/wrapper_functions)
  5. Prepare your input xyz file, see [here](https://github.com/Savoie-Research-Group/yarp/blob/metal/Metal-Example/input_files/reaction_xyz/4-5-inp.xyz) for example.
     * Up = reactant, bottom = product
  6. Setup your [parameters.yaml](https://github.com/Savoie-Research-Group/yarp/blob/metal/Metal-Example/input_files/parameters.yaml) file 
