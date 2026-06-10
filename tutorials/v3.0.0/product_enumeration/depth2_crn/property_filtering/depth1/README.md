# How to reproduce this data

1. Run `yarp-init input.yaml`
    - STDOUT should look nearly identical to contents of `init.out`
    - You should see `3HP_depth1.pkl` and `STATUS.json` files generated in the same directory level as `input.yaml`
2. Run `yarp-progress .` from the same directory level as `3HP_depth1.pkl` and `STATUS.json`
    - STDOUT should look nearly identical to contents of `prog1.out`
    - You should see one job hit the queue which is using 8 CPUs
3. Once the job has completed, run `yarp-progress .` again from the same directory level as `3HP_depth1.pkl` and `STATUS.json`
    - Now the STDOUT should look nearly identical to the contents of `prog2.out`
4. Check the contents of the EGAT characterized reactions with `yarp-read -ia 3HP_depth1.pkl`
    - The STDOUT should look identical to the contents of `yarp-read.out`
