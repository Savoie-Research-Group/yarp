# How to reproduce this data

1. Run `yarp-init input.yaml`
    - STDOUT should look nearly identical to contents of `init.out`
    - You should see `HAA_depth2.pkl` and `STATUS.json` files generated in the same directory level as `input.yaml`
2. Run `yarp-progress .` from the same directory level as `HAA_depth2.pkl` and `STATUS.json`
    - You should see one job hit the queue which is using 8 CPUs
3. Once the job has completed, run `yarp-progress .` again from the same directory level as `HAA_depth2.pkl` and `STATUS.json`
4. Check the contents of the separated products with `yarp-read -i HAA_depth2.pkl`
    - The STDOUT should look identical to the contents of `yarp-read.out`
