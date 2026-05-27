# How to reproduce this data

1. Run `yarp-init input.yaml`
    - STDOUT should look nearly identical to contents of `init.out`
    - You should see `HAA_depth1.pkl` and `STATUS.json` files generated in the same directory level as `input.yaml`
2. Run `yarp-progress .` from the same directory level as `HAA_depth1.pkl` and `STATUS.json`
    - STDOUT should look nearly identical to contents of `prog1.out`
    - You should see one job hit the queue which is using 8 CPUs
3. Once the job has completed, run `yarp-progress .` again from the same directory level as `HAA_depth1.pkl` and `STATUS.json`
    - Now the STDOUT should look nearly identical to the contents of `prog2.out`
    - You should see 3 jobs hit the queue, each using 4 CPUs. These are the CREST conformer jobs, and should take a few moments to finish.
    - If you check the `HAA_depth1.pkl`, you should see EGAT characterization results displayed
4. After the 3 conformer jobs finish, run `yarp-progress .` again
    - Now the STDOUT should look nearly identical to the contents of `prog3.out`
