# How to reproduce this data

1. Run `yarp-init input.yaml`
    - STDOUT should look nearly identical to contents of `init.out`
    - You should see `HAA_depth1.pkl` and `STATUS.json` files generated in the same directory level as `input.yaml`
    - You may need to play around with the mem_per_cpu requirements of this module
2. Run `nohup yarp-loop -w . -i 1 -d 1000 > loop.out &` from the same directory as `HAA_depth1.pkl` and `STATUS.json`
3. Check the contents of the EGAT characterized reactions with `yarp-read -ia HAA_depth1.pkl`
