# How to reproduce this data

1. Run `yarp-init input.yaml`
    - STDOUT should look nearly identical to contents of `init.out`
    - You should see `HAA_depth1.pkl` and `STATUS.json` files generated in the same directory level as `input.yaml`
2. Run `nohup yarp-loop -w . -i 1 -d 16 > loop.out &` from the same directory level as `HAA_depth1.pkl` and `STATUS.json`
    - Takes 5 minutes to finish
3. Check the contents of the EGAT characterized reactions with `yarp-read -ia HAA_depth1.pkl`
