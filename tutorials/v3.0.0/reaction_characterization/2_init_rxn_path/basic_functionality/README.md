# How to reproduce this data

1. Run `yarp-init input.yaml`
    - STDOUT should look nearly identical to contents of `init.out`
    - You should see `HAA_depth1.pkl` and `STATUS.json` files generated in the same directory level as `input.yaml`
2. Run `yarp-loop -w . -i 1 -d 60` from the same directory as `HAA_depth1.pkl` and `STATUS.json`
    - To run loop in background instead, run `nohup yarp-loop -w . -i 1 -d 60 > loop.out &`
    - An output `yarp_loop.out` file should look nearly identical to `eg_yarp_loop.out`
3. Check the contents of the EGAT characterized reactions with `yarp-read -ia HAA_depth1.pkl`
    - STDOUT should look identical to `yarp_read.out`
