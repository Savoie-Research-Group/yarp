# How to generate data

1. Perform product enumeration and initialize reaction objects/status tracker:
    ```
    cd /path/to/depth1
    yarp-init depth1.yaml > init.out
    ```
2. Activate YARP loop script to progress individual tasks forward in the pipeline:
    ```
    nohup yarp-loop -w . -i 1 -d 2 > loop.out &
    ```
3. After the analysis of `depth1` has fully completed, move to the `depth2` directory, and repeat steps 1 and 2
4. You can quickly check the contents of a YARP reactions pickle file using the `helper/read_pkl.py` script!
    ```
    python /path/to/helper/read_pkl.py yarp.pkl
    ```