# Welcome to an example YARP-again workflow!

Here we have done the following:
1. Generate a single round of enumerated products from a single starting SMILES molecule
2. Compute reaction barriers using an EGAT model trained on the RGD1 database
3. Filter out products with barriers above a user-defined kcal/mol threshold
4. Perform a second round of product enumeration, computing their reaction barriers again with EGAT
5. Visualize the resulting pickle file output containing YARP reaction objects
6. Analyze the chemical reaction network using graph theory tools

## `depth_1` folder - how to reproduce

```
cd depth_1/

python /path/to/yarp-again/yarp/main_yarp.py depth1.yaml > depth1.out 2> depth1.err

python /path/to/yarp-again/helper/read_pkl.py depth1.pkl > depth1_read_pkl.out
```

## `depth_2` folder - how to reproduce

```
cd depth_2/

python /path/to/yarp-again/yarp/main_yarp.py depth2.yaml > depth2.out 2> depth2.err

python /path/to/yarp-again/helper/read_pkl.py depth2.pkl > depth2_read_pkl.out

python analyze_network.py depth2.pkl > analyze_network.out
```
