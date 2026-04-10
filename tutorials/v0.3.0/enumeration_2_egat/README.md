# How to generate data

1. Run the first round of product enumeration --> EGAT characterization

Perform product enumeration and initialize reaction objects/status tracker:
```
cd /path/to/depth1
python ../../../../yarp/initialize_yarp.py input.yaml > init.out 2> init.err
```
You should see a `STATUS.json` and a `depth1.pkl` file be generated.
Two reaction objects are created.

Then, making sure your Docker desktop is open and running...
```
python ../../../../yarp/progress_yarp.py . > prog1.out
```
You should see a `SCRATCH` directory be generated, along with a Docker container launching.

The EGAT container should finish running in a couple of seconds.
Once it's done, run `progress_yarp.py` once more to actually
scrape the EGAT data output into the `depth1.pkl` reaction objects.
```
python ../../../../yarp/progress_yarp.py . > prog2.out
```

2. Run the second round of product enumeration --> EGAT characterization

Change directories and initialize new reaction objects
```
cd ../depth2
python ../../../../yarp/initialize_yarp.py input.yaml > init.out 2> init.err
```

Then launch the EGAT container:
```
python ../../../../yarp/progress_yarp.py . > prog1.out
```

And when the container is done running (wait about 10 seconds),
scrape the data into the `depth2.pkl` reaction objects.
```
python ../../../../yarp/progress_yarp.py . > prog2.out
```