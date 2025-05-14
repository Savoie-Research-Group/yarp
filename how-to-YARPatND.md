# Installing YARP on the CRC at Notre Dame

1. Get yourself some YARP code from GitHub
```
cd /path/to/some/repo/
git clone https://github.com/Savoie-Research-Group/yarp.git
cd yarp
```

2. Create a `classy-yarp.yaml` file in the `yarp` repo with the following contents
```
name: classy-yarp
channels:
 - defaults
 - conda-forge
dependencies:
 - defaults::python=3.9
 - defaults::pandas
 - defaults::xgboost
 - defaults::pip
 - conda-forge::xtb
 - conda-forge::openbabel
 - conda-forge::crest
 - conda-forge::ase
 - pip:
   - pysisyphus
   - rdkit
   - scikit-learn==1.3.2
```

3. Set up `conda` and create a YARP environment
```
# Initial installation (see CRC docs for more info https://docs.crc.nd.edu/popular_modules/conda.html)

module load conda
conda init
source ~/.bashrc
module unload conda

# Create YARP environment and then activate it

conda env create -f classy-yarp.yaml
conda activate classy-yarp
```

4. Install YARP
```
# From the `yarp` repo
pip install .
```

5. Make xTB work properly via pysis
```
# Create a file in your user home repo

vi ~/.pysisyphusrc
```

Fill in the `.pysisyphusrc` file with the following contents
```
[xtb]
cmd=xtb
```

6. Run some tests to make sure everything is working

First, install pytest on the `classy-yarp` environment
```
pip install pytest
```

The super quick (few seconds) test:
```
cd /path/to/yarp/examples/
pytest -s
```

The longer (about 10 minutes) test:
```
cd /path/to/yarp/pyTEST_Example
pytest -s
```