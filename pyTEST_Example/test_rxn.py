import pytest, os, re, yaml
import shutil
import subprocess
import pandas as pd
import numpy as np
#import yarp as yp
#from calculator import add
def truthy(value):
    return bool(value)
def falsy(value):
    return not bool(value)

def check_metal(xyz):

    finish = False
    FeCO5 = yp.yarpecule(xyz)
    # first check adj_mat
    nBonds = 20
    nE = 58
    nDative= 5
    if(FeCO5.adj_mat.sum() == nBonds and FeCO5.bond_mats[0].sum() == nE):
        # then check bmat
        if(FeCO5.adj_mat.sum(axis=1)[0]==nDative):
            finish = True
    return finish

def form_bond(a, hashes, nform):
    mols = [a]
    for i in range(0, nform):
        mols = list(set([ y for y in yp.form_bonds(mols,hashes=hashes)]))
        hashes.update([ _.hash for _ in mols ])
        print(f"form {i} bond resulted in {len(mols)} new products")

def break_bond(a, hashes, nbreak):
    mols = [a]
    mols = list(set([ y for y in yp.break_bonds(mols,n=nbreak)]))
    hashes.update([ _.hash for _ in mols ])
    print(f"break {nbreak} bond resulted in {len(mols)} new products")

def rxn_setYAML(current_path, model_path, gsm_path, conda_path):
    if not os.path.isfile("template.yaml"): return
    shutil.copyfile("template.yaml", "parameters.yaml")
    if not os.path.isfile("parameters.yaml"): return

    with open('parameters.yaml', 'r') as file: filedata = file.read()
    # Replace the target string
    filedata = filedata.replace('{current_path}',  current_path)
    filedata = filedata.replace('{model_path}',    model_path)
    filedata = filedata.replace('{gsm_file_path}', gsm_path)
    filedata = filedata.replace('{conda_path}',    conda_path)
    with open('parameters.yaml', 'w') as file: file.write(filedata)

def rxn_xtb():
    subprocess.call("python main_xtb.py parameters.yaml", shell=True)
    subprocess.call("cat RESULT/IRC-record.txt", shell=True)

    df = pd.read_csv('RESULT/IRC-record.txt', delim_whitespace=True)
    # Check for 'intended' entry in 'type' column and if the barrier equals 6.747
    intended_row = df[df['type'] == 'intended']
    # Print result if intended row exists and barrier check
    barrier = 1000
    if not intended_row.empty:
        barrier = float(intended_row['barrier'].values[0])
    return barrier

def test_file():
    current_directory = os.getcwd() + '/'
    #subprocess.call(f"cd {current_directory}/pyTEST_Example/", shell=True)

    CONDA="CONDA_PATH" # will be replaced by a real path when running the github workflow #
    if CONDA=="CONDA_" + "PATH":
        # use "which crest" to find the path #
        STR = os.popen('which crest').read().rstrip()
        CONDA = STR.split("/bin/crest")[0]

    rxn_setYAML(current_path = current_directory, 
            model_path = f"{current_directory}/bin",
            gsm_path   = f"{current_directory}/bin/inpfileq",
            conda_path = f"{CONDA}/bin")

    # run YARP-xtb for DA example #
    barrier = rxn_xtb()
    assert(np.abs(barrier - 6.747132) < 0.01)
    print(f"YARP-xtb CHECK FINISHED\n")

    '''
    # run Organometallics YARP for Fe(CO)5 #
    subprocess.call(f"cd {current_directory}/examples/", shell=True)
    assert  os.path.exists('FeCO5.xyz')
    assert  check_metal("FeCO5.xyz")
    print("Organometallics CHECK FINISHED\n")
    reactant="C=CC=C.C=C"
    a = yp.yarpecule(reactant)
    hashes = set([a.hash])
    print(f"reactant: {reactant}")
    form_bond(a, hashes, 2)
    break_bond(a, hashes, 2)
    assert len(hashes) == 29
    '''

