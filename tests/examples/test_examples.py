# test_calculator.py
import pytest, os, re
import yarp as yp
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

def test_file():
    #current_directory = os.getcwd() + '/'
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
    #print(f"hashes: {len(hashes)}\n")
