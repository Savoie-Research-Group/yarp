import yarp as yp
from yarp.util.rdkit import yarpecule_to_rdmol, geom_from_rdmol
import numpy as np

class TestGeom:
    def test_preserved_geom(self):
        fran_smi = '[c:0]12[c:1]([H:7])[c:2]([H:8])[c:3]([H:9])[c:4]([H:10])[c:5]1[C-:12]([O-:11])[N:13]([H:6])[S+2:14]2'
        fran = yp.yarpecule(fran_smi)
        mol = yarpecule_to_rdmol(fran.elements, fran.adj_mat, fran.bond_mats[0], fran._atom_info, fran.geo)

        geo = geom_from_rdmol(mol)

        diff = fran.geo - geo
        assert np.all(diff == 0)

