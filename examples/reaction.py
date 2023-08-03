#from yarp import yarpecule, draw_yarpecules, form_bonds, break_bonds
import yarp as yp
from copy import copy
import numpy as np


reactant="c1ccccc1"
a = yp.yarpecule("c1ccccc1")
mols = [ y for y in yp.form_bonds([a]) ]
yp.draw_yarpecules(mols,"benzene.pdf",label_ind=True,mol_labels=[ "score {}".format(_.bond_mat_scores[0]) for _ in mols ])

reactant="C=CC=C.C=C"
a = yp.yarpecule(reactant)
hashes = set([a.hash])
print(f"reactant: {reactant}")
yp.draw_yarpecules([a],"reactant.pdf",label_ind=True,mol_labels=[ "score {}".format(_.bond_mat_scores[0]) for _ in [a] ])

mols = list(set([ y for y in yp.form_bonds([a],hashes=hashes)]))
hashes.update([ _.hash for _ in mols ])
print(f"form 1 bond resulted in {len(mols)} new products")
yp.draw_yarpecules(mols,"f1.pdf",label_ind=True,mol_labels=[ "score {}".format(_.bond_mat_scores[0]) for _ in mols ])

mols = list(set([ y for y in yp.form_bonds(mols,hashes=hashes)]))
hashes.update([ _.hash for _ in mols ])
print(f"form 2 bonds resulted in {len(mols)} new products")
yp.draw_yarpecules(mols,"f2.pdf",label_ind=True,mol_labels=[ "score {}".format(_.bond_mat_scores[0]) for _ in mols ])

mols = list(set([ y for y in yp.break_bonds([a],n=1)]))
hashes.update([ _.hash for _ in mols ])
print(f"break 1 bond resulted in {len(mols)} new products")
yp.draw_yarpecules(mols,"b1.pdf",label_ind=True,mol_labels=[ "score {}".format(_.bond_mat_scores[0]) for _ in mols ])

mols = list(set([ y for y in yp.form_bonds(mols,hashes=hashes)]))
hashes.update([ _.hash for _ in mols ])
print(f"break 1 bond form 1 bond resulted in {len(mols)} products")
yp.draw_yarpecules(mols,"b1f1.pdf",label_ind=True,mol_labels=[ "score {}".format(_.bond_mat_scores[0]) for _ in mols ])

mols = list(set([ y for y in yp.break_bonds([a],n=2)]))
hashes.update([ _.hash for _ in mols ])            
print(f"break 2 bonds resulted in {len(mols)} products")
yp.draw_yarpecules(mols,"b2.pdf",label_ind=True,mol_labels=[ "score {}".format(_.bond_mat_scores[0]) for _ in mols ])

mols = list(set([ y for y in yp.form_bonds(mols,hashes=hashes)]))
hashes.update([ _.hash for _ in mols ])            
print(f"break 2 bonds form 1 bond resulted in {len(mols)} products")
yp.draw_yarpecules(mols,"b2f1.pdf",label_ind=True,mol_labels=[ "score {}".format(_.bond_mat_scores[0]) for _ in mols ])
mols = list(set([ y for y in yp.form_bonds(mols,hashes=hashes) ]))
print(f"break 2 bonds form 2 bond resulted in {len(mols)} products")
mols = list(set([ y for y in mols if y.bond_mat_scores[0] <= 0 ]))
print(f"break 2 bonds form 2 bond resulted in {len(mols)} products after filtering bad lewis_structures")

for i in range(len(mols)//50 + 1):
    yp.draw_yarpecules(mols[i*50:(i+1)*50],"b2f2_{}-{}.pdf".format(i*50,(i+1)*50-1),label_ind=True,mol_labels=[ "score {}".format(_.bond_mat_scores[0]) for _ in mols[i*50:(i+1)*50] ])
