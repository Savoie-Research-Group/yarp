import yarp as yp
import numpy as np
reactant=yp.yarpecule("test.xyz")

print("running enumeration")
yp.draw_bmats(reactant,"reactant.pdf")

reactant=yp.yarpecule("test2.xyz")

print("running enumeration")
yp.draw_bmats(reactant,"reactant2.pdf")
exit()
break_mols=list(yp.break_bonds(reactant, n=2))
print(f"We have {len(break_mols)} reaction intermediates with breaking 2 bonds.")
import numpy as np
products=yp.form_n_bonds(break_mols, n=2)
print(f"Raw products: {len(products)}")
products=[_ for _ in products if _.bond_mat_scores[0]<=5.0]
# Question: what's the purpose of this line?
print(f"We have {len(products)} products")
yp.draw_yarpecules(products, "cyclopetene_H2.pdf", label_ind=True, mol_labels=[f"score {_.bond_mat_scores[0]}" for _ in products])

