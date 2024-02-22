import yarp as yp
import numpy as np
reactant=yp.yarpecule("n_hex.xyz")
print("bond-electron matrix")
for count_i, i in enumerate(reactant.elements):
    print(f"{i} {reactant.bond_mats[0][count_i]}")
break_mols=list(yp.break_bonds(reactant, n=2))
print(f"We have {len(break_mols)} reaction intermediates with breaking 2 bonds.")
import numpy as np
products=yp.form_bonds(break_mols)
products=[_ for _ in products if _.bond_mat_scores[0]<=0.0 and sum(np.abs(_.fc))<=2.0]
# Question: what's the purpose of this line?
print(f"We have {len(products)} products from n-hexane.")
yp.draw_yarpecules(products, "n-hexane.pdf", label_ind=True, mol_labels=[f"score {_.bond_mat_scores[0]}" for _ in products])

