import pickle
import os, sys

data=pickle.load(open(sys.argv[1], 'rb'))
for i in data:
    print(i.product_inchi)
    print(i.reactant_dft_opt)
    print(i.TS_dft)
