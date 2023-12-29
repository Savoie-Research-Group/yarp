import pickle
import os, sys

data=pickle.load(open(sys.argv[1], 'rb'))
for i in data:
    print(i.constrained_TS)
