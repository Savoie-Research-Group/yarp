import sys
import pickle

import yarp as yp
from yarp.network.network import network

def main(rxns):
    crn = network(rxns=rxns, dG_lot='egat')

    terminals = crn.get_terminal_species()
    print(f"There are {len(terminals)} terminal node species in this graph!")

    start_node = yp.yarpecule('O=CCCOO')
    end_node = yp.yarpecule('O=CO')

    paths = crn.get_simple_paths(start_node, end_node)
    print(f"There are {len(paths)} simple paths connecting O=CCCOO and O=CO")



if __name__ == "__main__":
    inp = sys.argv[1]
    with open(inp, "rb") as f:
        data = pickle.load(f)

    main(data)
