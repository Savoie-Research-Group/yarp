"""
This is a helper script to help visualize the contents of a YARP pickle file.

How to use:

python read_pkl.py yarp.pkl [optional: --visualize]
"""
import os
import argparse
import pickle
from tabulate import tabulate

def main(args):
    file = args.filename
    rxns = pickle.load(open(file, 'rb')) # rxns is a dictionary object!
    print(f"Well folks, looks like we have {len(rxns)} reactions on our hands")

    headers = ['Reaction ID', 'Reactant', 'Product', 'EGAT barrier', 'Reverse EGAT barrier']
    data = []
    for rxn in rxns.values():
        # access data for printing to screen via tabulate
        if 'egat' in rxn.barrier:
            data.append([rxn.id, rxn.reactant.canon_smi, rxn.product.canon_smi, f"{rxn.barrier['egat']:.5}", f"{rxn.reverse_barrier.get('egat'):.5}"])
        else:
            data.append([rxn.id, rxn.reactant.canon_smi, rxn.product.canon_smi, 'none'])
        
        # optionally, generate PDFs for each reactant/product pair
        if args.visualize:
            folder = f"visuals/{rxn.id}"
            os.makedirs(folder, exist_ok=True)
            rxn.reactant._graph.draw_bmats(outfile=f"{folder}/reactant.pdf")
            rxn.product._graph.draw_bmats(outfile=f"{folder}/product.pdf")
    
    print(tabulate(data, headers=headers, tablefmt='pretty'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Visualize YARP pickle file")
    parser.add_argument("filename", help="Path to the pickle file")
    parser.add_argument("--visualize", action="store_true", help="Visualize each reaction's reactant and product")

    args = parser.parse_args()
    main(args)
