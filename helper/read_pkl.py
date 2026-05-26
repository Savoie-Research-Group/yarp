"""
This is a helper script to help visualize the contents of a YARP pickle file.

How to use:

python read_pkl.py yarp.pkl [optional: --visualize]
"""
import os
import argparse
import pickle
from tabulate import tabulate

def _format_optional_barrier(value):
    if value is None:
        return "none"
    return f"{value:.5}"

def main(args):
    file = args.filename
    rxns = pickle.load(open(file, 'rb')) # rxns is a dictionary object!
    print(f"Well folks, looks like we have {len(rxns)} reactions on our hands")

    # Convert dict_values to a list so we can sort it
    rxn_list = list(rxns.values())


##################################################################################
    # SHQK : To view the contents of the rxn objects:
    print("type of rxns = ", type(rxns))
    print("type of rxn_list = ", type(rxn_list))
    print("################################################################")
    print("Printing all Reaction Attributes Below for each rxn in rxn_list")
    print("################################################################")
    for i, rxn in enumerate(rxn_list):
#        print("################################################################")
#        print("Rxn Attributes for ", rxn.reactant_inchi, "to", rxn.product_inchi)
#        print("################################################################")
        for key, value in vars(rxn).items():    # SHQK : Returns the __dict__ atribute of an object. The __dict__ attribute is a dictionary containing the object's changeable attributes in key-value format.
            print(f"{key}: {value}")
    # SHQK : End
##################################################################################




    # 1. First pass: Find all unique barrier keys across all reactions
    unique_barrier_keys = set()
    for rxn in rxn_list:
        if rxn.barrier:
            unique_barrier_keys.update(rxn.barrier.keys())
            
    # Sort the keys alphabetically so the columns are always in a consistent order
    unique_barrier_keys = sorted(list(unique_barrier_keys))
    print ("unique_barrier_keys = ", unique_barrier_keys)

    # SHQK: Sort the data in increasing order of EGAT barrier
    # float('inf') ensures reactions without an 'egat' key go to the bottom
    rxn_list.sort(key=lambda r: r.barrier.get('egat_rgd1', float('inf')) if r.barrier else float('inf'))

    # 2. Dynamically build the headers
    headers = ['Reaction ID', 'Reactant', 'Product']
    for key in unique_barrier_keys:
        headers.append(f"{key} dG_activation")

    # 3. Second pass: Extract the data for the table
    data = []
    for rxn in rxn_list:
        # Start the row with the standard identifiers
        row = [
#            rxn.hash,
            rxn.id,    # SHQK: Changing this from hash to id because that's easier to track
            rxn.reactant.canon_smi,
            rxn.product.canon_smi
        ]
        
        # Dynamically pull the barriers for each unique key we found
        for key in unique_barrier_keys:
            fwd_barrier = rxn.barrier.get(key) if rxn.barrier else None
            
            row.append(_format_optional_barrier(fwd_barrier))
            
        data.append(row)
        
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
