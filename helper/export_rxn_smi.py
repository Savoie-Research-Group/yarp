"""
This is a helper script to extract atom-mapped SMILES from a YARP pickle file.
SMILES for each reaction will be written to a CSV file.

How to use:

python export_rxn_smi.py yarp.pkl output.csv
"""

import argparse
import pickle
import pandas as pd

def main(args):
    print("So I've heard you'd like some SMILES strings...")

    # Load in the pickle file from YARP
    file = args.filename
    rxns = pickle.load(open(file, 'rb')) # rxns is a dictionary object!

    # Stuff data into a Pandas dataframe for easy export to CSV later
    df = pd.DataFrame(columns=['rxn_id', 'reactant_smi', 'product_smi'])
    for idx, rxn in enumerate(rxns.values()):
        df.loc[idx] = [rxn.id, rxn.reactant.map_smi, rxn.product.map_smi]
    
    df.to_csv(args.output, index=False)
    print(f"...and now you can find them in {args.output}!")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Extract mapped SMILES from YARP pickle file")
    parser.add_argument("filename", help="Path to the pickle file")
    parser.add_argument("output", help="Path to the CSV file with SMILES strings")

    args = parser.parse_args()
    main(args)