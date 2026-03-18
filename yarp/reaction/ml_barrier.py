"""
Placeholder for code allowing for ML predicted reaction barriers (and other reaction properties?)
"""
import omegaconf
import os
import pandas as pd

from yarp.reaction.egat.predict_from_smiles import load_model, predict_activation_energy
from yarp.reaction.egat.dataset import FastDataset

def get_egat_barriers(yp_rxns, model, args, verbose=False):
    """
    yp_rxns : dict
        Dictionary of reaction class objects (values) stored by reaction hash (key)

    model : ???
        Loaded pytorch model
    """

    rxn_list = list(yp_rxns.values())
    dataframe = []
    for rxn in rxn_list:
        rsmiles = rxn.reactant.map_smi
        psmiles = rxn.product.map_smi
        reaction_smiles = f"{rsmiles}>>{psmiles}"
        dataframe.append(reaction_smiles)
    dataframe = pd.DataFrame(dataframe, columns=['AAM'])
    if verbose:
        print("Dataframe generated from reaction objects")
        print(dataframe)
    os.makedirs('tmp', exist_ok=True)
    csv_path = os.path.join('tmp', 'egat_barriers.csv')
    dataframe.to_csv(csv_path, index=False)
    test_dataset = FastDataset(args, dataset=csv_path)
    try:
        for data_idx in range(len(test_dataset)):
            datapoint = test_dataset[data_idx]
            if datapoint is None:
                dataframe.loc[data_idx, 'egat_barrier'] = None
                print(f"Error building datapoint for {data_idx}")
                if data_idx < len(dataframe):
                    print(f"  AAM: {dataframe.loc[data_idx, 'AAM']}")
                continue
            else:
                dp_idx, rgraph, pgraph, strings = datapoint
                if verbose:
                    print(rgraph)
                try:
                    prediction = predict_activation_energy(model, rgraph, pgraph)
                    dataframe.loc[dp_idx, 'egat_barrier'] = prediction
                except Exception as e:
                    print(f"Error predicting barrier for {strings}: {e}")
                    dataframe.loc[dp_idx, 'egat_barrier'] = None
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)
        fail_path = os.path.join('tmp', 'fail.txt')
        exclude_path = os.path.join('tmp', 'exclude.txt')
        if os.path.exists(fail_path):
            print(f"EGAT datapoint failures logged to {fail_path}")
        if os.path.exists(exclude_path):
            print(f"EGAT datapoint exclusions logged to {exclude_path}")

    # Update reaction objects with EGAT barriers
    for rxn_idx, rxn in enumerate(rxn_list):
        rxn.barrier['egat'] = dataframe.loc[rxn_idx, 'egat_barrier']
    return yp_rxns

def get_egat_barries_from_csv(csv_path, model, args, verbose=False):
    """
    csv_path : str
        Path to CSV file with reaction SMILES
    model : ???
        Loaded pytorch model
    """
    df = pd.read_csv(csv_path)
    test_dataset = FastDataset(args, dataset=csv_path)
    for idx in range(len(test_dataset)):
        datapoint = test_dataset[idx]
        if datapoint is None:
            print(f"Error building datapoint for {idx}")
            continue
        else:
            idx, rgraph, pgraph, strings = datapoint
            if verbose:
                print(rgraph)
            try:
                prediction = predict_activation_energy(model, rgraph, pgraph)
                df.loc[idx, 'egat_barrier'] = prediction
            except Exception as e:
                print(f"Error predicting barrier for {strings}: {e}")
                df.loc[idx, 'egat_barrier'] = None
    return df
