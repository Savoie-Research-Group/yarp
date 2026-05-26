"""
Placeholder for code allowing for ML predicted reaction barriers (and other reaction properties?)
"""
import os
import pandas as pd
from rdkit import Chem


def _normalize_reaction_smiles_for_egat(rsmiles, psmiles):
    """
    Return temporary EGAT-only mapped smiles with dense 0..N-1 atom-map labels.

    The original YARP reaction objects remain untouched. This exists because
    EGAT feature generation assumes dense positional map labels on both sides
    of the reaction.
    """
    return normalize_reaction_smiles_for_egat(rsmiles, psmiles)


def get_egat_barriers(yp_rxns, model, args, verbose=False):
    """
    yp_rxns : dict
        Dictionary of reaction class objects (values) stored by reaction hash (key)

    model : ???
        Loaded pytorch model
    """
    from yarp.reaction.egat.predict_from_smiles import predict_activation_energy
    from yarp.reaction.egat.dataset import FastDataset

    rxn_list = list(yp_rxns.values())
    dataframe = []
    for rxn in rxn_list:
        rsmiles, psmiles = _normalize_reaction_smiles_for_egat(rxn.reactant.map_smi, rxn.product.map_smi)
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
        tmp_dir = 'tmp'
        fail_path = os.path.join(tmp_dir, 'fail.txt')
        exclude_path = os.path.join(tmp_dir, 'exclude.txt')
        if os.path.exists(fail_path):
            print(f"EGAT datapoint failures logged to {fail_path}")
        if os.path.exists(exclude_path):
            print(f"EGAT datapoint exclusions logged to {exclude_path}")
        if os.path.isdir(tmp_dir) and not os.path.exists(fail_path) and not os.path.exists(exclude_path):
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass

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
    from yarp.reaction.egat.predict_from_smiles import predict_activation_energy
    from yarp.reaction.egat.dataset import FastDataset

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



def _explicit_atom_maps(mol):
    maps = []
    missing = []
    for atom in mol.GetAtoms():
        if not atom.HasProp("molAtomMapNumber"):
            missing.append(atom.GetIdx())
            continue
        maps.append(int(atom.GetProp("molAtomMapNumber")))
    if missing:
        raise ValueError(f"Atoms missing explicit atom maps: {missing}")
    return maps


def normalize_reaction_smiles_for_egat(rsmiles, psmiles):
    """
    Return temporary EGAT-only mapped SMILES with dense 0..N-1 atom-map labels.

    The original YARP reaction objects remain untouched. This exists because
    EGAT feature generation assumes dense positional map labels on both sides
    of the reaction.
    """
    rmol = Chem.MolFromSmiles(rsmiles, sanitize=False)
    pmol = Chem.MolFromSmiles(psmiles, sanitize=False)
    if rmol is None or pmol is None:
        raise ValueError("Could not parse mapped reaction smiles for EGAT normalization.")

    r_maps = _explicit_atom_maps(rmol)
    p_maps = _explicit_atom_maps(pmol)

    if len(set(r_maps)) != len(r_maps):
        dupes = sorted({m for m in r_maps if r_maps.count(m) > 1})
        raise ValueError(f"Duplicate reactant atom maps encountered during EGAT normalization: {dupes}")
    if len(set(p_maps)) != len(p_maps):
        dupes = sorted({m for m in p_maps if p_maps.count(m) > 1})
        raise ValueError(f"Duplicate product atom maps encountered during EGAT normalization: {dupes}")

    if set(r_maps) != set(p_maps):
        raise ValueError(
            "Reactant/product atom-map sets differ during EGAT normalization. "
            f"Reactant-only: {sorted(set(r_maps) - set(p_maps))}; "
            f"Product-only: {sorted(set(p_maps) - set(r_maps))}"
        )

    old_to_new = {
        old_map: new_map
        for new_map, old_map in enumerate(sorted(set(r_maps)))
    }

    for mol in (rmol, pmol):
        for atom in mol.GetAtoms():
            atom.SetProp("molAtomMapNumber", str(old_to_new[int(atom.GetProp("molAtomMapNumber"))]))

    return Chem.MolToSmiles(rmol), Chem.MolToSmiles(pmol)


def dense_reaction_smiles_for_egat(rsmiles, psmiles):
    rsmiles, psmiles = normalize_reaction_smiles_for_egat(rsmiles, psmiles)
    return f"{rsmiles}>>{psmiles}"
