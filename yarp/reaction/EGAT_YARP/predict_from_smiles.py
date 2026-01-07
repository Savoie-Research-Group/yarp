#!/usr/bin/env python3
"""
Predict activation energies from SMILES using the EGAT model
This script processes SMILES strings and uses the model for predictions
Uses the same approach as dataset.py for graph generation
"""

import torch
import dgl
import pandas as pd
import numpy as np
import argparse
import os
from types import SimpleNamespace
try:
    from model import EGAT_Rxn
except ImportError:
    from yarp.reaction.EGAT_YARP.model import EGAT_Rxn
try:
    from molecule import Molecule, OLD_BOND_ENCODE, bond_encode
except ImportError:
    from yarp.reaction.EGAT_YARP.molecule import Molecule, OLD_BOND_ENCODE, bond_encode
try:
    from graphgenhelperfunctions import return_reactive
except ImportError:
    from yarp.reaction.EGAT_YARP.graphgenhelperfunctions import return_reactive
try:
    from RDKit.RDKitHelpers import RemoveMapping
except ImportError:
    from yarp.reaction.EGAT_YARP.RDKit.RDKitHelpers import RemoveMapping
from rdkit import Chem
import omegaconf
try:
    from dataset import FastDataset
except ImportError:
    from yarp.reaction.EGAT_YARP.dataset import FastDataset
from torch.utils.data import DataLoader



def predict_activation_energy(model, graphR, graphP):
    """
    Predict activation energy for a reaction
    
    Args:
        model: EGAT_Rxn model
        graphR: DGL graph of reactant
        graphP: DGL graph of product
        
    Returns:
        float: Predicted activation energy, or None if graph conversion failed
    """
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        prediction = model(graphR, graphP)
        return prediction.item()




def load_model(checkpoint_path, config, verbose=False):
    """
    Load EGAT_Rxn model from checkpoint file
    
    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        config: Config object
        
    Returns:
        tuple: (model, args) where args is compatible with dataset.py
    """
    if verbose:
        print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Infer config from state dict (same logic as compile_model.py)
    egat1_fc_nodes_weight = state_dict['egat1.fc_nodes.weight']
    egat1_fc_attn_weight = state_dict['egat1.fc_attn.weight']
    num_node_feats = state_dict['egat1.fc_nodes.weight'].shape[1]
    
    # hidden_dim * num_heads = output size of fc_nodes
    hidden_dim_times_heads = egat1_fc_nodes_weight.shape[0]
    num_heads = egat1_fc_attn_weight.shape[0]
    hidden_dim = hidden_dim_times_heads // num_heads
    
    # Infer input dimensions
    num_node_feats = egat1_fc_nodes_weight.shape[1]
    num_edge_feats = state_dict['egat1.fc_edges.weight'].shape[1] - 2 * num_node_feats
    config.num_node_feats = num_node_feats
    config.num_edge_feats = num_edge_feats
    config.hidden_dim = hidden_dim
    config.num_heads = num_heads
    # Create model config
    model_config = config
    
    if verbose:
        print(f"Model config: num_node_feats={num_node_feats}, num_edge_feats={num_edge_feats}, "
            f"hidden_dim={hidden_dim}, num_heads={num_heads}")
    
    # Create model
    model = EGAT_Rxn(model_config)
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create args for graph generation (compatible with dataset.py)
    args = config
    
    if verbose:
        print(f"Model loaded successfully!")
    return model, args


def process_csv_file(csv_path:str, model_path:str, config_path:str, output_path:str=None):
    """
    Process a CSV file with reaction SMILES and predict activation energies
    
    Args:
        csv_path: Path to input CSV file
        model_path: Path to model checkpoint (.pth file)
        output_path: Path to output CSV file (optional)
    """
    config = omegaconf.OmegaConf.load(config_path)
    
    model, args = load_model(model_path, config)

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Found {len(df)} reactions")
    print(f"Columns: {list(df.columns)}")
    
    # Check if we have the required columns
    if 'AAM' in df.columns:
        # Original format: single AAM column with reaction SMILES
        use_aam_format = True
    elif 'Rsmiles' in df.columns and 'Psmiles' in df.columns:
        # New format: separate Rsmiles and Psmiles columns
        use_aam_format = False
    else:
        raise ValueError("CSV file must have either 'AAM' column or both 'Rsmiles' and 'Psmiles' columns")
    
    # Add prediction column
    predictions = []
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            print(f"Processing reaction {idx+1}/{len(df)}")
        
        try:
            if use_aam_format:
                # Parse reaction SMILES from AAM column
                reaction_smiles = row['AAM']
                if '>>' in reaction_smiles:
                    parts = reaction_smiles.split('>>')
                    if len(parts) == 2:
                        reactant_smiles = parts[0].strip()
                        product_smiles = parts[1].strip()
                    else:
                        print(f"Invalid reaction SMILES format for reaction {idx+1}: {reaction_smiles}")
                        predictions.append(np.nan)
                        continue
                else:
                    print(f"Reaction SMILES missing '>>' separator for reaction {idx+1}: {reaction_smiles}")
                    predictions.append(np.nan)
                    continue
            else:
                # Use separate Rsmiles and Psmiles columns
                reactant_smiles = row['Rsmiles']
                product_smiles = row['Psmiles']
            
            # Make prediction
            pred_activation = predict_activation_energy(model, reactant_smiles, product_smiles, args)
            if pred_activation is None:
                if use_aam_format:
                    print(f"Failed to predict for reaction {idx+1}: {reaction_smiles}")
                else:
                    print(f"Failed to predict for reaction {idx+1}: {reactant_smiles} >> {product_smiles}")
                predictions.append(np.nan)
            else:
                predictions.append(pred_activation)
            
        except Exception as e:
            print(f"Error processing reaction {idx+1}: {e}")
            predictions.append(np.nan)
    
    # Add predictions to dataframe
    df['Activation_PRED'] = predictions
    
    # Save results
    if output_path is None:
        output_path = csv_path.replace('.csv', '_predictions.csv')
    
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    
    # Print statistics
    valid_predictions = [p for p in predictions if not np.isnan(p)]
    if valid_predictions:
        print(f"\nPrediction Statistics:")
        print(f"Valid predictions: {len(valid_predictions)}/{len(predictions)}")
        print(f"Mean prediction: {np.mean(valid_predictions):.2f}")
        print(f"Std prediction: {np.std(valid_predictions):.2f}")
        print(f"Min prediction: {np.min(valid_predictions):.2f}")
        print(f"Max prediction: {np.max(valid_predictions):.2f}")
        
        # Compare with actual values if available
        if 'Activation' in df.columns:
            actual_values = df['Activation'].dropna()
            if len(actual_values) > 0:
                print(f"\nComparison with actual values:")
                print(f"Mean actual: {np.mean(actual_values):.2f}")
                print(f"Std actual: {np.std(actual_values):.2f}")
                
                # Calculate MAE for valid predictions
                valid_indices = [i for i, p in enumerate(predictions) if not np.isnan(p)]
                if valid_indices:
                    mae = np.mean([abs(predictions[i] - df.iloc[i]['Activation']) 
                                 for i in valid_indices if not np.isnan(df.iloc[i]['Activation'])])
                    print(f"MAE: {mae:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Predict activation energies from SMILES using compiled EGAT model')
    parser.add_argument('--config', help='Path to config file' , required=True)
    parser.add_argument('--csv', help='Path to CSV file with reaction SMILES' , required=True)
    parser.add_argument('--model', help='Path to model checkpoint file (.pth)' , required=True)
    parser.add_argument('--output', help='Path to output CSV file' , required=True)
    
    args = parser.parse_args()
    config = omegaconf.OmegaConf.load(args.config)
    omegaconf.OmegaConf.set_struct(config, False)

    # Test dataset 
    test_dataset = FastDataset(config, dataset=args.csv)
    model, _ = load_model(args.model, config)
    for idx in range(len(test_dataset)):
        idx, rgraph, pgraph, strings = test_dataset[idx]
        print('idx:', idx)
        print('rgraph:', rgraph)
        print('pgraph:', pgraph)
        print('strings:', strings)

        prediction = predict_activation_energy(model, rgraph, pgraph)
        print('prediction:', prediction)


if __name__ == "__main__":
    exit(main())
