import pytest
from yarp.reaction.ml_barrier import get_egat_barriers, get_egat_barries_from_csv
import pandas as pd
import numpy as np
import yarp.reaction.ml_barrier as ml_barrier_module
class TestEgat:
    def test_get_egat_barriers(self, glucose_single_path, egat_pretrain):
        model, args = egat_pretrain
        glucose_single_path = get_egat_barriers(glucose_single_path, model)
        for rxn in glucose_single_path.values():
            assert rxn.barrier['egat'] is not None
    
    def test_get_egat_barries_from_csv(self, egat_csv, egat_pretrain):
        """Test that predictions match Activation_PRED from CSV file."""
        model, args = egat_pretrain
        
        # Set args globally for the function (since it uses global args)
        
        ml_barrier_module.args = args
        
        # Load the CSV file to get expected predictions
        expected_df = pd.read_csv(egat_csv)
        
        # Get predictions from the model
        df = get_egat_barries_from_csv(egat_csv, model)
        
        # Compare predictions with Activation_PRED

        
        predicted_values = df['egat_barrier'].values
        expected_values = expected_df['Activation_PRED'].values
        
        # Check that we have predictions
        assert len(predicted_values) > 0, "No valid predictions were generated"
        
        # Compare predictions (using numpy for element-wise comparison)
        # Allow tolerance for floating point comparisons (0.0001 kcal/mol)
        tolerance = 0.0001
        differences = np.abs(predicted_values - expected_values)
        
        # Check that all predictions are within tolerance
        assert np.all(differences < tolerance), (
            f"Predictions do not match Activation_PRED within tolerance ({tolerance} kcal/mol). "
            f"Max difference: {differences.max():.6f} kcal/mol, "
            f"Mean difference: {differences.mean():.6f} kcal/mol. "
            f"Number of mismatches: {np.sum(differences >= tolerance)}/{len(differences)}"
        )
        
