import pytest
from yarp.reaction.ml_barrier import get_egat_barriers, get_egat_barries_from_csv
import pandas as pd

class TestEgat:
    def test_get_egat_barriers(self, glucose_single_path, egat_pretrain):
        model, args = egat_pretrain
        glucose_single_path = get_egat_barriers(glucose_single_path, model)
        for rxn in glucose_single_path.values():
            assert rxn.barrier['egat'] is not None
    def test_get_egat_barries_from_csv(self, egat_csv, egat_pretrain):
        model, args = egat_pretrain
        df = get_egat_barries_from_csv(egat_csv, model)
        assert df.loc[0, 'egat_barrier'] is not None