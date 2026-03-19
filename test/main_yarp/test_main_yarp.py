import pytest
import pickle
import yaml
import sys
from pathlib import Path

from yarp.main_yarp import main


def test_enum_minimal(enum_min_options):
    input_dict = enum_min_options
    main(input_dict)

    output_str = input_dict['initialize']['output']
    output = Path(output_str)
    assert output.exists(), "Output pickle file was not created"

    with open(output_str, "rb") as f:
        saved_reactions = pickle.load(f)
    
    assert len(saved_reactions) == 11

def test_enum_full(enum_full_options):
    input_dict = enum_full_options
    main(input_dict)

    output_str = input_dict['initialize']['output']
    output = Path(output_str)
    assert output.exists(), "Output pickle file was not created"

    with open(output_str, "rb") as f:
        saved_reactions = pickle.load(f)

    assert len(saved_reactions) == 2

def test_egat_minimal(egat_min_options):
    input_dict = egat_min_options
    main(input_dict)

    output_str = input_dict['initialize']['output']
    output = Path(output_str)
    assert output.exists(), "Output pickle file was not created"

    with open(output_str, "rb") as f:
        saved_reactions = pickle.load(f)

    assert len(saved_reactions) == 2

    rxns = list(saved_reactions.values())
    assert rxns[0].barrier['egat'] == pytest.approx(95.54852294921875, rel=1e-5)
    assert rxns[1].barrier['egat'] == pytest.approx(73.41398620605469, rel=1e-5)

def test_d2_default(d2_default_enum, d1_path):
    input_dict = d2_default_enum
    input_dict['initialize']['initial species'] = d1_path
    main(input_dict)

    output_str = input_dict['initialize']['output']
    output = Path(output_str)
    assert output.exists(), "Output pickle file was not created"

    with open(output_str, "rb") as f:
        saved_reactions = pickle.load(f)

    assert len(saved_reactions) == 49

def test_d2_sep_prods(d2_sep_prods_enum, d1_path):
    input_dict = d2_sep_prods_enum
    input_dict['initialize']['initial species'] = d1_path
    main(input_dict)

    output_str = input_dict['initialize']['output']
    output = Path(output_str)
    assert output.exists(), "Output pickle file was not created"

    with open(output_str, "rb") as f:
        saved_reactions = pickle.load(f)

    assert len(saved_reactions) == 44

def test_d2_dg_filter(d2_dg_filter_enum, d1_path):
    input_dict = d2_dg_filter_enum
    input_dict['initialize']['initial species'] = d1_path
    main(input_dict)

    output_str = input_dict['initialize']['output']
    output = Path(output_str)
    assert output.exists(), "Output pickle file was not created"

    with open(output_str, "rb") as f:
        saved_reactions = pickle.load(f)

    assert len(saved_reactions) == 16 # keep all originally provided rxns, even if above dG barrier