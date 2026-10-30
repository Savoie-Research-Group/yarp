"""
Tests for yarp/reaction/generate_rxns.py
"""
import copy
import pytest
from yarp.util.input import InputParser
from yarp.reaction.generate_rxns import generate_rxns


# enum_min_options starts from a species, enum_d2_pkl from a reaction pickle;
# generate_rxns calls enumerate_products separately in each mode.
@pytest.mark.parametrize("inp_fixture", ["enum_min_options", "enum_d2_pkl"])
@pytest.mark.parametrize("exhaustive", [False, True])
def test_exhaustive_reaches_enumerate_products(inp_fixture, exhaustive, request, monkeypatch):
    cfg = copy.deepcopy(request.getfixturevalue(inp_fixture))
    cfg["initialize"]["enumeration"]["exhaustive"] = exhaustive
    calls = []
    monkeypatch.setattr("yarp.reaction.generate_rxns.enumerate_products",
                        lambda r_yp, **kwargs: calls.append(kwargs) or [])
    generate_rxns(InputParser(cfg))
    assert calls
    assert all(kwargs["exhaustive"] is exhaustive for kwargs in calls)
