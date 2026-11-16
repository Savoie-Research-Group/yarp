"""
Testing suite for the pre-enumeration filters in yarp/reaction/filters.py
"""
from copy import deepcopy

from yarp.reaction.filters import filter_enum_candidates
from yarp.util.config import ProductBlindersConfig


def test_filter_enum_candidates_leaves_input_rxns_unmodified(khp2pp22_soergel_beam2_cyc3):
    """
    Product blinders selects enumeration candidates without writing anything
    back onto the input reactions. (It used to set
    network_meta["prod_blind_selected"] on the reactions that produced a
    selected candidate; that tracking was removed.)
    """
    # Key by str(rxn.hash), as generate_rxns does before calling the filter.
    rxns = {str(rxn.hash): rxn for rxn in deepcopy(khp2pp22_soergel_beam2_cyc3).values()}
    for rxn in rxns.values():
        rxn.network_meta = dict()  # the fixture carries flags from older code

    netconfig = ProductBlindersConfig(target_product="OC1CCOO1", distance_metric="soergel",
                                      mode="beam", n_nodes=2, tie_window=0.0)
    candidates = filter_enum_candidates(rxns, separate_prods=True, netconfig=netconfig)

    assert len(candidates) >= 2
    assert all(rxn.network_meta == dict() for rxn in rxns.values())
