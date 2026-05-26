import pytest
from types import SimpleNamespace
from yarp.reaction.filters import apply_target_blinders
from yarp.reaction.filters import filter_enum_candidates
from yarp.reaction.reaction import reaction
from yarp.yarpecule.yarpecule import yarpecule

@pytest.fixture
def target_yp():
    mol = yarpecule('O=COO')
    mol.get_smiles()
    return mol

# class TestCappedBlinderMode:

    # def test_aggressive_cap(self, khp_d1, target_yp):
    #     candidates = apply_target_blinders(
    #         raw_rxns=khp_d1, target_yp=target_yp,
    #         dist='soergel', mode='capped', cap='aggressive'
    #     )

    #     assert len(candidates) == 2
    
    # def test_moderate_cap(self, khp_d1, target_yp):
    #     candidates = apply_target_blinders(
    #         raw_rxns=khp_d1, target_yp=target_yp,
    #         dist='soergel', mode='capped', cap='moderate'
    #     )

    #     assert len(candidates) == 3


def test_separated_candidate_filter_accepts_missing_reactive_maps():
    reactant = yarpecule('CC')
    product = yarpecule('[CH3].[CH3]')
    rxn = reaction(reactant, product)

    candidates = filter_enum_candidates(
        {rxn.hash: rxn},
        separate_prods='all',
        netconfig=SimpleNamespace(target_product=None),
        react_atoms=[set([999])],
        verbose=False,
    )

    assert len(candidates) == 1
