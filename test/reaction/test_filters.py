import pytest
from yarp.reaction.filters import apply_target_blinders
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


