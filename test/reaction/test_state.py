"""
Testing suite for the state class
"""

from yarp.reaction.state import state
from yarp.yarpecule.yarpecule import yarpecule as ypcule

class TestInitialization:

    def test_bimolecular(self):

        graph = ypcule('C=C.O')
        st = state(graph)

        assert len(st.species) == 2
        assert len(st.conc) == 2