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

class TestStateIdentity:
    """
    `state.identity` is what progress_yarp keys conformer pooling and the
    redundancy blocker on. It must separate two atom mappings of the same
    molecule, because conformer geometries are index-ordered arrays and a
    state's paired_bem, adj_mat and bond_changes are indexed to its own
    mapping. `state.hash` alone does not separate them.
    """

    def test_hash_alone_does_not_separate_mappings(self, khp_remapped_products):
        """Documents the reason identity exists, so the risk stays visible."""
        for smi, mols in khp_remapped_products.items():
            states = [state(m) for m in mols]
            assert len({s.hash for s in states}) == 1, (
                f"{smi} mappings no longer share a state hash"
            )

    def test_identity_separates_mappings(self, khp_remapped_products):
        for smi, mols in khp_remapped_products.items():
            states = [state(m) for m in mols]
            identities = [s.identity for s in states]
            assert len(set(identities)) == len(states), (
                f"state.identity merged {len(states)} atom mappings of {smi}"
            )

    def test_identity_matches_for_the_same_mapping(self, khp_products):
        for smi, product in khp_products.items():
            assert state(product).identity == state(product).identity, (
                f"state.identity was not stable for {smi}"
            )

    def test_pooling_by_identity_keeps_mappings_apart(self, khp_remapped_products):
        """
        The PASS 0.1 failure mode, reproduced directly: pooling conformers by
        state.hash hands one mapping's geometry to another mapping's state.
        """
        smi, mols = next(iter(khp_remapped_products.items()))
        first, second = state(mols[0]), state(mols[1])
        first.conformers["marker"] = "belongs_to_first_mapping"

        by_hash = {}
        by_identity = {}
        for sp in (first, second):
            by_hash.setdefault(sp.hash, {}).update(sp.conformers)
            by_identity.setdefault(sp.identity, {}).update(sp.conformers)

        assert "marker" in by_hash[second.hash], (
            f"expected pooling by hash to leak across mappings of {smi}"
        )
        assert "marker" not in by_identity[second.identity], (
            f"pooling by identity leaked a conformer across mappings of {smi}"
        )
