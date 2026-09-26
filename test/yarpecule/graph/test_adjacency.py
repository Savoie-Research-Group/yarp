"""
Tests for the shared adjacency comparison helper.

`compare_adjacency` centralises the "does this geometry still have the bonding
we asked for" check that callers were each hand-rolling as
`table_generator(...) - target_adj` tested against zero.
"""
import numpy as np
import pytest

from yarp.yarpecule.graph.adjacency import (
    compare_adjacency,
    describe_adjacency_change,
    table_generator,
)


class TestCompareAdjacency:
    def test_own_geometry_matches_own_graph(self, khp_parent):
        matches, n_broken, n_formed = compare_adjacency(
            khp_parent.elements, khp_parent.geo, khp_parent.adj_mat
        )

        assert matches
        assert (n_broken, n_formed) == (0, 0)

    def test_agrees_with_table_generator(self, khp_parent):
        """The helper must not perceive bonds differently from the raw call."""
        perceived = table_generator(khp_parent.elements, khp_parent.geo)

        assert compare_adjacency(khp_parent.elements, khp_parent.geo, perceived)[0]

    def test_counts_a_missing_bond_as_broken(self, khp_parent):
        """A target demanding a bond the geometry does not show reads as broken."""
        target = khp_parent.adj_mat.copy()
        i, j = np.argwhere(target == 0)[0]
        if i == j:
            i, j = np.argwhere(target == 0)[1]
        target[i][j] = 1
        target[j][i] = 1

        matches, n_broken, n_formed = compare_adjacency(
            khp_parent.elements, khp_parent.geo, target
        )

        assert not matches
        assert (n_broken, n_formed) == (1, 0)

    def test_counts_an_extra_bond_as_formed(self, khp_parent):
        """A target missing a bond the geometry does show reads as formed."""
        target = khp_parent.adj_mat.copy()
        i, j = np.argwhere(target == 1)[0]
        target[i][j] = 0
        target[j][i] = 0

        matches, n_broken, n_formed = compare_adjacency(
            khp_parent.elements, khp_parent.geo, target
        )

        assert not matches
        assert (n_broken, n_formed) == (0, 1)

    def test_counts_are_per_bond_not_per_matrix_element(self, khp_parent):
        """
        Both matrices are symmetric, so every disagreement appears twice. The
        counts must be halved or every warning reports double.
        """
        target = khp_parent.adj_mat.copy()
        bonds = np.argwhere(np.triu(target) == 1)[:2]
        for i, j in bonds:
            target[i][j] = 0
            target[j][i] = 0

        _, n_broken, n_formed = compare_adjacency(
            khp_parent.elements, khp_parent.geo, target
        )

        assert (n_broken, n_formed) == (0, 2)

    def test_products_do_not_match_the_parent_graph(self, khp_parent, khp_products):
        """
        Enumerated products carry the parent's coordinates under their own
        bonding, so almost none of them reproduce their own graph before
        relaxation. That is exactly the condition the pre-optimization exists
        to fix, and the helper has to be able to see it.
        """
        mismatched = 0
        for prod in khp_products.values():
            matches, _, _ = compare_adjacency(prod.elements, prod.geo, prod.adj_mat)
            if not matches:
                mismatched += 1

        assert mismatched > 0


class TestDescribeAdjacencyChange:
    @pytest.mark.parametrize("n_broken, n_formed", [(0, 0), (1, 0), (0, 3), (2, 5)])
    def test_mentions_both_counts(self, n_broken, n_formed):
        text = describe_adjacency_change(n_broken, n_formed)

        assert str(n_broken) in text
        assert str(n_formed) in text
        assert "broken" in text and "formed" in text
