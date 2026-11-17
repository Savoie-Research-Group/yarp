"""
Tests for duplicate TS detection (conformer.is_duplicate_of, unique_ts).

Nothing used to check whether two TSs of a reaction were the same structure, so
duplicates were each run through the next TS optimization and IRC. In the
ts_handoff survey (KHP pickle, 70 DFT TSs), 10 were duplicates of another TS.
Two TSs are duplicates when their electronic energies agree within DUP_DE_MAX
and their geometries within DUP_RMSD_MAX, after pairing up equivalent H atoms
that swapped labels; mirror images count while DUP_MATCH_MIRRORS is on.
"""
import importlib

import numpy as np
import pytest

from yarp.reaction.conformer import conformer, unique_ts
from yarp.util.constants import Constants

conformer_module = importlib.import_module("yarp.reaction.conformer")

# A made-up, asymmetric C2O3H4 TS: no symmetry, so its mirror image can't be
# rotated onto it. H5 and H6 are both on C0.
ELEMENTS = ["C", "C", "O", "O", "O", "H", "H", "H", "H"]
GEO = np.array([
    [0.000, 0.000, 0.000],
    [1.520, 0.000, 0.000],
    [2.080, 1.330, 0.210],
    [-0.480, 1.290, -0.330],
    [2.010, -0.950, 0.880],
    [-0.360, -0.330, 0.980],
    [-0.330, -0.720, -0.750],
    [1.870, -0.360, -0.970],
    [3.040, 1.270, 0.300],
])
E0 = -228.0


def ts(geo=GEO, energy=E0, elements=ELEMENTS):
    conf = conformer()
    conf.elements = list(elements)
    conf.geo = np.array(geo, dtype=float)
    conf.properties["internal_energy_Eh"] = energy
    return conf


def kcal_to_eh(x):
    return x / Constants.ha_to_kcalmol


def rotated(geo):
    """geo rotated 40 degrees about z and shifted: the same structure."""
    c, s = np.cos(np.radians(40)), np.sin(np.radians(40))
    return geo @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) + [3.0, -1.0, 2.0]


class TestIsDuplicateOf:
    def test_same_structure_moved_in_space(self):
        assert ts().is_duplicate_of(ts(rotated(GEO), E0 + kcal_to_eh(0.01)))

    def test_swapped_equivalent_h_atoms(self):
        # The same TS with the labels of the two CH2 hydrogens swapped.
        swapped = GEO.copy()
        swapped[[5, 6]] = GEO[[6, 5]]

        assert ts().is_duplicate_of(ts(rotated(swapped)))

    def test_mirror_image_counts_while_switched_on(self, monkeypatch):
        mirror = ts(GEO * [-1, 1, 1])
        assert ts().is_duplicate_of(mirror)

        monkeypatch.setattr(conformer_module, "DUP_MATCH_MIRRORS", False)
        assert not ts().is_duplicate_of(mirror)

    def test_h_rotamer_is_not_a_duplicate(self):
        # Same heavy atoms, the O-H rotated to another position, same energy.
        rotamer = GEO.copy()
        rotamer[8] = [2.300, 1.700, -0.650]

        assert not ts().is_duplicate_of(ts(rotamer))

    @pytest.mark.parametrize("de_kcal, expected", [(0.02, True), (0.05, False)])
    def test_energy_gap(self, de_kcal, expected):
        assert ts().is_duplicate_of(ts(energy=E0 + kcal_to_eh(de_kcal))) is expected

    def test_missing_energy_is_never_a_duplicate(self):
        assert not ts().is_duplicate_of(ts(energy=None))
        assert not ts(energy=None).is_duplicate_of(ts(energy=None))

    def test_fewer_than_three_heavy_atoms_use_all_atoms(self):
        # O2H2: two heavy atoms, so no heavy-atom fit; plain all-atom RMSD.
        elements = ["O", "O", "H", "H"]
        geo = np.array([[0.0, 0.0, 0.0], [1.45, 0.0, 0.0], [-0.3, 0.9, 0.2], [1.75, 0.2, 0.9]])
        moved = geo.copy()
        moved[3] = [1.75, -0.9, 0.2]

        assert ts(geo, elements=elements).is_duplicate_of(ts(rotated(geo), elements=elements))
        assert not ts(geo, elements=elements).is_duplicate_of(ts(moved, elements=elements))

        h3 = ["H", "H", "H"]
        h3_geo = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [1.8, 0.3, 0.0]])
        assert ts(h3_geo, elements=h3).is_duplicate_of(ts(rotated(h3_geo), elements=h3))


class TestUniqueTS:
    def test_keeps_lowest_energy_in_order_and_marks_the_rest(self):
        a = ts(energy=E0)
        b = ts(rotated(GEO), energy=E0 - kcal_to_eh(0.01))  # duplicate of a, lower
        c = ts(energy=E0 + kcal_to_eh(5.0))                 # distinct (energy)

        assert unique_ts([("a", a), ("b", b), ("c", c)]) == [b, c]
        assert a.properties["duplicate_of"] == "b"
        assert "duplicate_of" not in b.properties and "duplicate_of" not in c.properties

    def test_energy_tie_keeps_the_first(self):
        a, b = ts(), ts(rotated(GEO))

        assert unique_ts([("a", a), ("b", b)]) == [a]
        assert b.properties["duplicate_of"] == "a"

    def test_no_duplicates_no_change(self):
        confs = [(str(i), ts(energy=E0 + kcal_to_eh(i))) for i in range(3)]

        assert unique_ts(confs) == [c for _, c in confs]
        assert not any("duplicate_of" in c.properties for _, c in confs)
