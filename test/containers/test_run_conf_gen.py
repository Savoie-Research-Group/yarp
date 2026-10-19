"""
Tests for containers/rdkit_conf/run_conf_gen.py, the script that runs inside
the rdkit_conf image. Given input.mol, it embeds conformers, separates the
fragments of a multi-species state, optimizes with UFF or MMFF94, removes
duplicates, and writes rdkit_conformers.xyz and rdkit.energies.

test/reaction/test_conf_gen.py covers the YARP side (what goes into the
container and what is kept from its output). This file runs the script
itself.

The script has no YARP imports, so these tests run it with the host's RDKit,
not the image's pinned 2022.03.5. Both versions were checked by hand during
development and showed the same multi-fragment failure and the same fix, but
CI only exercises the host version.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

import yarp as yp
from yarp.util.rdkit import yarpecule_to_rdmol
from yarp.yarpecule.input_parsers import xyz_parse
from yarp.yarpecule.graph.adjacency import compare_adjacency

SCRIPT = Path(__file__).parents[2] / "containers" / "rdkit_conf" / "run_conf_gen.py"

_spec = importlib.util.spec_from_file_location("run_conf_gen", SCRIPT)
run_conf_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_conf_gen)


def _fragment_ids(adj):
    """Fragment index of every atom, from the adjacency matrix."""
    labels = [-1] * len(adj)
    n_frag = 0
    for seed in range(len(adj)):
        if labels[seed] >= 0:
            continue
        stack, labels[seed] = [seed], n_frag
        while stack:
            i = stack.pop()
            for j in np.nonzero(adj[i])[0]:
                if labels[j] < 0:
                    labels[j] = n_frag
                    stack.append(j)
        n_frag += 1
    return np.array(labels)


def _closest_contacts(geo, frag_ids):
    """Each fragment's closest atom-atom distance to the rest of the system."""
    dist = np.linalg.norm(geo[:, None] - geo[None], axis=-1)
    return [dist[frag_ids == f][:, frag_ids != f].min() for f in range(frag_ids.max() + 1)]


def _run(species, workdir, *extra):
    """
    Run the script end to end, as a YARP job would: write input.mol the way
    RdkitConfCalculator does, call the script with the arguments YARP passes
    for the KHP test input, check it exits cleanly and prints the line
    check_output looks for, and return the conformer geometries it wrote.
    """
    mol = yarpecule_to_rdmol(elements=species.elements, adj=species.adj_mat,
                             bond_orders=species.bond_mats[0],
                             atom_info=species._atom_info, geo=species.geo)
    Chem.MolToMolFile(mol, str(workdir / "input.mol"))
    # The arguments RdkitConfCalculator passes for the KHP test input.
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "input.mol", "--lot", "uff", "--n_conf", "50",
         "--prune_rms_thresh", "0.1", "--n_threads", "4", "--seed", "42", *extra],
        cwd=workdir, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert run_conf_gen.TERMINATION_MSG in result.stdout
    _, geos = xyz_parse(workdir / "rdkit_conformers.xyz", multiple=True)
    return [np.asarray(g) for g in geos]


class TestSeparateFragments:
    """
    separate_fragments on its own, on hand-built water molecules.

    RDKit embeds every fragment of a multi-species state (e.g. C=O.O=CCO)
    around the same origin, so they overlap. separate_fragments translates
    them apart, before the force-field step, so that each one sits `contact`
    A (3.0 by default) from the others: close enough to stay a bimolecular
    complex, far enough not to be squashed into each other.
    """

    # Three water-like fragments stacked on the same spot, as the embedding leaves them.
    FRAGS = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
    WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])

    def _overlapping(self, n_frag):
        offsets = [np.array([0.0, 0.0, 0.0]), np.array([0.3, 0.1, 0.0]), np.array([-0.2, 0.4, 0.1])]
        return np.vstack([self.WATER + offsets[i] for i in range(n_frag)])

    def test_single_fragment_is_untouched(self):
        """Ordinary one-molecule states must come back exactly as embedded."""
        geo = self.WATER.copy()
        assert np.array_equal(run_conf_gen.separate_fragments(geo, ((0, 1, 2),), 3.0), geo)

    @pytest.mark.parametrize("n_frag", [2, 3])
    @pytest.mark.parametrize("contact", [3.0, 5.0])
    def test_every_fragment_ends_at_least_contact_from_the_rest(self, n_frag, contact):
        """
        Start with 2 or 3 overlapping waters and separate them with a 3.0 or
        5.0 A target. Fragments are placed one at a time, so check that:
          - the closest approach between any two fragments is exactly
            `contact` (none left overlapping, none pushed further than needed);
          - each fragment after the first touches the ones already placed at
            exactly `contact`, so the complex stays together as a cluster.
        """
        frags = self.FRAGS[:n_frag]
        geo = run_conf_gen.separate_fragments(self._overlapping(n_frag), frags, contact)
        labels = np.repeat(np.arange(n_frag), 3)
        contacts = _closest_contacts(geo, labels)
        assert min(contacts) == pytest.approx(contact, abs=1e-6)
        # Each fragment placed after the first touches the ones before it at exactly `contact`.
        dist = np.linalg.norm(geo[:, None] - geo[None], axis=-1)
        for k in range(1, n_frag):
            earlier = [i for f in frags[:k] for i in f]
            assert dist[np.ix_(frags[k], earlier)].min() == pytest.approx(contact, abs=1e-6)

    def test_only_translates_each_fragment(self):
        """
        Every atom of a fragment moves by the same vector. So bond lengths,
        angles and the fragment's internal shape from embedding are unchanged.
        """
        before = self._overlapping(3)
        after = run_conf_gen.separate_fragments(before, self.FRAGS, 3.0)
        for frag in self.FRAGS:
            shift = after[list(frag)] - before[list(frag)]
            assert np.allclose(shift, shift[0], atol=1e-12)

    def test_repeated_calls_are_bit_identical(self):
        """There is no randomness in the placement, so the same input gives the same output."""
        geo = self._overlapping(3)
        first = run_conf_gen.separate_fragments(geo, self.FRAGS, 3.0)
        assert np.array_equal(first, run_conf_gen.separate_fragments(geo, self.FRAGS, 3.0))


# Multi-fragment states: four real KHP depth-1 products (one with free H2,
# the hardest case for the force field) plus a three-fragment state.
CASES = ["C=O.O=CCO", "C=C.O=COO", "C1CO1.O=CO", "O=C=CCOO.[H][H]", "C=O.C=O.O"]


@pytest.fixture(scope="module")
def species(khp_products):
    cases = {smi: khp_products[smi] for smi in CASES[:-1]}
    # No break-2/form-2 product of KHP has three fragments.
    cases["C=O.C=O.O"] = yp.yarpecule("C=O.C=O.O")
    return cases


class TestMultiFragmentConformers:
    """
    The whole script on the multi-fragment states in CASES.

    These tests guard the fix in a716100. Before fragments were separated
    ahead of the force-field optimization, the force field started from
    overlapping fragments. Then:
      - most conformers came out with mangled geometry (e.g. both H of
        CH2=O on the same spot), so YARP's connectivity check threw them
        away;
      - fragments were thrown tens of A apart, which crashed GSM downstream.
    """

    @pytest.mark.parametrize("smi", CASES)
    def test_every_conformer_keeps_the_bonding(self, smi, species, tmp_path):
        """
        The script writes at least one conformer, and every one of them
        still has the state's bonding, so none would be dropped by YARP's
        connectivity filter (TestRdkitScrapeData in test_conf_gen.py).
        """
        state = species[smi]
        geos = _run(state, tmp_path)
        assert geos
        failing = [i for i, g in enumerate(geos)
                   if not compare_adjacency(state.elements, g, state.adj_mat)[0]]
        assert not failing, f"{len(failing)} of {len(geos)} conformers of {smi} lost their bonding"

    @pytest.mark.parametrize("smi", CASES)
    def test_fragments_stay_in_contact(self, smi, species, tmp_path):
        """
        After optimization, every fragment's closest contact with the rest
        is between 2.5 and 3.5 A: still a complex, neither overlapping nor
        drifted apart. Measured 2.71-3.20 A with the fix; up to 72 A without.
        """
        state = species[smi]
        labels = _fragment_ids(state.adj_mat)
        for i, geo in enumerate(_run(state, tmp_path)):
            contacts = _closest_contacts(geo, labels)
            assert 2.5 < min(contacts) and max(contacts) < 3.5, (
                f"{smi} conformer {i}: fragment contacts {np.round(contacts, 2)} A"
            )

    @pytest.mark.parametrize("smi", ["C=C.O=COO", "C=O.C=O.O"])
    def test_repeated_runs_are_bit_identical(self, smi, species, tmp_path):
        """
        Two runs with the same seed (42) write byte-identical conformer and
        energy files. This is the point of the pinned container: everyone
        running the same input gets the same conformers.
        """
        first, second = tmp_path / "first", tmp_path / "second"
        first.mkdir()
        second.mkdir()
        _run(species[smi], first)
        _run(species[smi], second)
        for name in ("rdkit_conformers.xyz", "rdkit.energies"):
            assert (first / name).read_bytes() == (second / name).read_bytes(), name
