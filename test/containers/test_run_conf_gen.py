"""
Tests for the two container conformer scripts, both named run_conf_gen.py:
containers/rdkit_conf/run_conf_gen.py (first part of this file) and
containers/auto3d_conf/run_conf_gen.py (second part).

containers/rdkit_conf/run_conf_gen.py is the script that runs inside
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



# =====================================================================
# containers/auto3d_conf/run_conf_gen.py (Auto3D 3.2.2)
# =====================================================================
#
# This script imports Auto3D, which the host test env does not have, so it is
# tested two ways: statically, by reading its source next to the rdkit_conf
# script's (always runs), and end to end through the built image
# containers/erm42_yarp_auto3d_conf.sif (skipped when the image or apptainer is
# missing; ~3 min).

import ast
import shutil

AUTO3D_SCRIPT = Path(__file__).parents[2] / "containers" / "auto3d_conf" / "run_conf_gen.py"
AUTO3D_IMAGE = Path(__file__).parents[2] / "containers" / "erm42_yarp_auto3d_conf.sif"


def _function_source(path, name):
    """Source text of top-level function `name` in `path`, docstring included."""
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(path.read_text(), node)


def _embed_kwargs(path):
    """Keyword arguments of the EmbedMultipleConfs call in `path`, as source text."""
    tree = ast.parse(path.read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and getattr(n.func, "attr", None) == "EmbedMultipleConfs")
    return {kw.arg: ast.unparse(kw.value) for kw in call.keywords}


def _module_constant(path, name):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == name for t in n.targets))
    return ast.literal_eval(node.value)


class TestAuto3dScriptMatchesRdkitScript:
    """
    The auto3d option is meant to differ from the rdkit option only in the
    optimizer and the ranking: same embedding call, same fragment placement.
    The code for both is duplicated in the two scripts (each image builds from
    its own directory), so these tests keep the copies from drifting apart.
    """

    def test_separate_fragments_is_an_exact_copy(self):
        """Character-for-character the same function, docstring and comments included."""
        assert (_function_source(AUTO3D_SCRIPT, "separate_fragments")
                == _function_source(SCRIPT, "separate_fragments"))

    def test_embedding_call_has_the_same_settings(self):
        """
        The same EmbedMultipleConfs keyword arguments (classy_yarp's ETKDG
        settings plus YARP's n_conf / prune_rms_thresh / seed / n_threads).
        A changed or added setting in either script fails this.
        """
        assert _embed_kwargs(AUTO3D_SCRIPT) == _embed_kwargs(SCRIPT)

    def test_fragments_are_placed_at_the_same_contact(self):
        assert (_module_constant(AUTO3D_SCRIPT, "FRAGMENT_CONTACT")
                == _module_constant(SCRIPT, "FRAGMENT_CONTACT") == 3.0)

    def test_offers_exactly_the_three_models(self):
        """
        aimnet2 (default), aimnet2-nse (spin-aware, the only one valid for
        open-shell states) and ani2x. The image pre-fetches the weights of the
        two AIMNet2 models; a fourth model here would need a download at run time.
        """
        assert _module_constant(AUTO3D_SCRIPT, "ENGINES") == {
            "aimnet2": "aimnet2", "aimnet2-nse": "aimnet2-nse", "ani2x": "ANI2x"}


def _run_image(species, workdir, lot, multiplicity=1):
    """
    Run the auto3d_conf image the way a YARP job does (`apptainer run -e`, job
    directory bound to /work) on `species`, written as RdkitConfCalculator
    writes its input. Returns the CompletedProcess.
    """
    mol = yarpecule_to_rdmol(elements=species.elements, adj=species.adj_mat,
                             bond_orders=species.bond_mats[0],
                             atom_info=species._atom_info, geo=species.geo)
    Chem.MolToMolFile(mol, str(workdir / "input.mol"))
    return subprocess.run(
        ["apptainer", "run", "-e", "--bind", f"{workdir}:/work", "--pwd", "/work", str(AUTO3D_IMAGE),
         "input.mol", "--lot", lot, "--multiplicity", str(multiplicity), "--n_conf", "50",
         "--prune_rms_thresh", "0.1", "--n_threads", "4", "--seed", "42"],
        capture_output=True, text=True,
    )


def _lowest_energy(workdir):
    return float((workdir / "auto3d.energies").read_text().split()[1])


@pytest.mark.skipif(not (AUTO3D_IMAGE.exists() and shutil.which("apptainer")),
                    reason="needs containers/erm42_yarp_auto3d_conf.sif and apptainer")
class TestAuto3dImage:
    """The auto3d_conf image end to end, as YARP runs it."""

    def test_multi_fragment_conformers_keep_their_bonding(self, khp_products, tmp_path):
        """
        C=O.O=CCO with AIMNet2. Without fragment separation, AIMNet2 optimizes
        overlapping fragments and most conformers come out with broken bonds or
        never converge; Auto3D's ranking then silently drops them, so the job
        still "succeeds" with a few survivors (measured: 15 of 50 kept without
        separation, 47 of 50 with it; with Auto3D's own embedding, none
        survive -- measured during development).
        Expected: a clean exit; most embedded conformers survive (>= 60%);
        every one keeps the state's bonding (YARP's own check); no fragment is
        flung away (closest contact < 6 A: AIMNet2 leaves complexes at 1.9-4.7 A,
        the unfixed rdkit_conf failure reached 72 A); energies ascending.
        """
        state = khp_products["C=O.O=CCO"]
        result = _run_image(state, tmp_path, "aimnet2")
        assert result.returncode == 0, result.stderr[-2000:]
        assert "Auto3D conformer generation terminated normally." in result.stdout
        _, geos = xyz_parse(tmp_path / "auto3d_conformers.xyz", multiple=True)
        n_embedded = int(result.stdout.split("Embedded ")[1].split()[0])
        assert len(geos) >= 0.6 * n_embedded, f"only {len(geos)} of {n_embedded} conformers survived"
        labels = _fragment_ids(state.adj_mat)
        for geo in map(np.asarray, geos):
            assert compare_adjacency(state.elements, geo, state.adj_mat)[0]
            assert max(_closest_contacts(geo, labels)) < 6.0
        energies = np.loadtxt(tmp_path / "auto3d.energies", ndmin=2)[:, 1]
        assert np.all(np.diff(energies) >= 0)

    def test_multiplicity_reaches_the_spin_aware_model(self, khp_products, tmp_path):
        """
        CCC=O.O=O: YARP's Lewis structure draws O2 closed-shell (O=O), so
        Auto3D's own guess would be a singlet. With --multiplicity 3 the
        spin-aware model must see a triplet, which lies well below the
        singlet (measured 34 kcal/mol). If the multiplicity were not passed
        through, both runs would be the same singlet calculation.
        """
        state = khp_products["CCC=O.O=O"]
        energies = {}
        for mult in (1, 3):
            workdir = tmp_path / f"m{mult}"
            workdir.mkdir()
            result = _run_image(state, workdir, "aimnet2-nse", multiplicity=mult)
            assert result.returncode == 0, result.stderr[-2000:]
            energies[mult] = _lowest_energy(workdir)
        assert (energies[1] - energies[3]) * 627.509 > 10

    def test_ani2x_refuses_an_open_shell_state(self, tmp_path):
        """
        ANI-2x cannot model a radical; Auto3D refuses it rather than computing
        it as closed-shell. The job then fails visibly (non-zero exit, no
        termination line), so YARP marks the task failed instead of keeping a
        wrong result. (Default AIMNet2 does NOT refuse: it runs a radical as
        closed-shell with only a warning, so YARP's input check has to stop
        that combination before submission.)
        """
        result = _run_image(yp.yarpecule("C[CH2]"), tmp_path, "ani2x", multiplicity=2)
        assert result.returncode != 0
        assert "terminated normally" not in result.stdout
        assert not (tmp_path / "auto3d_conformers.xyz").exists()
