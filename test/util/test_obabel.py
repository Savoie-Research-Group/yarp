from copy import deepcopy

import numpy as np
import pytest
from openbabel import pybel, openbabel as ob

import yarp as yp
from yarp.reaction.enum import enumerate_products
from yarp.yarpecule.graph.adjacency import table_generator
from yarp.yarpecule.hashes import yarpecule_hash
from yarp.util.write_files import mol_write_yp

from yarp.util.obabel import obabel_joint_opt, _ensure_setup_needed
from yarp.util.obabel import obabel_embed
from rdkit import Chem

class TestFFOpt:
    def test_haa_2_ring(self):
        """
        Recording a bug in Open Babel for the formation of strained
        3 member ring product geometry when starting from HAA reactant geometry
        """
        reactant = yp.yarpecule('O=CCO')
        products = enumerate_products(reactant, 2, 2, mode="concerted")
        target_hash = yarpecule_hash(yp.yarpecule('O=C1CO1.[H][H]'))
        target_product = next(p for p in products if str(p.hash) == str(target_hash))

        opt_geo = obabel_joint_opt(target_product, target_product.bond_mats[0],
                                   target_product.adj_mat, lot='uff', maxiter=200)
        opt_adj = table_generator(elements=target_product.elements, geometry=opt_geo)

        diff = opt_adj - target_product.adj_mat
        assert not np.all(diff == 0)

    def test_3hp_2_acetaldehyde(self):
        """
        Open Babel correctly optimizes product acetaldehyde + O2 from 3HP geom
        """
        reactant = yp.yarpecule('O=CCOO')
        products = enumerate_products(reactant, 2, 2, mode="concerted")
        target = yp.yarpecule('CC=O.O=O')

        target_product = next(p for p in products if str(p.hash) == str(target.hash))

        opt_geo = obabel_joint_opt(target_product, target_product.bond_mats[0],
                                   target_product.adj_mat, lot='uff', maxiter=200)
        opt_adj = table_generator(elements=target_product.elements, geometry=opt_geo)

        diff = opt_adj - target_product.adj_mat
        assert np.all(diff == 0)

    def test_3hp_2_aldehyde(self):
        """
        Open Babel correctly optimizes product formaldehyde + ester from 3HP geom
        """
        reactant = yp.yarpecule('O=CCOO')
        products = enumerate_products(reactant, 2, 2, mode="concerted")
        target = yp.yarpecule('C=O.O=CO')

        target_product = next(p for p in products if str(p.hash) == str(target.hash))

        opt_geo = obabel_joint_opt(target_product, target_product.bond_mats[0],
                                   target_product.adj_mat, lot='uff', maxiter=200)
        opt_adj = table_generator(elements=target_product.elements, geometry=opt_geo)

        diff = opt_adj - target_product.adj_mat
        assert np.all(diff == 0)


class TestForceFieldCache:
    """
    Open Babel's force field is a module-level singleton, and
    `OBForceField::Setup` skips re-parameterization when `IsSetupNeeded()`
    reports the incoming molecule resembles the cached one. Per the Open Babel
    header that check compares only atom count, bond count and atomic numbers.

    Products enumerated from one parent share the atom count and element order
    exactly and take only a handful of distinct bond counts, so they collide
    routinely -- and the second molecule of a colliding pair then gets
    optimized with the first one's parameters. Measured at 2.33 A on
    O=CCCOO -> O=COCCO before `_ensure_setup_needed` was added.
    """

    # (previous product, product whose setup the cache would swallow)
    COLLIDING_PAIRS = [
        ("C1CO1.O=CO", "O=COCCO"),
        ("O.O=C[C@H]1CO1", "O=C[C@H](O)CO"),
        ("O.O=C1CCO1", "O=C(O)CCO"),
    ]

    # Subset whose targets are also bit-reproducible, so "same answer either
    # way" is a meaningful assertion. O=C(O)CCO is excluded because Open Babel
    # returns a different geometry for it on every call regardless of the cache
    # -- see TestVerdictStability.
    REPRODUCIBLE_PAIRS = [
        ("C1CO1.O=CO", "O=COCCO"),
        ("O.O=C[C@H]1CO1", "O=C[C@H](O)CO"),
        ("OOC1CCO1", "OC1CCOO1"),
    ]

    def _as_obmol(self, product, tmp_path, name):
        mol_file = str(tmp_path / f"{name}.mol")
        mol_write_yp(mol_file, product.elements, product.geo,
                     product.bond_mats[0], product.adj_mat)
        return next(pybel.readfile("mol", mol_file))

    def _invalidate(self, ff):
        """Clear the cache using a molecule that cannot be confused for a product."""
        scratch = pybel.readstring("smi", "O")
        scratch.make3D(forcefield="uff", steps=1)
        ff.Setup(scratch.OBMol)

    def _park(self, ff, product, tmp_path, name):
        """
        Genuinely install `product` as the cached molecule.

        A bare `ff.Setup(product)` is not enough: if whatever is already cached
        collides with `product`, that Setup is itself swallowed and the old
        molecule stays. An earlier version of this test parked that way, saw no
        effect, and passed with the fix reverted.
        """
        self._invalidate(ff)
        ff.Setup(self._as_obmol(product, tmp_path, name).OBMol)

    @pytest.mark.parametrize("previous_smi,target_smi", COLLIDING_PAIRS)
    def test_guard_forces_reparameterization(self, khp_products, tmp_path,
                                             previous_smi, target_smi):
        """After the guard, Open Babel must always report a setup is needed."""
        ff = ob.OBForceField.FindForceField("uff")

        self._park(ff, khp_products[previous_smi], tmp_path, "previous")

        target = self._as_obmol(khp_products[target_smi], tmp_path, "target")
        # sanity: this pair is only interesting because the cache would be reused
        assert not ff.IsSetupNeeded(target.OBMol), (
            f"{previous_smi} -> {target_smi} no longer collides; pick another pair"
        )

        _ensure_setup_needed(target, "uff")
        assert ff.IsSetupNeeded(target.OBMol)

    @pytest.mark.parametrize("previous_smi,target_smi", REPRODUCIBLE_PAIRS)
    def test_result_independent_of_previous_molecule(self, khp_products, tmp_path,
                                                     previous_smi, target_smi):
        """
        The regression itself: a poisoned cache must not change the answer.

        Both the reference and the poisoned run need their cache state set
        explicitly -- see `_park`. Measured deviation with the guard removed is
        2.1 to 2.7 A on these pairs.
        """
        ff = ob.OBForceField.FindForceField("uff")
        target = khp_products[target_smi]
        bem, adj = target.bond_mats[0], target.adj_mat

        self._invalidate(ff)
        reference = obabel_joint_opt(deepcopy(target), bem, adj, lot="uff")
        assert reference is not None

        self._park(ff, khp_products[previous_smi], tmp_path, "previous")
        target_obmol = self._as_obmol(target, tmp_path, "target")
        assert not ff.IsSetupNeeded(target_obmol.OBMol), (
            f"{previous_smi} -> {target_smi} no longer collides; pick another pair"
        )

        poisoned = obabel_joint_opt(deepcopy(target), bem, adj, lot="uff")

        assert poisoned is not None
        assert np.allclose(reference, poisoned), (
            f"a cache holding {previous_smi} changed the result for "
            f"{target_smi} by up to {np.abs(reference - poisoned).max():.4f} A"
        )


class TestVerdictStability:
    """
    Open Babel is not bit-reproducible for every molecule: a few return a
    different geometry each call, even from byte-identical input in a cold
    process. Cause is unidentified and lives inside Open Babel; it is not
    strain, not threading, and not the singleton cache. Raising maxiter does
    not help.

    That is accepted deliberately. What must not vary is the adjacency
    verdict, because that is what decides whether a reaction survives. Over 8
    repeats none of these ever failed, and all did so with zero mismatches.
    """

    KNOWN_VARIABLE = ["CC(=O)COO", "OOC1CCO1", "O=C(O)CCO"]
    REPEATS = 3

    @pytest.mark.parametrize("smi", KNOWN_VARIABLE)
    def test_adjacency_verdict_is_stable(self, khp_products, smi):
        product = khp_products[smi]
        bem, adj = product.bond_mats[0], product.adj_mat

        for attempt in range(self.REPEATS):
            geo = obabel_joint_opt(deepcopy(product), bem, adj, lot="uff")
            assert geo is not None, f"{smi} returned no geometry on attempt {attempt}"
            n_mismatch = int(
                np.abs(table_generator(product.elements, geo) - adj).sum() // 2
            )
            assert n_mismatch == 0, (
                f"{smi} failed the adjacency check on attempt {attempt} "
                f"({n_mismatch} mismatches); coordinates may vary run to run "
                f"but the verdict may not"
            )


def kekule_graph(smiles):
    """Elements, kekulized bond orders and formal charges from RDKit, explicit H included."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    Chem.Kekulize(mol, clearAromaticFlags=True)
    bond_mat = np.zeros([mol.GetNumAtoms(), mol.GetNumAtoms()])
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_mat[i, j] = bond_mat[j, i] = bond.GetBondTypeAsDouble()
    return [_.GetSymbol() for _ in mol.GetAtoms()], bond_mat, [_.GetFormalCharge() for _ in mol.GetAtoms()]


class TestObabelEmbed:
    def test_graphs(self, obabel_embed_cases):
        """
        A finite geometry in the order of the given atoms, with no two atoms on top of each
        other, identical when built again (Open Babel's builder is random unless each build
        is a fresh, seeded process); bond lengths sane where UFF keeps the bonds. Every case
        is checked; all failing cases are reported together.
        """
        failed = []
        for smiles, check_bonds in obabel_embed_cases:
            try:
                elements, bond_mat, fc = kekule_graph(smiles)
                geo = obabel_embed(elements, bond_mat, fc)
                again = obabel_embed(elements, bond_mat, fc)
                dist = np.linalg.norm(geo[:, None] - geo[None], axis=-1) + 10 * np.eye(len(elements))
                bonded = dist[np.triu(bond_mat, 1) > 0]
                problems = [msg for bad, msg in [
                    (geo.shape != (len(elements), 3) or not np.isfinite(geo).all(), "geometry shape or values"),
                    (not np.array_equal(geo, again), "a second build gave a different geometry"),
                    (dist.min() <= 0.5, "overlapping atoms"),
                    (check_bonds and not ((bonded > 0.9) & (bonded < 2.4)).all(), f"bond lengths {bonded.round(2)}"),
                ] if bad]
            except Exception as e:
                problems = [f"{type(e).__name__}: {e}"]
            if problems:
                failed.append(f"{smiles}: {'; '.join(problems)}")
        assert not failed, f"{len(failed)} case(s) failed:\n" + "\n".join(failed)

    def test_isolated_worker(self, obabel_worker_shadows, tmp_path, monkeypatch):
        """
        The worker process imports nothing from the current directory: with each shadowing
        file there, the build is the same as without it. A worker that dies without a
        traceback raises RuntimeError. Every case is checked; all failing cases are reported together.
        """
        elements, bond_mat, fc = kekule_graph("CCO")
        expected = obabel_embed(elements, bond_mat, fc)
        failed = []
        for shadow in obabel_worker_shadows:
            folder = tmp_path / shadow.replace("/", "_")
            (folder / shadow).parent.mkdir(parents=True, exist_ok=True)
            (folder / shadow).write_text(f"raise ImportError('{shadow} was imported from the current directory')\n")
            monkeypatch.chdir(folder)
            try:
                got = np.array_equal(obabel_embed(elements, bond_mat, fc), expected)
            except Exception as e:
                got = f"{type(e).__name__}: {e}"
            if got is not True:
                failed.append(f"{shadow} in the current directory: {got}")

        monkeypatch.setattr("sys.executable", "/bin/false")
        try:
            obabel_embed(elements, bond_mat, fc)
            failed.append("worker that exits without a traceback: no error raised")
        except RuntimeError:
            pass
        except Exception as e:
            failed.append(f"worker that exits without a traceback: {type(e).__name__}: {e}")
        assert not failed, f"{len(failed)} case(s) failed:\n" + "\n".join(failed)
