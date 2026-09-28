from copy import deepcopy

import numpy as np
import pytest
from openbabel import pybel, openbabel as ob

import yarp as yp
from yarp.reaction.enum import enumerate_products
from yarp.yarpecule.graph.adjacency import table_generator
from yarp.util.write_files import mol_write_yp

from yarp.util.obabel import obabel_joint_opt, _ensure_setup_needed

class TestFFOpt:
    def test_haa_2_ring(self):
        """
        Recording a bug in Open Babel for the formation of strained
        3 member ring product geometry when starting from HAA reactant geometry
        """
        reactant = yp.yarpecule('O=CCO')
        products = enumerate_products(reactant, 2, 2, mode="concerted")
        target_hash = "1034502.7961211"
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
