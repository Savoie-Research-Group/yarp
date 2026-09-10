import numpy as np

import yarp as yp
from yarp.reaction.enum import enumerate_products
from yarp.yarpecule.graph.adjacency import table_generator

from yarp.util.rdkit import (
    yarpecule_to_rdmol,
    geom_from_rdmol,
    rdkit_joint_opt,
    smiles_to_rdmol,
    adj_from_rdmol,
    el_from_rdmol,
)
from yarp.yarpecule.atom_mapping import canon_order
from yarp.util.properties import el_mass

class TestGeom:
    def test_preserved_geom(self):
        fran_smi = '[c:0]12[c:1]([H:7])[c:2]([H:8])[c:3]([H:9])[c:4]([H:10])[c:5]1[C-:12]([O-:11])[N:13]([H:6])[S+2:14]2'
        fran = yp.yarpecule(fran_smi)
        mol = yarpecule_to_rdmol(fran.elements, fran.adj_mat, fran.bond_mats[0], fran._atom_info, fran.geo)

        geo = geom_from_rdmol(mol)

        diff = fran.geo - geo
        assert np.all(diff == 0)

class TestFFOpt:
    def test_haa_2_ring(self):
        """
        RDkit correctly forms strained
        3 member ring product geometry when starting from HAA reactant geometry
        """
        reactant = yp.yarpecule('O=CCO')
        products = enumerate_products(reactant, 2, 2, mode="concerted")
        target_hash = "1034502.7961211"
        target_product = next(p for p in products if str(p.hash) == str(target_hash))

        opt_geo = rdkit_joint_opt(target_product, target_product.bond_mats[0],
                                  target_product.adj_mat, lot='uff', maxiter=200)
        opt_adj = table_generator(elements=target_product.elements, geometry=opt_geo)

        diff = opt_adj - target_product.adj_mat
        assert np.all(diff == 0)

    def test_3hp_2_acetaldehyde(self):
        """
        Recording a bug:
        RDKit fails to optimize product acetaldehyde + O2 from 3HP geom
        """
        reactant = yp.yarpecule('O=CCOO')
        products = enumerate_products(reactant, 2, 2, mode="concerted")
        target = yp.yarpecule('CC=O.O=O')

        target_product = next(p for p in products if str(p.hash) == str(target.hash))

        opt_geo = rdkit_joint_opt(target_product, target_product.bond_mats[0],
                                  target_product.adj_mat, lot='uff', maxiter=200)
        opt_adj = table_generator(elements=target_product.elements, geometry=opt_geo)

        diff = opt_adj - target_product.adj_mat
        assert not np.all(diff == 0)

    def test_3hp_2_aldehyde(self):
        """
        Recording a bug:
        RDKit fails to correctly optimize product formaldehyde + ester from 3HP geom
        """
        reactant = yp.yarpecule('O=CCOO')
        products = enumerate_products(reactant, 2, 2, mode="concerted")
        target = yp.yarpecule('C=O.O=CO')

        target_product = next(p for p in products if str(p.hash) == str(target.hash))

        opt_geo = rdkit_joint_opt(target_product, target_product.bond_mats[0],
                                  target_product.adj_mat, lot='uff', maxiter=200)
        opt_adj = table_generator(elements=target_product.elements, geometry=opt_geo)

        diff = opt_adj - target_product.adj_mat
        assert not np.all(diff == 0)

class TestRDKitSMILESHelpers:
    def test_mapped_aromatic_adjacency_matches_unmapped(self):
        mapped_smiles = "[n:0]1([H:7])[n:4][c:1]([O:3][H:6])[c:2]([H:8])[n:5]1"
        unmapped_smiles = "Oc1cn[nH]n1"

        mapped_mol = smiles_to_rdmol(mapped_smiles)
        unmapped_mol = smiles_to_rdmol(unmapped_smiles)

        mapped_elements = [el.lower() for el in el_from_rdmol(mapped_mol)]
        unmapped_elements = [el.lower() for el in el_from_rdmol(unmapped_mol)]

        mapped_adj = adj_from_rdmol(mapped_mol)
        unmapped_adj = adj_from_rdmol(unmapped_mol)

        mapped_masses = np.array([el_mass[el] for el in mapped_elements])
        unmapped_masses = np.array([el_mass[el] for el in unmapped_elements])

        mapped_elements, mapped_adj, _ = canon_order(
            mapped_elements,
            mapped_adj,
            masses=mapped_masses,
            return_index=False,
        )
        unmapped_elements, unmapped_adj, _ = canon_order(
            unmapped_elements,
            unmapped_adj,
            masses=unmapped_masses,
            return_index=False,
        )

        assert mapped_elements == unmapped_elements
        assert np.array_equal(mapped_adj, unmapped_adj)


class TestJointOptDeterminism:
    """
    RDKit UFF must be a pure function of its input.

    Several checks lean on this: it is the reference the Open Babel
    nondeterminism is measured against, and it is why `quick_geom_opt` trying
    RDKit first keeps most products reproducible even though the Open Babel
    fallback is not. If RDKit ever stops being deterministic, those
    conclusions need revisiting rather than the tests being relaxed.
    """

    REPEATS = 3

    def test_repeated_calls_are_bit_identical(self, khp_products):
        product = khp_products["CC(=O)COO"]
        bem, adj = product.bond_mats[0], product.adj_mat

        first = rdkit_joint_opt(product, bem, adj, lot="uff", maxiter=200)
        assert first is not None

        for attempt in range(1, self.REPEATS):
            again = rdkit_joint_opt(product, bem, adj, lot="uff", maxiter=200)
            assert np.array_equal(first, again), (
                f"rdkit_joint_opt returned a different geometry on attempt "
                f"{attempt}, differing by up to {np.abs(first - again).max():.2e} A"
            )

    def test_deterministic_for_a_product_open_babel_varies_on(self, khp_products):
        """
        Open Babel returns a different geometry for these on every call. RDKit,
        handed exactly the same input, does not -- which is what isolates the
        nondeterminism to Open Babel rather than to anything upstream in yarp.
        """
        for smi in ("CC(=O)COO", "OOC1CCO1", "O=C(O)CCO"):
            product = khp_products[smi]
            bem, adj = product.bond_mats[0], product.adj_mat

            geometries = [rdkit_joint_opt(product, bem, adj, lot="uff", maxiter=200)
                          for _ in range(self.REPEATS)]
            assert all(g is not None for g in geometries), f"{smi} produced no geometry"
            for attempt, geo in enumerate(geometries[1:], start=1):
                assert np.array_equal(geometries[0], geo), (
                    f"rdkit_joint_opt was not reproducible for {smi} on attempt {attempt}"
                )
