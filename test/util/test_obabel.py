import numpy as np

import yarp as yp
from yarp.reaction.enum import enumerate_products
from yarp.yarpecule.graph.adjacency import table_generator

from yarp.util.obabel import obabel_ff_opt

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

        opt_geo = obabel_ff_opt(target_product, lot='uff', maxiter=200)
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

        opt_geo = obabel_ff_opt(target_product, lot='uff', maxiter=200)
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

        opt_geo = obabel_ff_opt(target_product, lot='uff', maxiter=200)
        opt_adj = table_generator(elements=target_product.elements, geometry=opt_geo)

        diff = opt_adj - target_product.adj_mat
        assert np.all(diff == 0)
