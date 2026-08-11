"""Clean EGAT reaction model for enthalpy (DH) regression.

Differences from the legacy model:
  * Atomic number enters via an nn.Embedding table (index = atomic number),
    not as a raw integer feature.
  * LayerNorm after every message-passing layer for stable training.
  * Plain sum-readout via dgl.sum_nodes/sum_edges (no Python unbatch loop).
  * Single scalar output, unbounded -> predicts both +DH and -DH.
"""

import dgl
import torch
import torch.nn as nn

from rxn_egat.egat_conv import EGATConv
from rxn_egat.featurize import MAX_Z, NODE_FEAT_DIM, EDGE_FEAT_DIM


class EGATReactionNet(nn.Module):
    def __init__(self, embed_dim=32, hidden_dim=128, num_heads=4,
                 num_layers=3, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embed = nn.Embedding(MAX_Z + 1, embed_dim, padding_idx=0)

        in_node = embed_dim + NODE_FEAT_DIM
        in_edge = EDGE_FEAT_DIM
        hd = hidden_dim * num_heads

        self.layers = nn.ModuleList()
        self.node_norms = nn.ModuleList()
        self.edge_norms = nn.ModuleList()
        for li in range(num_layers):
            nf = in_node if li == 0 else hd
            ef = in_edge if li == 0 else hd
            self.layers.append(EGATConv(
                in_node_feats=nf, in_edge_feats=ef,
                out_node_feats=hidden_dim, out_edge_feats=hidden_dim,
                num_heads=num_heads))
            self.node_norms.append(nn.LayerNorm(hd))
            self.edge_norms.append(nn.LayerNorm(hd))

        # combine reactant/product differences
        self.agg_node = nn.Sequential(nn.Linear(hd, hd), nn.GELU())
        self.agg_edge = nn.Sequential(nn.Linear(hd, hd), nn.GELU())

        self.head = nn.Sequential(
            nn.Linear(2 * hd, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, 1),
        )

    def _encode(self, g):
        """Run the shared EGAT stack on one graph; returns (node_feats, edge_feats)."""
        h = torch.cat([self.embed(g.ndata["Z"]), g.ndata["feat"]], dim=1)
        e = g.edata["feat"]
        for layer, n_norm, e_norm in zip(self.layers, self.node_norms, self.edge_norms):
            h, e = layer(g, h, e)
            h = h.reshape(g.num_nodes(), self.hidden_dim * self.num_heads)
            e = e.reshape(g.num_edges(), self.hidden_dim * self.num_heads)
            h = torch.relu(n_norm(h))
            e = torch.relu(e_norm(e))
        return h, e

    def forward(self, gR, gP):
        hR, eR = self._encode(gR)
        hP, eP = self._encode(gP)

        node_diff = self.agg_node(hP - hR)
        edge_diff = self.agg_edge(eP - eR)

        # sum readout per graph (gR carries the batch structure / topology)
        with gR.local_scope():
            gR.ndata["_n"] = node_diff
            gR.edata["_e"] = edge_diff
            g_node = dgl.sum_nodes(gR, "_n")
            g_edge = dgl.sum_edges(gR, "_e")

        g_feat = torch.cat([g_node, g_edge], dim=1)
        return self.head(g_feat)
