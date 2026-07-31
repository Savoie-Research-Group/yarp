#!/usr/bin/env python3
"""
Load compiled EGAT activation model. Self-contained, no external yarp deps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import dgl


class EGATConv(nn.Module):
    """Standalone EGATConv implementation."""

    def __init__(
        self,
        in_node_feats,
        in_edge_feats,
        out_node_feats,
        out_edge_feats,
        num_heads,
        **kw_args,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self.fc_nodes = nn.Linear(in_node_feats, out_node_feats * num_heads, bias=True)
        self.fc_edges = nn.Linear(
            in_edge_feats + 2 * in_node_feats, out_edge_feats * num_heads, bias=True
        )
        self.fc_attn = nn.Linear(out_edge_feats, num_heads, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = init.calculate_gain("relu")
        init.xavier_normal_(self.fc_nodes.weight, gain=gain)
        init.xavier_normal_(self.fc_edges.weight, gain=gain)
        init.xavier_normal_(self.fc_attn.weight, gain=gain)

    def edge_attention(self, edges):
        h_src = edges.src["h"]
        h_dst = edges.dst["h"]
        f = edges.data["f"]
        stack = torch.cat([h_src, f, h_dst], dim=-1)
        f_out = self.fc_edges(stack)
        f_out = F.leaky_relu(f_out)
        f_out = f_out.view(-1, self._num_heads, self._out_edge_feats)
        a = self.fc_attn(f_out).sum(-1).unsqueeze(-1)
        return {"a": a, "f": f_out}

    def message_func(self, edges):
        return {"h": edges.src["h"], "a": edges.data["a"]}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox["a"], dim=1)
        h = torch.sum(alpha * nodes.mailbox["h"], dim=1)
        return {"h": h}

    def forward(self, graph, nfeats, efeats):
        with graph.local_scope():
            graph.edata["f"] = efeats
            graph.ndata["h"] = nfeats
            graph.apply_edges(self.edge_attention)
            nfeats_ = self.fc_nodes(nfeats)
            nfeats_ = nfeats_.view(-1, self._num_heads, self._out_node_feats)
            graph.ndata["h"] = nfeats_
            graph.update_all(
                message_func=self.message_func, reduce_func=self.reduce_func
            )
            return graph.ndata.pop("h"), graph.edata.pop("f")


class EGAT_Rxn_Standalone(nn.Module):
    """Standalone EGAT_Rxn model for activation barrier."""

    def __init__(self, config_dict):
        super().__init__()
        self.num_node_feats = config_dict["num_node_feats"]
        self.num_edge_feats = config_dict["num_edge_feats"]
        self.hidden_dim = config_dict["hidden_dim"]
        self.num_heads = config_dict["num_heads"]
        self.config = config_dict

        self.egat1 = EGATConv(
            in_node_feats=self.num_node_feats,
            in_edge_feats=self.num_edge_feats,
            out_node_feats=self.hidden_dim,
            out_edge_feats=self.hidden_dim,
            num_heads=self.num_heads,
        )
        self.egat2 = EGATConv(
            in_node_feats=self.hidden_dim * self.num_heads,
            in_edge_feats=self.hidden_dim * self.num_heads,
            out_node_feats=self.hidden_dim,
            out_edge_feats=self.hidden_dim,
            num_heads=self.num_heads,
        )
        self.agg_N_feats = nn.Sequential(
            nn.Linear(
                self.hidden_dim * self.num_heads,
                self.hidden_dim * self.num_heads,
                bias=True,
            ),
            nn.GELU(),
        )
        self.agg_E_feats = nn.Sequential(
            nn.Linear(
                self.hidden_dim * self.num_heads,
                self.hidden_dim * self.num_heads,
                bias=True,
            ),
            nn.GELU(),
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_heads * 2, 256, bias=True),
            nn.GELU(),
        )
        self.mlp2 = nn.Sequential(nn.Linear(256, 128, bias=True), nn.GELU())
        self.mlp3 = nn.Linear(128, 1, bias=True)

    def forward(self, graphR, graphP):
        Rnode_feats, Redge_feats = self.egat1(
            graphR, graphR.ndata["x"], graphR.edata["x"]
        )
        Rnode_feats = Rnode_feats.view(
            graphR.number_of_nodes(), self.hidden_dim * self.num_heads
        )
        Redge_feats = Redge_feats.view(
            graphR.number_of_edges(), self.hidden_dim * self.num_heads
        )
        Pnode_feats, Pedge_feats = self.egat1(
            graphP, graphP.ndata["x"], graphP.edata["x"]
        )
        Pnode_feats = Pnode_feats.view(
            graphP.number_of_nodes(), self.hidden_dim * self.num_heads
        )
        Pedge_feats = Pedge_feats.view(
            graphP.number_of_edges(), self.hidden_dim * self.num_heads
        )

        for _ in range(3):
            Rnode_feats, Redge_feats = self.egat2(graphR, Rnode_feats, Redge_feats)
            Rnode_feats = Rnode_feats.view(
                graphR.number_of_nodes(), self.hidden_dim * self.num_heads
            )
            Redge_feats = Redge_feats.view(
                graphR.number_of_edges(), self.hidden_dim * self.num_heads
            )
            Pnode_feats, Pedge_feats = self.egat2(graphP, Pnode_feats, Pedge_feats)
            Pnode_feats = Pnode_feats.view(
                graphP.number_of_nodes(), self.hidden_dim * self.num_heads
            )
            Pedge_feats = Pedge_feats.view(
                graphP.number_of_edges(), self.hidden_dim * self.num_heads
            )

        Rxn_node_feature = self.agg_N_feats(Pnode_feats - Rnode_feats)
        Rxn_edge_feature = self.agg_E_feats(Pedge_feats - Redge_feats)
        graphR.ndata["x"] = Rxn_node_feature
        graphR.edata["x"] = Rxn_edge_feature
        individual_graphs = dgl.unbatch(graphR)
        G_node_feats = []
        G_edge_feats = []
        for graph in individual_graphs:
            G_node_feats.append(graph.ndata["x"].sum(dim=0))
            G_edge_feats.append(graph.edata["x"].sum(dim=0))
        G_node_feats = torch.stack(G_node_feats)
        G_edge_feats = torch.stack(G_edge_feats)
        G_features = torch.cat((G_node_feats, G_edge_feats), axis=1)
        x = self.mlp1(G_features)
        x = self.mlp2(x)
        x = self.mlp3(x)
        return x


def load_compiled_model(model_path):
    """Load compiled activation model. Returns (model, config_dict)."""
    print(f"Loading compiled model from: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if "config" in checkpoint:
        config = checkpoint["config"]
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        egat1 = state_dict.get("egat1.fc_nodes.weight")
        if egat1 is None:
            raise ValueError("Checkpoint missing model weights.")
        num_node_feats = egat1.shape[1]
        num_edge_feats = state_dict["egat1.fc_edges.weight"].shape[1] - 2 * num_node_feats
        num_heads = state_dict["egat1.fc_attn.weight"].shape[0]
        hidden_dim = state_dict["egat1.fc_attn.weight"].shape[1]
        config = {
            "num_node_feats": num_node_feats,
            "num_edge_feats": num_edge_feats,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
        }
    model = EGAT_Rxn_Standalone(config)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")
    print(f"Config: {config}")
    print(f"Expected node feature size: {config['num_node_feats']}")
    print(f"Expected edge feature size: {config['num_edge_feats']}")
    return model, config
