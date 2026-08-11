#!/usr/bin/env python3
"""
Load EGAT enthalpy model from checkpoint. Self-contained, uses egat.model.EGAT_Rxn.
"""

import torch
import omegaconf

from egat.model import EGAT_Rxn


def load_model_from_checkpoint(checkpoint_path, config):
    """
    Load EGAT enthalpy model from checkpoint. Infers architecture from weights.
    Returns (model, checkpoint).
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    egat1_fc_nodes_weight = state_dict.get("egat1.fc_nodes.weight", None)
    if egat1_fc_nodes_weight is None:
        raise ValueError("Checkpoint missing egat1.fc_nodes.weight")
    num_node_feats = egat1_fc_nodes_weight.shape[1]
    hidden_dim_times_heads = egat1_fc_nodes_weight.shape[0]

    egat1_fc_edges_weight = state_dict.get("egat1.fc_edges.weight", None)
    if egat1_fc_edges_weight is None:
        raise ValueError("Checkpoint missing egat1.fc_edges.weight")
    edge_plus_nodes = egat1_fc_edges_weight.shape[1]
    num_edge_feats = edge_plus_nodes - 2 * num_node_feats

    egat1_fc_attn_weight = state_dict.get("egat1.fc_attn.weight", None)
    if egat1_fc_attn_weight is None:
        raise ValueError("Checkpoint missing egat1.fc_attn.weight")
    num_heads = egat1_fc_attn_weight.shape[0]
    hidden_dim = egat1_fc_attn_weight.shape[1]

    config.num_node_feats = num_node_feats
    config.num_edge_feats = num_edge_feats
    config.hidden_dim = hidden_dim
    config.num_heads = num_heads

    model = EGAT_Rxn(config)
    model_state = model.state_dict()
    full_state = {}
    for key, target_tensor in model_state.items():
        if key in state_dict:
            ckpt_tensor = state_dict[key]
            if ckpt_tensor.shape != target_tensor.shape:
                new_tensor = torch.zeros_like(target_tensor)
                common_slices = tuple(
                    slice(0, min(ckpt_tensor.shape[d], target_tensor.shape[d]))
                    for d in range(ckpt_tensor.ndim)
                )
                new_tensor[common_slices] = ckpt_tensor[common_slices]
                full_state[key] = new_tensor
            else:
                full_state[key] = ckpt_tensor
        else:
            full_state[key] = target_tensor

    model.load_state_dict(full_state, strict=True)
    model.eval()
    return model, checkpoint
