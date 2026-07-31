"""Load the trained enthalpy checkpoint and predict DH for reactions.

Checkpoint (egat_dh.pth) stores: model state, cfg (architecture), and the
training target standardization (y_mean, y_std). Predictions are denormalized
back to kcal/mol.

CPU performance (no change to numerics / fp32):
  * featurization is parallelized across worker processes via a DataLoader and
    overlapped with the model forward pass (set num_workers > 0);
  * the matmul thread count is the caller's responsibility (torch.set_num_threads
    in the entrypoint) -- ~8 is the sweet spot for these small graphs.
"""

import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rxn_egat.featurize import featurize_reaction
from rxn_egat.model import EGATReactionNet


def load_model(checkpoint_path, device="cpu"):
    """Returns (model, y_mean, y_std). model is in eval mode."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = EGATReactionNet(
        embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"], num_layers=cfg["num_layers"],
        mlp_dim=cfg["mlp_dim"], dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, float(ckpt["y_mean"]), float(ckpt["y_std"])


def _record_to_graphs(rec, device="cpu"):
    u = torch.from_numpy(rec["u"])
    v = torch.from_numpy(rec["v"])
    Z = torch.from_numpy(rec["Z"])
    gR = dgl.graph((u, v), num_nodes=len(Z))
    gP = dgl.graph((u, v), num_nodes=len(Z))
    gR.ndata["Z"] = Z
    gP.ndata["Z"] = Z
    gR.ndata["feat"] = torch.from_numpy(rec["x_R"])
    gP.ndata["feat"] = torch.from_numpy(rec["x_P"])
    gR.edata["feat"] = torch.from_numpy(rec["e_R"])
    gP.edata["feat"] = torch.from_numpy(rec["e_P"])
    if device != "cpu":
        gR, gP = gR.to(device), gP.to(device)
    return gR, gP


class _InferDataset(Dataset):
    """Featurizes one reaction per item; runs inside DataLoader workers."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        r, p = self.pairs[i]
        try:
            return i, featurize_reaction(r, p)
        except Exception:
            return i, None


def _collate(batch):
    idxs, gRs, gPs = [], [], []
    for i, rec in batch:
        if rec is None:
            continue
        gR, gP = _record_to_graphs(rec)
        idxs.append(i)
        gRs.append(gR)
        gPs.append(gP)
    if not gRs:
        return [], None, None
    return idxs, dgl.batch(gRs), dgl.batch(gPs)


@torch.inference_mode()
def predict_reactions(model, y_mean, y_std, pairs, device="cpu",
                      batch_size=512, num_workers=0):
    """pairs: list of (reactant_smiles, product_smiles). Returns list of floats
    (DH in kcal/mol); NaN where featurization fails.

    num_workers > 0 parallelizes featurization and overlaps it with inference.
    Numerics are unchanged (fp32) regardless of num_workers / thread count.
    """
    preds = [float("nan")] * len(pairs)

    if num_workers and num_workers > 0:
        loader = DataLoader(
            _InferDataset(pairs), batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=_collate,
            persistent_workers=False, prefetch_factor=4,
        )
        for idxs, gR, gP in loader:
            if gR is None:
                continue
            if device != "cpu":
                gR, gP = gR.to(device), gP.to(device)
            out = model(gR, gP).cpu().numpy().ravel() * y_std + y_mean
            for k, i in enumerate(idxs):
                preds[i] = float(out[k])
        return preds

    # sequential fallback (num_workers == 0)
    recs, valid_idx = [], []
    for i, (r, p) in enumerate(pairs):
        try:
            recs.append(featurize_reaction(r, p))
            valid_idx.append(i)
        except Exception:
            pass
    for start in range(0, len(recs), batch_size):
        chunk = recs[start:start + batch_size]
        gRs, gPs = zip(*(_record_to_graphs(rec, device) for rec in chunk))
        out = model(dgl.batch(gRs), dgl.batch(gPs)).cpu().numpy().ravel()
        out = out * y_std + y_mean
        for k, val in enumerate(out):
            preds[valid_idx[start + k]] = float(val)
    return preds
