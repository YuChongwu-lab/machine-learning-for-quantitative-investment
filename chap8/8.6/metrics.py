"""Cross-sectional trading signal, rank/correlation utilities, and the training loss.

Shared by training, hyperparameter search, and evaluation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data


def _compute_signal(preds: torch.Tensor, risks: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Args:
      preds: [B, A] or [A], a score per asset
      risks: same shape, or None; if given, the signal is risk-adjusted
    Returns:
      s: a cross-sectional z-score signal, same shape as preds (per sample)
    Notes:
      - training/search computes Spearman-IC directly from s
      - the backtest applies softmax(s/temperature) to s to get weights
    """
    if risks is not None:
        s = preds / (risks + 1e-8)
    else:
        s = preds
    if s.dim() == 2:
        s = (s - s.mean(dim=1, keepdim=True)) / (s.std(dim=1, keepdim=True) + 1e-8)
    else:
        s = (s - s.mean()) / (s.std() + 1e-8)
    return s


def _rankdata_average_1d(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    order = torch.argsort(x, stable=True)
    xs = x[order]
    # find where adjacent values differ -> segment into tie groups
    diff = torch.ones(n, dtype=torch.bool, device=x.device)
    diff[1:] = xs[1:] != xs[:-1]
    gid = torch.cumsum(diff, dim=0) - 1                     # 0..G-1
    pos = torch.arange(n, device=x.device, dtype=torch.float32)

    G = int(gid[-1].item()) + 1 if n > 0 else 0
    first = torch.full((G,), float('inf'), device=x.device)
    last  = torch.full((G,), float('-inf'), device=x.device)
    # requires PyTorch 1.13+: scatter_reduce_
    first = first.scatter_reduce(0, gid, pos, reduce='amin', include_self=True)
    last  = last.scatter_reduce(0, gid, pos, reduce='amax', include_self=True)

    avg = (first + last) / 2.0 + 1.0                        # 1-based
    ranks_sorted = avg[gid]                                  # map back to each element
    ranks = torch.empty(n, device=x.device, dtype=torch.float32)
    ranks[order] = ranks_sorted
    return ranks


def _pearson_corr(x: torch.Tensor, y: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    Pearson correlation along dim; x, y have the same shape (e.g. [B, A]),
    returns a tensor with that dimension reduced away (e.g. [B]).
    """
    if dim != -1:
        x = x.transpose(dim, -1)
        y = y.transpose(dim, -1)
    xm = x - x.mean(dim=-1, keepdim=True)
    ym = y - y.mean(dim=-1, keepdim=True)
    num = (xm * ym).sum(dim=-1)
    den = xm.norm(dim=-1) * ym.norm(dim=-1) + eps
    return num / den


def mse_ic_loss(preds: torch.Tensor,
                targets: torch.Tensor,
                risks: torch.Tensor = None,
                alpha: float = 1.0,
                beta: float  = 0.5,
                eps: float = 1e-8) -> torch.Tensor:
    """
    preds/targets: [B, A]
    risks:         [B, A] or None
    """
    # 1) the cross-sectional signal (risk-adjusted + z-scored), used for IC
    s = _compute_signal(preds, risks)                 # [B,A], already z-scored

    # 2) Pearson-IC (differentiable)
    ic = _pearson_corr(s, targets, dim=1)             # [B]
    ic_loss = (-ic).mean()                            # maximize IC -> negate

    # 3) MSE (could use raw preds or the risk-adjusted preds/(risks+eps); raw preds is typical)
    mse = torch.mean((preds - targets)**2)            # matches magnitude; risk-adjusted would be mse = torch.mean((s - zscore(targets))**2)

    # 4) combine
    total = alpha * mse + beta * ic_loss
    return total, {'ic': ic.mean().detach(), 'mse': mse.detach()}


def _spearman_corr(x: torch.Tensor, y: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    Spearman correlation along dim: average-rank first, then Pearson.
    x, y have the same shape (e.g. [B, A]); returns a tensor shaped like [B].
    """
    # flatten to 2D for the loop (rank only the correlated dimension)
    if dim != -1:
        x = x.transpose(dim, -1)
        y = y.transpose(dim, -1)
    B = x.shape[:-1].numel()
    A = x.shape[-1]
    xr = x.reshape(B, A).contiguous()
    yr = y.reshape(B, A).contiguous()

    # average rank, per sample
    for i in range(B):
        xr[i] = _rankdata_average_1d(xr[i])
        yr[i] = _rankdata_average_1d(yr[i])

    # Pearson on ranks
    xm = xr - xr.mean(dim=1, keepdim=True)
    ym = yr - yr.mean(dim=1, keepdim=True)
    num = (xm * ym).sum(dim=1)
    den = xm.norm(dim=1) * ym.norm(dim=1) + eps
    corr = num / den

    if dim != -1:
        pass
    return corr


# =========================
# Metric helper: cross-sectional Spearman IC
# =========================
@torch.no_grad()
def evaluate_ic(model: nn.Module, dataset: List[Data], batch_size: int = 16) -> Dict[str, float]:
    model.eval()
    ics: List[float] = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        try:
            out = model(batch)
            preds = out['predictions'].detach().cpu().numpy()  # [B,A]
            ys    = np.stack([d.y.detach().cpu().numpy() for d in batch])  # [B,A]
        except Exception:
            continue
        for p, y in zip(preds, ys):
            if not (np.isfinite(p).all() and np.isfinite(y).all()):
                continue
            if np.std(p)==0 or np.std(y)==0:
                continue
            rx = pd.Series(p).rank(method='average').to_numpy()
            ry = pd.Series(y).rank(method='average').to_numpy()
            ic = np.corrcoef(rx, ry)[0, 1]
            if np.isfinite(ic):
                ics.append(float(ic))
    return {"ic_mean": float(np.mean(ics)) if ics else 0.0, "n": int(len(ics))}


@torch.no_grad()
def _eval_ic_with_risk(model: nn.Module, dataset: List[Data]) -> Dict[str, float]:
    model.eval(); ics=[]
    for d in dataset:
        out = model(d)
        p = out['predictions'][0].detach()
        r = out.get('risks', [None])[0]
        r = r.detach() if r is not None else (p.abs() + 1e-8)
        s = _compute_signal(p, r).cpu().numpy()
        y = d.y.detach().cpu().numpy()
        rs = pd.Series(s).rank(method='average').to_numpy()
        ry = pd.Series(y).rank(method='average').to_numpy()
        if np.std(rs) > 1e-12 and np.std(ry) > 1e-12:
            ics.append(np.corrcoef(rs, ry)[0, 1])
    return {"ic_mean": float(np.mean(ics)) if ics else 0.0, "n": int(len(ics))}
