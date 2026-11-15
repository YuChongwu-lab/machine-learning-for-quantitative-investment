"""Cross-sectional trading signal, correlation, and the auxiliary prediction loss."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data


def _compute_signal(preds: torch.Tensor, risks: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Produces a cross-sectional z-score signal, optionally risk-adjusting the predictions.
    Args:
      preds: [B, A] or [A], each asset's raw score
      risks: same shape as preds, or None; if given, computes preds / (risks+eps)
    Returns:
      s: same shape as preds, z-scored cross-sectionally
    Used by:
      - training/hyperparameter search: Spearman-IC is computed directly from s
      - the backtest: softmax(s/temperature) gives the allocation weights
    """
    if risks is not None:
        s = preds / (risks + 1e-8)
    else:
        s = preds

    # z-score each sample's cross section (de-mean/divide-by-std along the asset dim A)
    if s.dim() == 2:  # [B, A]
        s = (s - s.mean(dim=1, keepdim=True)) / (s.std(dim=1, keepdim=True) + 1e-8)
    else:            # [A]
        s = (s - s.mean()) / (s.std() + 1e-8)
    return s


def _pearson_corr(x: torch.Tensor, y: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    Differentiable Pearson correlation along dim.
    Args:
      x, y: same shape (e.g. [B, A])
      dim: which dimension to correlate over (default: the last one)
    Returns:
      corr: dim reduced away (e.g. input [B,A], dim=1 -> output [B])
    """
    if dim != -1:
        x = x.transpose(dim, -1)  # move the correlated dim to the end
        y = y.transpose(dim, -1)
    xm = x - x.mean(dim=-1, keepdim=True)
    ym = y - y.mean(dim=-1, keepdim=True)
    num = (xm * ym).sum(dim=-1)  # numerator: (unnormalized) covariance
    den = xm.norm(dim=-1) * ym.norm(dim=-1) + eps  # denominator: product of norms + eps
    return num / den


def mse_ic_loss(preds: torch.Tensor, targets: torch.Tensor, risks: torch.Tensor = None,
                alpha: float = 1.0, beta: float  = 0.5, eps: float = 1e-8) -> torch.Tensor:
    """
    Combined loss = alpha*MSE + beta*(-Pearson-IC), encouraging magnitude fit and
    cross-sectional correlation to improve together.
    Args:
      preds:   [B, A] model predictions
      targets: [B, A] supervision target (e.g. next-day return)
      risks:   [B, A] or None; if given, used by _compute_signal's risk adjustment
      alpha, beta: loss weights
    Returns:
      total: scalar total loss
      logs:  dict with {'ic': mean IC (no grad), 'mse': MSE (no grad)}
    """
    # 1) the cross-sectional signal (risk-adjusted + z-scored), used for IC
    s = _compute_signal(preds, risks)                 # [B,A]

    # 2) Pearson-IC (differentiable), correlated along the asset dim, one IC per sample, negated for the loss
    ic = _pearson_corr(s, targets, dim=1)             # [B]
    ic_loss = (-ic).mean()                            # scalar

    # 3) MSE: fits the raw preds directly to targets
    mse = torch.mean((preds - targets)**2)

    # 4) combine
    total = alpha * mse + beta * ic_loss
    return total, {'ic': ic.mean().detach(), 'mse': mse.detach()}


@torch.no_grad()
def evaluate_ic(model: nn.Module, dataset: List[Data], batch_size: int = 16) -> Dict[str, float]:
    """
    Evaluates the average daily Spearman-IC (numpy/pandas convention).
    Args:
      model: a callable PyTorch model whose forward returns a dict containing 'predictions':[B,A]
      dataset: List[Data], each Data.y is [A]
      batch_size: batch size
    Returns:
      {'ic_mean': float, 'n': int}: the mean IC and the number of valid samples
    """
    model.eval()
    ics: List[float] = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        try:
            out = model(batch)
            preds = out['predictions'].detach().cpu().numpy()  # [B,A]
            ys = np.stack([d.y.detach().cpu().numpy() for d in batch])  # [B,A]
        except Exception:
            continue
        # per-day Spearman-IC (pandas average rank)
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
