"""Wraps the policy as a {predictions, risks}-style "model" so it can reuse
the backtest engine's run_backtest()."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data

from policy import PolicyNet


class PolicyModelAdapter(nn.Module):
    """Wraps the policy into the {predictions, risks} output format expected
    by the original run_backtest().
    - By default: predictions uses the deterministic logits; risks uses
      |logits|+0.01, avoiding a zero denominator for risk.
    - Supports "action smoothing": an EMA over the weights during evaluation,
      to avoid excessive turnover. When smoothed, predictions is set to
      T*log(w_ema) and risks to 1, so that softmax(pred/T) = w_ema.
    """
    def __init__(self, policy: PolicyNet, smooth_weights: bool = False, smooth_rho: float = 0.3):
        super().__init__()
        self.policy = policy
        self.smooth_weights = bool(smooth_weights)
        self.smooth_rho = float(smooth_rho)
        self._ema_w: Optional[torch.Tensor] = None
        self.temperature = float(getattr(policy, 'temperature', 1.5))

    def reset_state(self):
        self._ema_w = None

    @torch.no_grad()
    def forward(self, data: Data):
        _, _, logits = self.policy(data)  # [A]
        if self.smooth_weights:
            T = max(1e-6, self.temperature)
            w = torch.softmax(logits / T, dim=-1)
            if self._ema_w is None:
                self._ema_w = w.clone()
            else:
                self._ema_w = self.smooth_rho * self._ema_w + (1.0 - self.smooth_rho) * w
            w_t = self._ema_w / (self._ema_w.sum() + 1e-8)
            eps = 1e-8
            preds = T * torch.log(w_t + eps)   # makes softmax(preds/T) = w_t
            risks = torch.ones_like(preds)     # avoids the risk denominator distorting the weights
        else:
            preds = logits
            risks = logits.abs() + 0.01        # fixes the zero-risk-denominator issue
        return {'predictions': preds.unsqueeze(0), 'risks': risks.unsqueeze(0)}
