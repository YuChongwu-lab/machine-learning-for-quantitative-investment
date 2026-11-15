"""The PPO policy network: E_DyGFormer as the encoder, feeding an Actor/Critic head."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from torch_geometric.data import Data

from model import E_DyGFormer


class PolicyNet(nn.Module):
    """Graph-RL policy:
    - encoder: E_DyGFormer.encode_tokens() -> asset_tokens[A,H], cls[H]
    - Actor: outputs a Dirichlet concentration alpha_i > 0 per asset token
    - Critic: outputs the state value V(s) from the CLS token (or pooled asset tokens)
    """
    def __init__(self, encoder: 'E_DyGFormer', hidden: int = 256,
                 alpha_scale: float = 1.0, alpha_min: float = 1e-3,
                 temperature: float = 1.5, max_pos: Optional[float] = 0.2,
                 use_cls_for_value: bool = True, freeze_encoder: bool = False):
        super().__init__()
        self.enc = encoder
        self.A = encoder.num_assets
        self.H = encoder.hidden_dim
        self.temperature = float(temperature)  # temperature tau, for softmax smoothing
        self.max_pos = max_pos   # the max weight for a single asset (used for the projection)
        self.alpha_scale = float(alpha_scale)    # scaling factor on the Dirichlet alpha
        self.alpha_min = float(alpha_min)   # the minimum bound on the Dirichlet alpha, keeps it > 0
        self.use_cls_for_value = use_cls_for_value  # whether the critic uses the CLS token as input
        self.pred_head = nn.Linear(self.H, 1)  # a per-asset prediction head (regresses the next-period cross-sectional return)

        # Actor: takes each asset's token (dim H) and outputs a scalar, then
        # transformed to alpha_i > 0 via softplus
        self.actor = nn.Sequential(
            nn.Linear(self.H, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        # Critic: estimates the state value V(s); input dim matches the token/CLS dim (H)
        value_in = self.H   # using an H-dimensional input here
        self.critic = nn.Sequential(
            nn.Linear(value_in, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        if freeze_encoder:      # optional: freeze the encoder's parameters (train only the policy heads)
            for p in self.enc.parameters():
                p.requires_grad = False

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        asset_tokens, cls_token = self.enc.encode_tokens(data)  # [A,H], [H] or None

        # Actor
        raw = self.actor(asset_tokens).squeeze(-1)
        alpha = F.softplus(raw).clamp(max=20.0) * self.alpha_scale + self.alpha_min  # smooth the alpha values
        logits_det = raw.clone()

        # Critic -- compatible with iTransformer (no CLS)
        if self.use_cls_for_value and cls_token is not None:
            v_in = cls_token
        else:
            v_in = asset_tokens.mean(dim=0)  # iTransformer always uses the asset-token mean

        V = self.critic(v_in)
        return alpha, V.squeeze(-1), logits_det

    def predict_next_return(self, data: Data) -> torch.Tensor:
        """
        Uses the shared encoder to get each asset's token, then regresses the
        next-period return via pred_head.
        Returns: [A]
        """
        asset_tokens, _ = self.enc.encode_tokens(data)  # [A,H]
        return self.pred_head(asset_tokens).squeeze(-1)  # [A]

    @torch.no_grad()  # no gradient tracking for the forward pass (inference/sampling stage)
    def act(self, data: Data, stochastic: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate an action (weights).
        Returns: action_raw[A], action_proj[A], logp, V
        - if stochastic=True: sample from Dirichlet; otherwise use softmax(logits/tau)
        - action_proj: projected onto the capped simplex (sum=1, each dim <= max_pos)
        - logp: the log-probability computed from the unprojected action_raw (an approximation)
        """
        device = next(self.parameters()).device
        alpha, V, logits_det = self.forward(data)
        if stochastic:                  # stochastic policy: sample from Dirichlet(alpha)
            dist = Dirichlet(alpha)     # build the Dirichlet distribution
            a_raw = dist.sample()       # sample weights (sum to 1, non-negative) [A]
            logp = dist.log_prob(a_raw)  # the log-probability of this sample, a scalar
            a_proj = self._project_to_capped_simplex(a_raw, self.max_pos)    # project onto the capped simplex
        else:                           # deterministic policy: use softmax(logits/tau)
            z = logits_det / max(1e-6, self.temperature)    # temperature scaling, avoids division by zero
            a_raw = torch.softmax(z, dim=-1)                # the deterministic "weight sample" [A]
            logp = torch.zeros((), device=device)           # no sampling log-probability; set to 0 (placeholder)
            a_proj = self._project_to_capped_simplex(a_raw, self.max_pos)
        return a_raw, a_proj, logp, V

    def weights_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Deterministic weights (used for backtest evaluation)."""
        z = logits / max(1e-6, self.temperature)
        w = torch.softmax(z, dim=-1)     # convert scores to probability weights
        return self._project_to_capped_simplex(w, self.max_pos)  # project onto the capped simplex

    @staticmethod
    def _project_to_capped_simplex(w: torch.Tensor, cap: Optional[float]) -> torch.Tensor:
        """Project onto the simplex { y>=0, sum y=1, y_i <= cap }.
        Uses an iterative "water-filling" approximation: numerically stable and fast enough.
        """
        w = torch.clamp(w, min=0)   # first clip to non-negative
        if cap is None:              # if there's no cap, just normalize to sum to 1
            s = w.sum()
            return (w / (s + 1e-8)) if s > 0 else torch.full_like(w, 1.0 / w.numel())
        x = w.clone()               # copy for iteration
        A = x.numel()               # number of assets
        cap_val = float(cap)
        for _ in range(10):         # at most 10 water-filling iterations
            x = torch.clamp(x, max=cap_val)  # clip components above the cap
            s = x.sum()
            if s.abs() < 1e-8:           # all zero (an extreme case); fall back to equal weight
                x = torch.full_like(x, 1.0 / A)
                break
            diff = 1.0 - s.item()        # the gap to the target sum of 1
            if abs(diff) < 1e-6:         # already close enough to 1; stop
                break
            # only redistribute mass across components that haven't hit the cap
            # (the "flowable slots" in water-filling)
            mask = (x < cap_val - 1e-12).float()     # a mask of dimensions not yet at the cap
            denom = mask.sum().item()                # the number of available "slots"
            if denom <= 0:                           # everything is capped: just rescale to sum to 1
                # everything is capped, so scale proportionally to sum to 1
                x = x / (s + 1e-8)
                break
            x = x + (diff / (denom + 1e-8)) * mask    # distribute/reclaim mass evenly across the uncapped dims
        # one final normalization
        x = torch.clamp(x, min=0.0, max=cap_val)      # make sure it's still within bounds
        x = x / (x.sum() + 1e-8)                      # normalize to a probability distribution
        return x                                      # return the projected weights
