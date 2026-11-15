"""The edge-aware dynamic graph Transformer (E-DyGFormer)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):  # [S, B, E]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EdgeAwareMHSA(nn.Module):
    """
    Multi-head self-attention (supports an edge bias + optional signed aggregation).
    - edge_bias_2d: [S, S] additive bias
    - use_signed_agg=True: multiply the softmax probabilities by sign(edge_bias)
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, use_signed_agg: bool = True):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.use_signed_agg = use_signed_agg

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        src: torch.Tensor,                       # [S, B, H]
        edge_bias_2d: Optional[torch.Tensor] = None,  # [S, S] or None
        need_weights: bool = False
    ):
        S, B, H = src.shape
        device = src.device

        q = self.q_proj(src)  # [S,B,H]
        k = self.k_proj(src)
        v = self.v_proj(src)

        def reshape_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(S, B, self.nhead, self.head_dim).permute(1, 2, 0, 3)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,nH,S,S]

        sign_gate = None
        if edge_bias_2d is not None:
            eb = edge_bias_2d.to(device)  # [S,S]
            attn_logits = attn_logits + eb.unsqueeze(0).unsqueeze(0)  # additive bias
            if self.use_signed_agg:
                sign_gate = torch.sign(eb)
                sign_gate = torch.where(sign_gate == 0, torch.ones_like(sign_gate), sign_gate)
                sign_gate = sign_gate.unsqueeze(0).unsqueeze(0)  # [1,1,S,S]

        attn = torch.softmax(attn_logits, dim=-1)  # [B,nH,S,S]
        attn = self.attn_drop(attn)

        if sign_gate is not None:
            attn = attn * sign_gate

        out = torch.matmul(attn, v)  # [B,nH,S,head_dim]
        out = out.permute(2, 0, 1, 3).contiguous().view(S, B, H)  # [S,B,H]
        out = self.out_proj(out)
        out = self.proj_drop(out)

        attn_to_return = attn.mean(dim=1) if need_weights else None  # [B,S,S] or None
        return out, attn_to_return


class EdgeAwareEncoderLayer(nn.Module):
    """Transformer encoder layer."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float,
                 use_signed_agg: bool = False):
        super().__init__()
        self.mhsa = EdgeAwareMHSA(d_model, nhead, dropout, use_signed_agg=use_signed_agg)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.act = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor,
                edge_bias: Optional[torch.Tensor] = None,
                need_weights: bool = False):
        attn_out, attn = self.mhsa(src, edge_bias_2d=edge_bias, need_weights=need_weights)  # [S,B,H], [B,S,S]
        src = self.norm1(src + self.drop1(attn_out))

        ff = self.linear2(self.drop2(self.act(self.linear1(src))))
        src = self.norm2(src + self.drop3(ff))
        return src, attn


class E_DyGFormer(nn.Module):
    """
    Edge-aware Dynamic Graph Transformer (E-DyGFormer)
    """

    def __init__(self, node_feat_dim=64, hidden_dim=128,
                 num_layers=6, num_heads=8, max_seq_len=8192, dropout=0.1,
                 num_assets=50, seq_len=20,
                 edge_bias_scale=0.3, edge_bias_temp=2.0,  # edge-bias parameters
                 use_signed_agg: bool = False,            # signed aggregation off by default
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_assets = num_assets
        self.seq_len = seq_len
        self.edge_bias_scale = float(edge_bias_scale)
        self.edge_bias_temp  = float(edge_bias_temp)

        self.node_embedding = nn.Linear(node_feat_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout, max_len=max_seq_len)

        self.layers = nn.ModuleList([
            EdgeAwareEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                use_signed_agg=use_signed_agg
            ) for _ in range(num_layers)
        ])

        self.cls_token = nn.Parameter(torch.randn(1, hidden_dim))

        self.asset_head = nn.Linear(hidden_dim, 1)      # prediction score
        self.asset_risk_head = nn.Linear(hidden_dim, 1) # risk (positive after softplus)

    def _build_edge_bias(self, data, S_total: int, device: torch.device) -> Optional[torch.Tensor]:
        if getattr(data, 'edge_index', None) is None or data.edge_index.numel() == 0:
            return None

        bias = torch.zeros(S_total, S_total, device=device, dtype=torch.float32)
        edge_index = data.edge_index.to(device)

        # -- extract the edge weight w --
        if getattr(data, 'edge_attr', None) is not None and data.edge_attr.numel() > 0:
            ea = data.edge_attr.to(device).float()
            if ea.dim() == 1:
                w = ea
            else:
                w = ea.squeeze(-1) if ea.size(-1) == 1 else ea[:, 0]

            if w.numel() != edge_index.size(1):
                w = torch.ones(edge_index.size(1), device=device, dtype=torch.float32)
            else:
                # map the correlation coefficient to an additive bias (tempered)
                eps = 1e-6
                temp = self.edge_bias_temp
                w = w.clamp(-1 + eps, 1 - eps)
                w = torch.atanh(w) / max(1e-6, float(temp))
        else:
            w = torch.ones(edge_index.size(1), device=device, dtype=torch.float32)

        # CLS offset (CLS sits at 0, nodes are all shifted by +1)
        src_nodes = (edge_index[0] + 1).clamp_(0, S_total - 1)
        dst_nodes = (edge_index[1] + 1).clamp_(0, S_total - 1)

        # accumulate the additive bias into the corresponding (i,j) entries
        bias.index_put_((src_nodes, dst_nodes), w * self.edge_bias_scale, accumulate=True)

        # -- DropEdge during training (20% drop, non-zero edges only) --
        if self.training:
            mask = (bias != 0)
            if mask.any():
                keep = torch.rand_like(bias).ge(0.2)  # keep 80%
                bias = torch.where(mask & (~keep), torch.zeros_like(bias), bias)

        return bias

    def forward(self, data_batch, store_attention: bool = False):
        # supports a single sample or a batch (list[Data])
        if isinstance(data_batch, list):
            if len(data_batch) == 0:
                raise ValueError("Empty data batch")

            preds, risks, attns = [], [], []
            for data in data_batch:
                p, r, a = self._forward_single_edgeaware(data, store_attention)
                preds.append(p)
                risks.append(r)
                if a is not None:
                    attns.append(a)

            out = {'predictions': torch.stack(preds), 'risks': torch.stack(risks)}
            if attns:
                out['attention_weights'] = attns
            return out
        else:
            p, r, a = self._forward_single_edgeaware(data_batch, store_attention)
            out = {'predictions': p.unsqueeze(0), 'risks': r.unsqueeze(0)}
            if a is not None:
                out['attention_weights'] = [a]
            return out

    def _forward_single_edgeaware(self, data, store_attention: bool = False):
        device = next(self.parameters()).device

        # 1) length alignment (N = seq_len * num_assets)
        x = data.x.to(device)  # [N, F]
        N = x.size(0)
        expected_N = self.num_assets * self.seq_len
        if N != expected_N:
            if N > expected_N:
                x = x[:expected_N]
            else:
                pad = torch.zeros(expected_N - N, x.size(1), device=device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=0)
            N = expected_N

        # 2) node encoding
        node_emb = self.node_embedding(x)  # [N, H]

        # 3) prepend CLS
        cls = self.cls_token  # [1, H]
        seq = torch.cat([cls, node_emb], dim=0)  # [S=1+N, H]
        h = seq.unsqueeze(1)  # [S, B=1, H]
        h = self.pos_encoding(h)  # [S, 1, H]

        # 4) build the edge bias
        S = 1 + N
        edge_bias_2d = self._build_edge_bias(data, S_total=S, device=device)

        # 5) stack the encoder layers
        attn_list = []
        if store_attention:
            for layer in self.layers:
                h, attn = layer(h, edge_bias=edge_bias_2d, need_weights=True)
                if attn is not None:
                    attn_list.append(attn.detach().cpu())
            attention_info = torch.stack(attn_list) if attn_list else None
        else:
            for layer in self.layers:
                h, _ = layer(h, edge_bias=edge_bias_2d, need_weights=False)
            attention_info = None

        # 6) per-asset readout -- average over the last 3 days to cut single-day noise
        tokens_no_cls = h[1:, 0, :].view(self.seq_len, self.num_assets, -1)  # [T,A,H]
        k = 3 if self.seq_len >= 3 else 1
        last_step_tokens = tokens_no_cls[-k:, :, :].mean(dim=0)              # [A,H]

        asset_scores = self.asset_head(last_step_tokens).squeeze(-1)              # [A]
        asset_risks  = F.softplus(self.asset_risk_head(last_step_tokens)).squeeze(-1)  # [A] >= 0

        return asset_scores, asset_risks, attention_info
