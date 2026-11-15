import torch
import torch.nn as nn
import numpy as np

# -------------------------
# RevIN implementation (same pattern as before)
# -------------------------
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta  = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
        self._stored_mean = None
        self._stored_std  = None

    def forward(self, x, mode: str):
        """
        x: [B, L, C]
        mode: 'norm' or 'denorm'
        """
        if mode == 'norm':
            mean = x.mean(dim=1, keepdim=True)          # [B,1,C]
            std  = x.std(dim=1, keepdim=True) + self.eps
            self._stored_mean = mean
            self._stored_std  = std
            x_norm = (x - mean) / std
            if self.affine:
                x_norm = x_norm * self.gamma + self.beta
            return x_norm

        elif mode == 'denorm':
            if self._stored_mean is None or self._stored_std is None:
                raise RuntimeError("RevIN.denorm called before RevIN.norm in same forward pass.")
            x_hat = x
            if self.affine:
                x_hat = (x_hat - self.beta) / (self.gamma + self.eps)
            x_hat = x_hat * self._stored_std + self._stored_mean
            return x_hat
        else:
            raise ValueError("mode must be 'norm' or 'denorm'")

# -------------------------
# ProbSparse-style attention used by Informer (simplified)
# -------------------------
class ProbAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        # Q,K,V : [B, L, H, D]
        B, Lq, H, D = Q.shape
        _, Lk, _, _ = K.shape
        # scores: [B, H, Lq, Lk] via einsum -> [bhls] in previous code we used [bhls]
        scores = torch.einsum("blhd,bshd->bhls", Q, K) / np.sqrt(D)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhls,bshd->blhd", attn, V)  # [B, Lq, H, D]
        return out.contiguous()

class MultiHeadProbAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn = ProbAttention(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Q,K,V: [B, L, d_model]
        returns: [B, L, d_model]
        """
        B, Lq, _ = Q.shape
        _, Lk, _ = K.shape
        # project and split heads
        Q = self.q_proj(Q).view(B, Lq, self.n_heads, self.d_k)  # [B, Lq, H, d_k]
        K = self.k_proj(K).view(B, Lk, self.n_heads, self.d_k)
        V = self.v_proj(V).view(B, Lk, self.n_heads, self.d_k)

        out = self.attn(Q, K, V, mask)  # [B, Lq, H, d_k]
        out = out.view(B, Lq, -1)       # [B, Lq, d_model]
        return self.o_proj(out)

# -------------------------
# Encoder layer (Informer-style)
# -------------------------
class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadProbAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: [B, L, d_model]
        attn_out = self.attn(x, x, x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x

# -------------------------
# Informer core (encoder-only simplified)
# -------------------------
class InformerCore(nn.Module):
    def __init__(self, input_size, d_model, n_heads, num_layers, dropout, output_size):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff=4*d_model, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, output_size)

        # init
        nn.init.xavier_uniform_(self.input_proj.weight); nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.fc_out.weight);     nn.init.zeros_(self.fc_out.bias)

    def forward(self, x, mask=None):
        """
        x: [B, L, C] (assumes already normalized when called under RevIN)
        returns: [B, output_size] in the same (normalized) space
        """
        h = self.input_proj(x)  # [B, L, d_model]
        for layer in self.encoder_layers:
            h = layer(h, mask)
        out = self.fc_out(h[:, -1, :])  # last time-step representation
        return out

# -------------------------
# Wrapper: Informer + RevIN
# -------------------------
class InformerRevIN(nn.Module):
    """
    Informer + RevIN wrapper.
    __init__(input_size, d_model, num_heads, num_layers, dropout, output_size)
    Behavior:
      - If output_size == input_size: returns denormalized [B, C] (original scale)
      - Else: returns standardized-space preds [B, output_size] (caller should inverse via scaler)
    """
    def __init__(self, input_size, d_model, num_heads, num_layers, dropout, output_size, affine=True):
        super().__init__()
        self.revin = RevIN(num_features=input_size, affine=affine)
        self.core = InformerCore(input_size, d_model, num_heads, num_layers, dropout, output_size)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        """
        x: [B, L, C]
        returns:
          - if output_size == input_size -> denormed [B, C]
          - else -> normalized-space preds [B, output_size]
        """
        # 1) normalize
        x_norm = self.revin(x, mode='norm')  # stores mean/std

        # 2) core predictor in normalized space
        y_norm = self.core(x_norm)           # [B, output_size]

        # 3) denorm only when sizes match
        if self.output_size == self.input_size:
            y_exp = y_norm.unsqueeze(1)      # [B,1,C]
            y_denorm = self.revin(y_exp, mode='denorm')  # [B,1,C]
            return y_denorm.squeeze(1)       # [B,C]
        else:
            return y_norm
