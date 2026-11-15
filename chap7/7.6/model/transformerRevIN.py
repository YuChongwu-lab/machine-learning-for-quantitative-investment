import torch
import torch.nn as nn

# -------------------------
# RevIN (same as before)
# -------------------------
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)
        self._stored_mean = None
        self._stored_std = None

    def forward(self, x, mode: str):
        """
        x: [B, L, C]
        mode: 'norm' or 'denorm'
        """
        if mode == "norm":
            mean = x.mean(dim=1, keepdim=True)         # [B,1,C]
            std = x.std(dim=1, keepdim=True) + self.eps
            self._stored_mean = mean
            self._stored_std = std
            x_norm = (x - mean) / std
            if self.affine:
                x_norm = x_norm * self.gamma + self.beta
            return x_norm
        elif mode == "denorm":
            if self._stored_mean is None or self._stored_std is None:
                raise RuntimeError("RevIN.denorm called before RevIN.norm")
            x_hat = x
            if self.affine:
                x_hat = (x_hat - self.beta) / (self.gamma + self.eps)
            x_hat = x_hat * self._stored_std + self._stored_mean
            return x_hat
        else:
            raise ValueError("mode must be 'norm' or 'denorm'")

# -------------------------
# Standard Transformer Encoder
# -------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
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
        # Multi-head attention
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Feed-forward
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x

class TransformerCore(nn.Module):
    def __init__(self, input_size, d_model, n_heads, num_layers, dropout, output_size):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff=4*d_model, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, output_size)

        # init
        nn.init.xavier_uniform_(self.input_proj.weight); nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.fc_out.weight);     nn.init.zeros_(self.fc_out.bias)

    def forward(self, x, mask=None):
        """
        x: [B, L, C]  (already normalized)
        return: [B, output_size]
        """
        h = self.input_proj(x)  # [B, L, d_model]
        for layer in self.encoder_layers:
            h = layer(h, mask)
        out = self.fc_out(h[:, -1, :])  # last time-step
        return out

# -------------------------
# Transformer + RevIN wrapper
# -------------------------
class TransformerRevIN(nn.Module):
    """
    Transformer + RevIN wrapper.
    __init__(input_size, d_model, num_heads, num_layers, dropout, output_size)
    Behavior:
      - If output_size == input_size: returns denormalized [B, C] (original scale)
      - Else: returns normalized-space preds [B, output_size] (caller should inverse via scaler)
    """
    def __init__(self, input_size, d_model, num_heads, num_layers, dropout, output_size, affine=True):
        super().__init__()
        self.revin = RevIN(num_features=input_size, affine=affine)
        self.core = TransformerCore(input_size, d_model, num_heads, num_layers, dropout, output_size)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        """
        x: [B, L, C]
        """
        # normalize
        x_norm = self.revin(x, mode="norm")

        # transformer prediction (normalized space)
        y_norm = self.core(x_norm)

        # if predicting all features, denormalize
        if self.output_size == self.input_size:
            y_exp = y_norm.unsqueeze(1)          # [B,1,C]
            y_denorm = self.revin(y_exp, mode="denorm")
            return y_denorm.squeeze(1)           # [B,C]
        else:
            return y_norm
