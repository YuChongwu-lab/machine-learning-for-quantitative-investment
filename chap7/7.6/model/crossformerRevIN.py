import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# RevIN implementation
# -------------------------
class RevIN(nn.Module):
    """
    RevIN (Kim et al., 2022) - reversible instance normalization
    Works on x: [B, L, C]
    Call with mode='norm' before model, and mode='denorm' after model (if desired).
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            # shape [1,1,C] so it broadcasts over batch & time
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta  = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
        # placeholders to store statistics computed during norm()
        self._stored_mean = None
        self._stored_std  = None

    def forward(self, x, mode: str):
        """
        x: [B, L, C]
        mode: 'norm' or 'denorm'
        """
        if mode == 'norm':
            mean = x.mean(dim=1, keepdim=True)         # [B,1,C]
            std = x.std(dim=1, keepdim=True) + self.eps # [B,1,C]
            self._stored_mean = mean
            self._stored_std  = std
            x_norm = (x - mean) / std
            if self.affine:
                x_norm = x_norm * self.gamma + self.beta
            return x_norm

        elif mode == 'denorm':
            if self._stored_mean is None or self._stored_std is None:
                raise RuntimeError("RevIN.denorm called before RevIN.norm was run in same forward pass.")
            x_hat = x
            if self.affine:
                x_hat = (x_hat - self.beta) / (self.gamma + self.eps)
            # broadcast stored mean/std to x_hat shape if needed
            x_hat = x_hat * self._stored_std + self._stored_mean
            return x_hat

        else:
            raise ValueError("mode must be 'norm' or 'denorm'")

# -------------------------
# CrossFormer core components
# -------------------------
class PatchEmbedding(nn.Module):
    """Split sequence into non-overlapping patches of length patch_len and project."""
    def __init__(self, input_size, patch_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.input_size = input_size
        self.proj = nn.Linear(input_size * patch_len, d_model)

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        # number of full patches
        num_patches = L // self.patch_len
        if num_patches == 0:
            # fallback: if seq shorter than patch_len, pad/truncate to patch_len
            x_pad = F.pad(x, (0, 0, 0, self.patch_len - L))  # pad time dim to patch_len
            x_p = x_pad.view(B, 1, self.patch_len * C)
            return self.proj(x_p)  # [B,1,d_model]
        x = x[:, :num_patches * self.patch_len, :]  # truncate residual tail
        x = x.view(B, num_patches, self.patch_len * C)  # [B, N, patch_len*C]
        return self.proj(x)  # [B, N, d_model]


class CrossScaleAttention(nn.Module):
    """
    Cross-scale attention: attend across concatenated tokens of two scales.
    Simplified: perform multihead attention on concatenation [x_large || x_small], then return updated x_large part.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_large, x_small):
        """
        x_large: [B, N1, D]
        x_small: [B, N2, D]
        returns updated x_large [B, N1, D]
        """
        B = x_large.shape[0]
        q = x_large                           # query tokens (we only update these)
        kv = torch.cat([x_large, x_small], dim=1)  # keys/values from both scales
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)  # [B, N1, D]
        out = self.norm(x_large + self.dropout(self.proj(attn_out)))
        return out


class CrossFormerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.cross_attn = CrossScaleAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x_large, x_small):
        x_large = self.cross_attn(x_large, x_small)
        x_large = self.norm2(x_large + self.ffn(x_large))
        return x_large


# -------------------------
# CrossFormer core model
# -------------------------
class CrossFormerCore(nn.Module):
    """
    Core CrossFormer:
      - create two patch embeddings (e.g., patch_len and patch_len*2)
      - stack CrossFormerLayer
      - project final token to output_size
    """
    def __init__(self, input_size, d_model, num_heads, num_layers, dropout, output_size, patch_len=4):
        super().__init__()
        self.patch_len = patch_len
        self.patch_large = PatchEmbedding(input_size, patch_len, d_model)
        self.patch_small = PatchEmbedding(input_size, patch_len * 2, d_model)
        self.layers = nn.ModuleList([
            CrossFormerLayer(d_model, num_heads, d_ff=4 * d_model, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, output_size)

        # weight init
        nn.init.xavier_uniform_(self.fc_out.weight); nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        """
        x: [B, L, C] (assumed reasonably long; patching handles short sequences too)
        returns: [B, output_size]
        """
        # get patch tokens
        x_large = self.patch_large(x)  # [B, N1, D]
        x_small = self.patch_small(x)  # [B, N2, D]

        for layer in self.layers:
            x_large = layer(x_large, x_small)

        x_large = self.norm(x_large)
        # select last token representation (or pooled) -> project
        # if there are no patches, patch embedding returns 1 token; we still use last token.
        rep = x_large[:, -1, :]   # [B, D]
        out = self.fc_out(rep)    # [B, output_size]
        return out

# -------------------------
# CrossFormer + RevIN wrapper
# -------------------------
class CrossformerRevIN(nn.Module):
    """
    Wrapper that applies RevIN before and (optionally) after CrossFormerCore.
    Signature matches older models so you can replace with:
        model = CrossformerRevIN(input_size, d_model, num_heads, num_layers, dropout, output_size).to(device)
    Behavior:
      - If output_size == input_size: model returns denormalized [B, C]
      - Else: model returns predictions in standardized space [B, output_size];
              external scaler.inverse_transform_targets(...) should be used to map back to original scale.
    """
    def __init__(self, input_size, d_model, num_heads, num_layers, dropout, output_size, patch_len=4, affine=True):
        super().__init__()
        self.revin = RevIN(num_features=input_size, affine=affine)
        self.core = CrossFormerCore(input_size, d_model, num_heads, num_layers, dropout, output_size, patch_len=patch_len)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        """
        x: [B, L, C]
        returns:
          - if output_size == input_size -> denormed [B, C]
          - else -> standardized-space preds [B, output_size]
        """
        # 1) RevIN normalization (stores mean/std in module)
        x_norm = self.revin(x, mode='norm')   # [B,L,C]

        # 2) core model (works in normalized space)
        y_norm = self.core(x_norm)            # [B, output_size]

        # 3) denorm only when output_size equals input feature count
        if self.output_size == self.input_size:
            # expand to [B, 1, C] to be compatible with RevIN.denorm
            y_exp = y_norm.unsqueeze(1)      # [B,1,C]
            y_denorm = self.revin(y_exp, mode='denorm')  # [B,1,C]
            return y_denorm.squeeze(1)       # [B,C]
        else:
            # keep in normalized space; caller should inverse via scaler
            return y_norm
