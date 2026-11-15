import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, input_size, patch_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(input_size * patch_len, d_model)

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        num_patches = L // self.patch_len
        x = x[:, :num_patches*self.patch_len, :]  
        x = x.view(B, num_patches, self.patch_len * C)
        return self.proj(x)  # [B, num_patches, d_model]


class CrossScaleAttention(nn.Module):
    """ Cross-Scale Self-Attention """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, x_small):
        # x: [B, N1, D], x_small: [B, N2, D]
        q = k = v = torch.cat([x, x_small], dim=1)
        attn_out, _ = self.attn(q, k, v)
        out = self.norm(x + self.dropout(self.proj(attn_out[:, :x.size(1), :])))
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
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, x_small):
        x = self.cross_attn(x, x_small)
        x = self.norm(x + self.ffn(x))
        return x


class Crossformer(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, dropout, output_size,
                 patch_len=4):
        super().__init__()
        self.patch_embed_large = PatchEmbedding(input_size, patch_len, d_model)
        self.patch_embed_small = PatchEmbedding(input_size, patch_len*2, d_model)

        self.layers = nn.ModuleList([
            CrossFormerLayer(d_model, num_heads, d_ff=4*d_model, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, output_size)

        nn.init.xavier_uniform_(self.fc_out.weight); nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        """
        x: [B, L, C]
        """
        x_large = self.patch_embed_large(x)  # [B, N1, D]
        x_small = self.patch_embed_small(x)  # [B, N2, D]

        out = x_large
        for layer in self.layers:
            out = layer(out, x_small)

        out = self.norm(out)
        out = self.fc_out(out[:, -1, :])  
        return out
