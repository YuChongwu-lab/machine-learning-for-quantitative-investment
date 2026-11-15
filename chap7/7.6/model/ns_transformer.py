import torch.nn as nn

# =========================
# Series Stationarization module: standardizes each sequence along the time dimension
# =========================
class SeriesStationarization(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        """
        x: [B, L, C]
        Returns:
          x_s: the stationarized sequence [B, L, C]
          mean: each sequence's mean [B, 1, C]
          std:  each sequence's standard deviation [B, 1, C]
        """
        mean = x.mean(dim=1, keepdim=True)                      # [B,1,C]
        std = x.std(dim=1, keepdim=True) + self.eps             # [B,1,C]
        x_s = (x - mean) / std
        return x_s, mean, std


# =========================
# A single encoder layer (with de-stationarization compensation gamma/beta)
# =========================
class NSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, h, gamma, beta, attn_mask=None):
        """
        h: [B, L, d_model]
        gamma, beta: [B, d_model] (compensation scale/bias shared across the whole sequence)
        """
        attn_out, _ = self.mha(h, h, h, attn_mask=attn_mask, need_weights=False)  # [B,L,d]
        # De-stationary compensation: an affine transform applied to the attention output
        attn_out = gamma.unsqueeze(1) * attn_out + beta.unsqueeze(1)              # [B,1,d] broadcast with [B,L,d]
        h = self.norm1(h + self.dropout(attn_out))

        ff = self.ffn(h)
        h = self.norm2(h + self.dropout(ff))
        return h


# =========================
# The Non-stationary Transformer itself
# =========================
class NS_Transformer(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, dropout, output_size):
        super().__init__()
        self.station = SeriesStationarization()

        # Projects the stationarized features to d_model
        self.input_proj = nn.Linear(input_size, d_model)

        # Generates the de-stationarization compensation parameters gamma/beta from the
        # sequence statistics (mean/std); both are projected to d_model first
        self.stats_proj = nn.Linear(input_size, d_model)
        self.gamma_net = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())  # constrained to (-1,1) for stability
        self.beta_net  = nn.Linear(d_model, d_model)

        self.layers = nn.ModuleList([
            NSTEncoderLayer(d_model, num_heads, d_ff=4*d_model, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, output_size)

        # Weight initialization
        nn.init.xavier_uniform_(self.input_proj.weight);   nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.stats_proj.weight);   nn.init.zeros_(self.stats_proj.bias)
        nn.init.xavier_uniform_(self.fc_out.weight);       nn.init.zeros_(self.fc_out.bias)
        for m in self.layers:
            for n in m.ffn:
                if isinstance(n, nn.Linear):
                    nn.init.xavier_uniform_(n.weight); nn.init.zeros_(n.bias)

    def forward(self, x, attn_mask=None):
        """
        x: [B, L, C] -- note: this still takes the features as already standardized
           by the DataLoader (outside this module)
        Output: [B, 1] (matching the training target's dimension)
        """
        # 1) Stationarize the sequence along the time dimension
        x_s, mean, std = self.station(x)                       # x_s:[B,L,C], mean/std:[B,1,C]

        # 2) Project to d_model
        h = self.input_proj(x_s)                               # [B,L,d]

        # 3) Project mean/std to d_model as well, and derive gamma/beta
        mean_d = self.stats_proj(mean.squeeze(1))              # [B,d]
        std_d  = self.stats_proj(std.squeeze(1))               # [B,d]
        gamma = self.gamma_net(std_d)                          # [B,d]
        beta  = self.beta_net(mean_d)                          # [B,d]

        # 4) Stack of encoder layers (all layers share the same gamma/beta; could be
        #    changed to a separate gamma/beta network per layer if needed)
        for layer in self.layers:
            h = layer(h, gamma, beta, attn_mask=attn_mask)     # [B,L,d]

        # 5) Take the last time step and map it to the output
        out = self.fc_out(h[:, -1, :])                         # [B,output_size]

        # Important: this model does not de-normalize its output. The labels are
        # already standardized by TimeSeriesScaler; outside of train/val/test, use
        # scaler.inverse_transform_targets(...) to recover the original scale.
        return out
