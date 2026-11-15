import torch.nn as nn


class CrossScaleNorm(nn.Module):
    """ Normalizes between scales, to mitigate distribution shift """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, ref_stats):
        """
        x: [B, L, D]
        ref_stats: (mean, std), each [B,1,D], the reference scale's statistics
        """
        mean_ref, std_ref = ref_stats
        return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + self.eps) * std_ref + mean_ref


class Scaleformer(nn.Module):
    """
    Scaleformer: Iterative Multi-scale Refining Transformer
    Compatible interface: init(input_size, d_model, num_heads, num_layers, dropout, output_size)
    """
    def __init__(self, input_size, d_model, num_heads, num_layers, dropout, output_size, scales=(4, 2, 1)):
        super().__init__()
        self.scales = scales
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=4*d_model, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.input_proj = nn.Linear(input_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.cross_norm = CrossScaleNorm()
        self.fc_out = nn.Linear(d_model, output_size)

        nn.init.xavier_uniform_(self.input_proj.weight); nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.fc_out.weight);     nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        """
        x: [B, L, C], same as before
        """
        B, L, C = x.shape
        prev_out = None
        stats = None  # stores the reference scale's mean/std

        for scale in self.scales:
            if scale > 1:
                # Downsample
                xs = x[:, ::scale, :]
            else:
                xs = x

            hs = self.input_proj(xs)  # project
            hs = self.transformer(hs)

            # cross-scale normalization, using the previous scale as reference
            if stats is not None:
                hs = self.cross_norm(hs, stats)
            # Update stats: the current scale's mean/std
            mean = hs.mean(dim=1, keepdim=True)
            std = hs.std(dim=1, keepdim=True)
            stats = (mean, std)

            # Upsample back to the original length, for refinement at the next scale
            if scale > 1:
                hs = hs.repeat_interleave(scale, dim=1)
                hs = hs[:, :L, :]  # truncate
            prev_out = hs

        h = self.norm(prev_out)
        out = self.fc_out(h[:, -1, :])
        return out
