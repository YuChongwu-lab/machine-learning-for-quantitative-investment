import torch
import torch.nn as nn
import torch.fft


# =========================
# Series decomposition: trend + seasonality
# =========================
class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=pad, count_include_pad=False)

    def forward(self, x: torch.Tensor):
        """
        x: [B, L, C]
        returns: season, trend 
        """
        trend = self.moving_avg(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B,L,C]
        season = x - trend
        return season, trend


# =========================
# Simplified Fourier Attention
# =========================
class FourierAttention(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, top_k: int = 64):
        super().__init__()
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be in [0,1], got {dropout}")
        self.top_k = top_k
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        """
        x: [B, L, D]
        """
        B, L, D = x.shape

        xf = torch.fft.rfft(x, dim=1) 
        # Retain only the low-frequency top_k components (set the high-frequency components to zero)
        k = min(self.top_k, xf.size(1))
        xf_low = torch.zeros_like(xf)
        xf_low[:, :k, :] = xf[:, :k, :]
        # Back to time domain
        out = torch.fft.irfft(xf_low, n=L, dim=1)  # [B,L,D]
        out = self.proj(self.dropout(out))
        return out


# =========================
# Encoder Layer
# =========================
class FEDformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 256, dropout: float = 0.1, top_k: int = 64):
        super().__init__()
        self.freq_attn = FourierAttention(d_model, dropout, top_k)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        h = self.norm1(x + self.dropout(self.freq_attn(x)))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        return h


# =========================
# FEDformer (main body)
# =========================
class FEDformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        num_heads: int,          
        num_layers: int,
        dropout: float,
        output_size: int,
        kernel_size: int = 25,
        top_k: int = 64,
    ):
        super().__init__()
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be in [0,1], got {dropout}")

        self.decomp = SeriesDecomposition(kernel_size)
        self.input_proj = nn.Linear(input_size, d_model)
        self.encoder_layers = nn.ModuleList([
            FEDformerEncoderLayer(d_model, d_ff=4 * d_model, dropout=dropout, top_k=top_k)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, output_size)

        nn.init.xavier_uniform_(self.input_proj.weight); nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.fc_out.weight);     nn.init.zeros_(self.fc_out.bias)
        for layer in self.encoder_layers:
            for m in layer.ffn:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        x: [B, L, C]
        return: [B, output_size]
        """
        season, trend = self.decomp(x)
        h = self.input_proj(season)            
        for layer in self.encoder_layers:
            h = layer(h)
        h = self.norm(h)
        out = self.fc_out(h[:, -1, :])         
        return out
