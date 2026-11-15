import torch
import torch.nn as nn
import torch.fft


# =========================
# Series decomposition: trend + seasonality
# =========================
class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        # Use simple mean filtering to extract the trend
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        """
        x: [B, L, C]
        """
        # Pool1d needs [B,C,L]
        trend = self.moving_avg(x.permute(0, 2, 1))
        trend = trend.permute(0, 2, 1)
        season = x - trend
        return season, trend


# =========================
# Auto-Correlation Attention
# =========================
class AutoCorrelationAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.query = nn.Linear(d_model, d_model)
        self.key   = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out   = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B,L,d]
        """
        B, L, D = x.shape
        H = self.n_heads
        q = self.query(x).view(B, L, H, self.d_k).transpose(1, 2)  # [B,H,L,d_k]
        k = self.key(x).view(B, L, H, self.d_k).transpose(1, 2)
        v = self.value(x).view(B, L, H, self.d_k).transpose(1, 2)

        # FFT-based correlation (simplified)
        q_fft = torch.fft.rfft(q, dim=2)
        k_fft = torch.fft.rfft(k, dim=2)
        corr = q_fft * torch.conj(k_fft)
        attn = torch.fft.irfft(corr, n=L, dim=2)  # [B,H,L,d_k]

        # softmax normalization
        attn = torch.softmax(attn / (self.d_k ** 0.5), dim=2)
        out = torch.einsum("bhld,bhld->bhld", attn, v)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(out)


# =========================
# Encoder Layer
# =========================
class AutoformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, dropout=0.1):
        super().__init__()
        self.attn = AutoCorrelationAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x + self.dropout(self.attn(x)))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        return h


# =========================
# Autoformer backbone (main body)
# =========================
class Autoformer(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, dropout, output_size, kernel_size=25):
        super().__init__()
        self.decomp = SeriesDecomposition(kernel_size)
        self.input_proj = nn.Linear(input_size, d_model)
        self.encoder_layers = nn.ModuleList([
            AutoformerEncoderLayer(d_model, num_heads, d_ff=4*d_model, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x):
        """
        x: [B, L, C]
        """
        season, trend = self.decomp(x)
        h = self.input_proj(season)  # only model the last seasonality section
        for layer in self.encoder_layers:
            h = layer(h)
        h = self.norm(h)
        out = self.fc_out(h[:, -1, :])  # take the last time step
        return out
