import torch
import torch.nn as nn
import numpy as np

# =========================
# Informer model
# =========================
class ProbAttention(nn.Module):
    """ ProbSparse Attention (Informer'core) """
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        # Q,K,V: [B, L, H, D]
        B, L_Q, H, D = Q.shape
        _, L_K, _, _ = K.shape

        # dot-product attention
        scores = torch.einsum("blhd,bshd->bhls", Q, K) / np.sqrt(D)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhls,bshd->blhd", attn, V)  # [B,L,H,D]
        return out.contiguous()


class MultiHeadProbAttention(nn.Module):
    """ multihead ProbAttention """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn = ProbAttention(dropout)

    def forward(self, Q, K, V, mask=None):
        B, L, _ = Q.shape
        Q = self.q_proj(Q).view(B, L, self.n_heads, self.d_k)
        K = self.k_proj(K).view(B, -1, self.n_heads, self.d_k)
        V = self.v_proj(V).view(B, -1, self.n_heads, self.d_k)

        out = self.attn(Q, K, V, mask)  # [B,L,H,D]
        out = out.view(B, L, -1)
        return self.o_proj(out)


class EncoderLayer(nn.Module):
    """ Informer encoder layers """
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
        new_x = self.attn(x, x, x, mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        new_x = self.ff(x)
        x = x + self.dropout(new_x)
        x = self.norm2(x)
        return x


class Informer(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, dropout, output_size):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff=4*d_model, dropout=dropout) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(d_model, output_size)

        # initialize the weights
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        for layer in self.encoder:
            x = layer(x, mask)
        out = self.fc_out(x[:, -1, :])  # keep the last time step
        return out
