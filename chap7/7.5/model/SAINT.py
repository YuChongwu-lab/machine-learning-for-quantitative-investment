import torch
import torch.nn as nn

import torch
import torch.nn as nn

class FeatureTokenizer(nn.Module):
    """
    Maps each numeric feature to a token (close to the numeric-feature
    tokenizer idea in FT-Transformer/SAINT).
    For column i: token_i = x_i * W_i + b_i  -> dimension d_model
    """
    def __init__(self, num_features: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, d_model) * 0.02)  # [F, d_model]
        self.bias   = nn.Parameter(torch.zeros(num_features, d_model))         # [F, d_model]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, F] (all-numeric features)
        return: [B, F, d_model] (one token per column)
        """
        # x[:, :, None] -> [B, F, 1]
        out = x[:, :, None] * self.weight[None, :, :] + self.bias[None, :, :]
        return self.dropout(out)


class SAINT(nn.Module):
    """
    A near-complete, numeric-feature version of SAINT (Somepalli et al., 2021):
    - FeatureTokenizer: turns 93 numeric features into 93 tokens
    - Column Embedding: learnable column positional encoding
    - Alternates Row-Attention (across features within a sample) and
      Column-Attention (across samples within the same column)
      * Row: self-attention over the F dimension of [B, F, d] (batch_first=True)
      * Col: transpose to [F, B, d], self-attention over the B dimension
        (across samples, same column), then transpose back
    - Dropout / LayerNorm
    - Classification head: global mean pooling + MLP
    """
    def __init__(
        self,
        num_features: int = 93,
        num_classes: int = 2,
        d_model: int = 128,
        n_heads: int = 8,
        n_blocks: int = 6,          # number of alternating blocks (Row/Col come in pairs)
        dropout: float = 0.2,
        mlp_hidden_mult: int = 2
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        # 1) Feature tokenizer (numeric features -> tokens)
        self.tokenizer = FeatureTokenizer(num_features, d_model, dropout=dropout)

        # 2) Column embedding (column positional encoding)
        self.col_embedding = nn.Parameter(torch.randn(1, num_features, d_model) * 0.02)

        # 3) Row-Attention encoder (within a sample, attends across columns/features)
        row_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.row_blocks = nn.ModuleList([nn.TransformerEncoder(row_encoder_layer, num_layers=1) for _ in range(n_blocks)])

        # 4) Column-Attention encoder (across samples, attends along the batch dimension)
        #    Also uses TransformerEncoderLayer, but in forward() we transpose to [F, B, d] before feeding it in
        col_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.col_blocks = nn.ModuleList([nn.TransformerEncoder(col_encoder_layer, num_layers=1) for _ in range(n_blocks)])

        # 5) Normalization and classification head
        self.norm = nn.LayerNorm(d_model)
        hidden = d_model * mlp_hidden_mult
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        """
        x: [B, F] (all-numeric input, F=93)
        return: [B, num_classes]
        """
        B, F = x.shape
        assert F == self.num_features, f"Expected {self.num_features} features, got {F}"

        # [B, F, d]
        h = self.tokenizer(x)
        h = h + self.col_embedding  # column embedding

        # Alternate Row / Column blocks
        for row_block, col_block in zip(self.row_blocks, self.col_blocks):
            # Row attention: self-attention over the feature sequence within each sample ([B, F, d])
            h = row_block(h)  # [B, F, d]

            # Column attention: lets information flow across samples within the same column
            # Transpose to [F, B, d] to attend over the batch dimension (still [*, *, d] with batch_first=True after the swap)
            h_col = h.permute(1, 0, 2).contiguous()  # [F, B, d]
            h_col = col_block(h_col)                  # attention over the "sample dimension"
            h = h_col.permute(1, 0, 2).contiguous()   # [B, F, d]

        # Pool + classify
        h = self.norm(h.mean(dim=1))  # global average pooling
        out = self.head(h)            # [B, num_classes]
        return out
