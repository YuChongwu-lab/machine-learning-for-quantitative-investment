import torch
import torch.nn as nn


class iTransformer(nn.Module):
    """
    iTransformer (Liu et al., 2024)
    """
    def __init__(self, input_size, d_model, num_heads, num_layers, dropout, output_size):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model

        self.time_proj = nn.LazyLinear(d_model)

        # Apply Transformer encoding along the variable dimension (tokens).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,   
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, output_size)

        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C]
        """
        # [B, L, C] -> [B, C, L]
        x = x.permute(0, 2, 1)           # [B, C, L]
        tokens = self.time_proj(x)       # [B, C, d_model]

        #Apply the Transformer over the variable dimension (C tokens)
        h = self.encoder(tokens)         # [B, C, d_model]

        # Aggregate over variables: mean pooling 
        h = self.norm(h).mean(dim=1)     # [B, d_model]
        out = self.fc_out(h)             # [B, output_size]
        return out
