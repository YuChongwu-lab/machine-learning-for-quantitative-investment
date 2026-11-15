import torch.nn as nn


class SimpleSAINT(nn.Module):
    def __init__(self, num_features, d_model=128, n_heads=8, n_layers=4, dropout=0.2, num_classes=2):
        super().__init__()
        self.embedding = nn.Linear(num_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [B, F]
        x = self.embedding(x).unsqueeze(1)  # [B, 1, d_model]
        x = self.transformer(x)  # [B, 1, d_model]
        x = self.norm(x[:, 0, :])
        x = self.dropout(x)
        return self.fc_out(x)
