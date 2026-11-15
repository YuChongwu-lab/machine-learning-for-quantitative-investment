import torch
import torch.nn as nn


class EleAttG(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [B, T, F]
        attn_weights = self.attn(x)  # [B, T, F]
        return x * attn_weights
class EleAttG_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(EleAttG_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.eleattg = EleAttG(input_size)  # add EleAttG

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # [B, T, F]
        x = self.eleattg(x)  # apply element-wise attention weighting

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        y_pred = self.fc(out[:, -1, :])
        return y_pred
