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

class EleAttG_GRU_Back(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.eleattg = EleAttG(hidden_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.eleattg(out) + out  # add the residual
        y_pred = self.fc(out[:, -1, :])
        return y_pred
