import torch
import torch.nn as nn


class TPAModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(input_size, hidden_size)
        self.value_proj = nn.Linear(input_size, hidden_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, h):
        # x: [B, T, F]  => input sequence
        # h: [B, H]     => hidden state from GRU/LSTM
        Q = self.query_proj(h).unsqueeze(1)      # [B, 1, H]
        K = self.key_proj(x)                     # [B, T, H]
        V = self.value_proj(x)                   # [B, T, H]

        scores = torch.bmm(Q, K.transpose(1, 2)) / (K.size(-1) ** 0.5)  # [B, 1, T]
        weights = self.softmax(scores)           # [B, 1, T]
        context = torch.bmm(weights, V)          # [B, 1, H]
        return context.squeeze(1)                # [B, H]


class TPA_GRU_NoRes(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.tpa = TPAModule(input_size=input_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [B, T, F]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, hn = self.gru(x, h0)  # out: [B, T, H], hn: [num_layers, B, H]
        h_last = hn[-1]            # [B, H]
        context = self.tpa(x, h_last)  # [B, H]  (no residual, unlike TPA_GRU)
        y_pred = self.fc(context)      # [B, output_size]
        return y_pred
