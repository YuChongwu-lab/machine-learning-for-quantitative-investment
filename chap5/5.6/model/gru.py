import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Fully-connected layer mapping the GRU output to a prediction
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize the hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # GRU forward pass
        out, _ = self.gru(x, h0)

        # Take only the last time step's output,
        # then pass it through the fully-connected layer
        y_pred = self.fc(out[:, -1, :])
        return y_pred
