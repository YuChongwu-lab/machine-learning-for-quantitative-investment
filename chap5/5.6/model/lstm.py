import torch
import torch.nn as nn

# Define the LSTM network
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size  # hidden layer size
        self.num_layers = num_layers  # number of LSTM layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #x shape:[batch_size, seq_len, input_size]
        # Initialize the hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        output, _ = self.lstm(x,(h0,c0))# output shape: [batch_size, seq_len, hidden_size]

        y_pred = self.fc(output[:,-1,:])  # output shape: [batch_size, output_size]

        return y_pred  # return the last time step's output
