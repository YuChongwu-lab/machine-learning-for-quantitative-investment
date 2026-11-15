import torch.nn as nn

"""
An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
"""

#TCN implementation
# Core TCN building block: causal convolution
class Chomp1d(nn.Module):
    """Trim the extra output left by padded convolution so the output
    sequence length matches the input."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


# Residual block
class TemporalBlock(nn.Module):
    """"TCN's basic module: dilated convolution, a residual connection, and ReLU activation"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# TCN model
class TemporalConvNet(nn.Module):
    """The TCN network body, stacked from multiple TemporalBlocks."""
    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i # dilation grows with depth
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        # permute to: [batch_size, features, seq_len]
        x = x.permute(0, 2, 1)
        y = self.network(x)
        # permute back to: [batch_size, seq_len, features]
        return y.permute(0, 2, 1)


# Full TCN prediction model
class TCNPredictor(nn.Module):
    def __init__(self, input_size, output_size, seq_len, num_channels, kernel_size, dropout):
        super(TCNPredictor, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.seq_len = seq_len

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        tcn_out = self.tcn(x)
        # use the last time step's output as the prediction
        final_output = tcn_out[:, -1, :]
        y_pred = self.linear(final_output)
        return y_pred #output shape: (batch_size, output_size)
