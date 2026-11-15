import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from paths import stock_file


# RevIN: Reversible Instance Normalization
class RevIN(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(RevIN, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        # x has shape [batch_size, seq_len, feature_size]
        batch_size, seq_len, feature_size = x.shape
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)

        # Normalize
        normalized_x = (x - mean) / (std + self.epsilon)
        return normalized_x, mean, std

    def inverse(self, normalized_x, mean, std):
        # Invert the normalization
        return normalized_x * (std + self.epsilon) + mean


# Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, output_size):
        super(TransformerModel, self).__init__()

        self.hidden_size = hidden_size

        # The Transformer's encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=hidden_size
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

        # Output projection
        self.fc_out = nn.Linear(input_size, output_size)

        # RevIN normalization layer
        self.rev_in = RevIN()

    def forward(self, x):
        # Normalize
        normalized_x, mean, std = self.rev_in(x)

        # Permute dims: [batch_size, seq_len, input_size] -> [seq_len, batch_size, input_size]
        normalized_x = normalized_x.permute(1, 0, 2)

        # Transformer encoder
        transformer_out = self.transformer_encoder(normalized_x)

        # Take the last time step's output
        output = transformer_out[-1, :, :]

        # Project to the prediction via the fully-connected layer
        predictions = self.fc_out(output)

        # Invert the normalization
        predictions = self.rev_in.inverse(predictions, mean, std)

        return predictions


# Train/validate the model
def train_validate_model(model, train_data, val_data, epochs, batch_size, lr):
    criterion = nn.MSELoss()  # swap in a different loss if needed
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for i in range(0, len(train_data), batch_size):
            inputs, targets = train_data[i:i + batch_size]

            inputs = torch.tensor(inputs, dtype=torch.float32)
            targets = torch.tensor(targets, dtype=torch.float32)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(inputs)

            # Compute the loss
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_losses.append(train_loss / len(train_data))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                inputs, targets = val_data[i:i + batch_size]

                inputs = torch.tensor(inputs, dtype=torch.float32)
                targets = torch.tensor(targets, dtype=torch.float32)

                predictions = model(inputs)
                loss = criterion(predictions, targets)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_data))

        print(
            f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_data)}, Val Loss: {val_loss / len(val_data)}')

    return train_losses, val_losses


# Generate synthetic time-series data
def generate_synthetic_data(seq_length=1000, num_features=3):
    time = np.linspace(0, 100, seq_length)
    data = np.zeros((seq_length, num_features))

    for i in range(num_features):
        data[:, i] = np.sin(time + i) + np.random.normal(0, 0.1, seq_length)

    return data


# Generate the data
data = generate_synthetic_data(seq_length=1000, num_features=3)
data = pd.read_csv(stock_file('688256.SH.csv'), index_col='trade_date')


# Parameters
input_size = 22  # number of features
hidden_size = 64
num_layers = 2
num_heads = 4
output_size = 12  # 12-step forecast
batch_size = 32
epochs = 20
lr = 1e-3

# Split into train/validation
train_data = data[:800]
val_data = data[800:]

# Reshape into the expected format ([batch_size, seq_len, feature_size])
train_data = [train_data[i:i + 10] for i in range(len(train_data) - 10)]
val_data = [val_data[i:i + 10] for i in range(len(val_data) - 10)]

# Initialize the model
model = TransformerModel(input_size, hidden_size, num_layers, num_heads, output_size)

# Train and validate
train_losses, val_losses = train_validate_model(model, train_data, val_data, epochs, batch_size, lr)
