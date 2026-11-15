import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x, mode: str):
        """
        x: [B, L, C]
        mode: 'norm' or 'denorm'
        """
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True)      # [B,1,C]
            self.stdev = x.std(dim=1, keepdim=True) + self.eps
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = self.gamma * x + self.beta
            return x
        elif mode == 'denorm':
            x_hat = x
            if self.affine:
                x_hat = (x_hat - self.beta) / (self.gamma + self.eps)
            # self.mean / self.stdev captured during the last 'norm' call
            x_hat = x_hat * self.stdev + self.mean
            return x_hat
        else:
            raise ValueError("mode must be 'norm' or 'denorm'")

class iTransformerCore(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, dropout, output_size):
        super().__init__()
        self.time_proj = nn.LazyLinear(d_model)
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

    def forward(self, x):   # x: [B, L, C]
        x = x.permute(0, 2, 1)        # [B, C, L]
        tokens = self.time_proj(x)    # [B, C, d_model]
        h = self.encoder(tokens)      # [B, C, d_model]
        h = self.norm(h).mean(dim=1)  # [B, d_model]
        out = self.fc_out(h)          # [B, output_size]
        return out

# --- New iTransformerRevIN: perform denormalization only when output_size == num_features---
class iTransformerRevIN(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, dropout, output_size, affine=True):
        super().__init__()
        self.revin = RevIN(num_features=input_size, affine=affine)
        self.core = iTransformerCore(input_size, d_model, num_heads, num_layers, dropout, output_size)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.revin(x, mode='norm')
        y_norm = self.core(x_norm)   # [B, output_size]

        if self.output_size == self.input_size:
            # expand to [B, 1, output_size] so RevIN.denorm returns [B, 1, C] -> squeeze -> [B,C]
            y_norm_exp = y_norm.unsqueeze(1)            # [B,1,C]
            y_denorm = self.revin(y_norm_exp, mode='denorm')  # [B,1,C]
            return y_denorm.squeeze(1)                  # [B, C]
        else:
            return y_norm
