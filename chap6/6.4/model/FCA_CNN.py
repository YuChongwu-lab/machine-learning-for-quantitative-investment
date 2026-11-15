import torch
import torch.nn as nn
import torch.nn.functional as F

class FcaModule(nn.Module):
    def __init__(self, channel, groups=4):
        super().__init__()
        self.groups = min(groups, channel)
        self.channel = channel
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // 2, bias=False),
            nn.ReLU(),
            nn.Linear(channel // 2, channel, bias=False),
            nn.Sigmoid()
        )

        # Prepare the DCT basis
        c_per_group = channel // self.groups
        u = [0] * self.groups  # customize to select different frequency components
        v = list(range(self.groups))
        self.register_buffer('dct_weights', self._build_dct_weights(channel, u, v))

    def _build_dct_weights(self, C, u, v):
        # supports only global pooling over the spatial map, dimension C
        W = torch.zeros(C)
        for i in range(self.groups):
            idx = i * (C // self.groups)
            W[idx:idx + (C // self.groups)] = 1.0  # simplified selection: uses only a subset of channels
        return W

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # step 1: channel-wise average pooling (acts like low-frequency DCT comp.)
        y = F.adaptive_avg_pool2d(x, 1).view(B, C)
        # step 2: frequency enhancement: multiply each channel by its DCT weight
        y = y * self.dct_weights.unsqueeze(0)
        # step 3: fc layer produces the attention vector
        att = self.fc(y)  # [B, C]
        att = att.view(B, C, 1, 1)
        return x * att

class FCA_CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.attn1 = FcaModule(32, groups=4)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.attn2 = FcaModule(64, groups=4)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.attn1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.attn2(x)

        x = self.pool(x).flatten(1)
        return self.fc(x)
