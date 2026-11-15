import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = in_channels // 2
            if inter_channels == 0:
                inter_channels = 1

        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )

        self.W_g = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # x: feature map from lower layer
        # g: gating signal (can be same shape)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
class AG_CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Attention gates: the features after each layer are fed into the AG, using
        # themselves as the gating signal (a deeper layer could be used instead)
        self.ag1 = AttentionGate(32, 32)
        self.ag2 = AttentionGate(64, 64)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = self.ag1(x1, x1)  # itself as the gating signal

        x2 = F.relu(self.bn2(self.conv2(x1)))
        x2 = self.ag2(x2, x2)

        x_out = self.pool(x2).flatten(1)
        return self.fc(x_out)
