import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformableAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3, num_heads=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.offset_conv = nn.Conv2d(in_channels, 2 * num_heads, kernel_size=3, padding=1)
        self.attn_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        B, C, H, W = x.size()
        offset = self.offset_conv(x)  # shape: [B, 2*num_heads, H, W]
        offset = offset.view(B * self.num_heads, 2, H, W)

        # Create base grid
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        grid = torch.stack((grid_x, grid_y), 0).float().to(x.device)  # [2, H, W]
        grid = grid.unsqueeze(0).repeat(B * self.num_heads, 1, 1, 1)  # [B*num_heads, 2, H, W]

        # Apply offset
        sampling_locations = grid + offset  # [B*num_heads, 2, H, W]
        sampling_locations[:, 0] = 2.0 * sampling_locations[:, 0] / max(W - 1, 1) - 1.0
        sampling_locations[:, 1] = 2.0 * sampling_locations[:, 1] / max(H - 1, 1) - 1.0
        sampling_grid = sampling_locations.permute(0, 2, 3, 1)  # [B*num_heads, H, W, 2]

        # Value projection
        value = self.value_conv(x)
        value = value.view(B * self.num_heads, C // self.num_heads, H, W)

        # Grid sample
        sampled = F.grid_sample(value, sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled = sampled.view(B, C, H, W)

        # Attention weights
        attn_weights = torch.sigmoid(self.attn_conv(x))
        return sampled * attn_weights
class DeformAttn_CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.def_attn1 = DeformableAttention(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.def_attn2 = DeformableAttention(64)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.def_attn1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.def_attn2(x)

        x = self.pool(x).flatten(1)
        return self.fc(x)

