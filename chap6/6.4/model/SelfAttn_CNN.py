import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalSelfAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=None):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads if dim_head is None else dim_head
        inner_dim = self.dim_head * heads

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.to_qkv(x).reshape(B, 3, self.heads, self.dim_head, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # [B, heads, dim_head, H*W]

        q = q.permute(0, 1, 3, 2)  # [B, heads, H*W, dim_head]
        k = k.permute(0, 1, 2, 3)  # [B, heads, dim_head, H*W]
        v = v.permute(0, 1, 3, 2)  # [B, heads, H*W, dim_head]

        attn = torch.matmul(q, k) / (self.dim_head ** 0.5)  # [B, heads, H*W, H*W]
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)  # [B, heads, H*W, dim_head]
        out = out.permute(0, 1, 3, 2).reshape(B, -1, H, W)  # [B, inner_dim, H, W]

        return self.to_out(out)

class SelfAttn_CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.attn1 = GlobalSelfAttention(dim=32, heads=4)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.attn2 = GlobalSelfAttention(dim=64, heads=4)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.attn1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.attn2(x)

        x = self.pool(x).flatten(1)
        return self.fc(x)
