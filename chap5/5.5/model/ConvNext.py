import torch.nn as nn
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x = x.view(b, -1, c)  # [B, HW, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        return x + residual

class ConvNeXt(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, dim=64, num_blocks=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.stages = nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        return self.head(x)
