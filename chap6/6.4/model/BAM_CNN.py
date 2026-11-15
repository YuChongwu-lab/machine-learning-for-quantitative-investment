import torch.nn as nn
import torch.nn.functional as F

class BAM(nn.Module):
    def __init__(self, channels, reduction=8, dilation=4):
        super(BAM, self).__init__()

        # Channel attention branch
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )

        # Spatial attention branch
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels // reduction, kernel_size=3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, 1, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att_channel = self.channel_att(x)
        att_spatial = self.spatial_att(x)
        att = self.sigmoid(att_channel + att_spatial)  # shape [B, C, H, W] & [B, 1, H, W] broadcast
        return x * att
class BAM_CNN(nn.Module):
    def __init__(self, num_classes):
        super(BAM_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bam1 = BAM(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.bam2 = BAM(64)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bam1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bam2(x)

        x = self.pool(x).flatten(1)
        return self.fc(x)
