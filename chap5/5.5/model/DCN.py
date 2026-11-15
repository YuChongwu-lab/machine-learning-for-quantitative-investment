import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class DeformableCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.offset1 = nn.Conv2d(1, 18, kernel_size=3, padding=1)
        self.conv1 = DeformConv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.offset2 = nn.Conv2d(32, 18, kernel_size=3, padding=1)
        self.conv2 = DeformConv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        offset1 = self.offset1(x)
        x = F.relu(self.bn1(self.conv1(x, offset1)))

        offset2 = self.offset2(x)
        x = F.relu(self.bn2(self.conv2(x, offset2)))

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
