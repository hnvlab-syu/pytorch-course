import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class EDSR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(32)]
        )
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.upscale = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(in_channels=256, out_channels=3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.res_blocks(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.upscale(x)
        return x
