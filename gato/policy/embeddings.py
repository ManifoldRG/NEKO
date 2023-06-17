import torch
import torch.nn as nn



class ResidualBlock_V2(nn.Module):

    def __init__(self, mid_channels: int = 128, num_groups: int = 32):

        super().__init__()
        in_channels = 3

        # Specific architecture not provided, potentially different
        self.gn1 = nn.GroupNorm(num_groups, in_channels)
        self.act1 = nn.GeLU()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1) # Could do 1x1, 0 padding

        self.gn2 = nn.GroupNorm(num_groups, mid_channels)
        self.act2 = nn.GeLU()
        self.conv2 = nn.Conv2d(mid_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor):
        # input: B x 3 x 16 x 16
        h = self.conv1(self.act1(self.g1(x)))
        h = self.conv2(self.act2(self.g2(h)))
        return x + h
