"""
ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import layer as vn_layer


class Split(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        c = int(x.size(1) * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]


class BasicBlock(nn.Module):
    def __init__(self, in_channels, split_ratio=0.5, groups=2):
        super().__init__()
        in_channels = int(in_channels * split_ratio)

        layer_list = [
            vn_layer.Conv2dBatchReLU(in_channels, in_channels, 1, 1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            vn_layer.Conv2dBatchReLU(in_channels, in_channels, 1, 1),
            ]

        self.split = Split(split_ratio)
        self.layers = nn.Sequential(*layer_list)
        self.shuffle = vn_layer.Shuffle(groups)

    def forward(self, x):
        x1, x2 = self.split(x)
        x2_1 = self.layers(x2)
        x_1 = torch.cat([x1, x2_1], 1)
        out = self.shuffle(x_1)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=2):
        super().__init__()
        mid_channels = out_channels // 2
        # left
        left_list = [
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            vn_layer.Conv2dBatchReLU(in_channels, mid_channels, 1, 1),
            ]
        # right
        right_list = [
            vn_layer.Conv2dBatchReLU(in_channels, mid_channels, 1, 1),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            vn_layer.Conv2dBatchReLU(mid_channels, mid_channels, 1, 1),
            ]

        self.left_layers = nn.Sequential(*left_list)
        self.right_layers = nn.Sequential(*right_list)
        self.shuffle = vn_layer.Shuffle(groups)

    def forward(self, x):
        left_x = self.left_layers(x)
        right_x = self.right_layers(x)
        # concat
        concat = torch.cat([left_x, right_x], 1)
        out = self.shuffle(concat)
        return out


class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, groups, num_blocks):
        super().__init__()
        layer_list = [DownBlock(in_channels, out_channels)]
        for i in range(num_blocks):
            layer_list.append(BasicBlock(out_channels))
            in_channels = out_channels

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)

