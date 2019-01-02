"""
ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import layer as vn_layer


class Block(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super().__init__()
        self.stride = stride

        mid_planes = int(out_planes / 4)
        g = 1 if in_planes==24 else groups

        layer_list = [
            nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False),
            nn.BatchNorm2d(mid_planes),
            nn.ReLU(inplace=True),
            vn_layer.Shuffle(groups=g),
            nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False),
            nn.BatchNorm2d(mid_planes),
            nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            ]
        self.layers = nn.Sequential(*layer_list)
        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.layers(x)
        res = self.shortcut(x)
        if self.stride == 2:
            out = torch.cat([y, res], 1)
        else:
            out = y + res
        out = self.activation(out)
        return out


class Stage(nn.Module):
    def __init__(self, in_planes, out_planes, groups, num_blocks):
        super().__init__()
        layer_list = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = in_planes if i == 0 else 0
            layer_list.append(Block(in_planes, out_planes - cat_planes, stride=stride, groups=groups))
            in_planes = out_planes
        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)


