"""
ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import layer as vn_layer


class Block(nn.Module):
    '''
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        in_list = [int(out_channels * ii) for ii in (0.5, 0.25, 0.5, 0.5)]
        in_list.insert(0, in_channels)
        out_list = in_list[1:] + [out_channels]
        kernel_list = [(1, 1), (1, 1), (3, 1), (1, 3), (1, 1)]
        assert(len(in_list) == len(out_list) == len(kernel_list))
        layer_list = []
        for ii in range(len(in_list)):
            in_ch = in_list[ii]
            out_ch = out_list[ii]
            kernel_size = kernel_list[ii]
            layer = vn_layer.Conv2dBatchReLU(in_ch, out_ch, kernel_size, 1)
            layer_list.append(layer)
        self.layer = nn.Sequential(*layer_list)

    def forward(self, x):
        out = self.layer(x)
        if self.in_channels == self.out_channels:
            out = out + x
        return out


class Stage(nn.Module):
    '''
    '''
    def __init__(self, in_channels, out_channels, repeat_times):
        super().__init__()
        layer_list = []

        if repeat_times >= 1:
            layer = Block(in_channels, out_channels)
            layer_list.append(layer)
        for ii in range(repeat_times - 1):
            layer = Block(out_channels, out_channels)
            layer_list.append(layer)
        self.layer = nn.Sequential(*layer_list)

    def forward(self, x):
        out = self.layer(x)
        return out


