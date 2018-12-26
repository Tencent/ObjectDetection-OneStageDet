import os
from collections import OrderedDict
import torch
import torch.nn as nn

from ... import layer as vn_layer


class StageBlock(nn.Module):
    custom_layers = ()
    def __init__(self, nchannels):
        super().__init__()
        self.features = nn.Sequential(
                    vn_layer.Conv2dBatchLeaky(nchannels, int(nchannels/2), 1, 1),
                    vn_layer.Conv2dBatchLeaky(int(nchannels/2), nchannels, 3, 1)
                )

    def forward(self, data):
        return data + self.features(data)


class Stage(nn.Module):
    custom_layers = (StageBlock, StageBlock.custom_layers)
    def __init__(self, nchannels, nblocks, stride=2):
        super().__init__()
        blocks = []
        blocks.append(vn_layer.Conv2dBatchLeaky(nchannels, 2*nchannels, 3, stride))
        for ii in range(nblocks - 1):
            blocks.append(StageBlock(2*nchannels))
        self.features = nn.Sequential(*blocks)

    def forward(self, data):
        return self.features(data)


class HeadBody(nn.Module):
    custom_layers = ()
    def __init__(self, nchannels, first_head=False):
        super().__init__()
        if first_head:
            half_nchannels = int(nchannels/2)
        else:
            half_nchannels = int(nchannels/3)
        in_nchannels = 2 * half_nchannels
        layers = [
                vn_layer.Conv2dBatchLeaky(nchannels, half_nchannels, 1, 1),
                vn_layer.Conv2dBatchLeaky(half_nchannels, in_nchannels, 3, 1),
                vn_layer.Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1),
                vn_layer.Conv2dBatchLeaky(half_nchannels, in_nchannels, 3, 1),
                vn_layer.Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1)
                ]
        self.feature = nn.Sequential(*layers)

    def forward(self, data):
        x = self.feature(data)
        return x


class Transition(nn.Module):
    custom_layers = ()
    def __init__(self, nchannels):
        super().__init__()
        half_nchannels = int(nchannels/2)
        layers = [
                vn_layer.Conv2dBatchLeaky(nchannels, half_nchannels, 1, 1),
                nn.Upsample(scale_factor=2)
                ]

        self.features = nn.Sequential(*layers)

    def forward(self, data):
        x = self.features(data)
        return x

