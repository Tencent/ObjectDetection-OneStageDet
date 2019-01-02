import os
from collections import OrderedDict
import torch
import torch.nn as nn

from .. import layer as vn_layer
from .brick import light_xception as blx

__all__ = ['LightXception']


class LightXception(nn.Module):
    def __init__(self):
        """ Network initialisation """
        super().__init__()

        # Network
        layers_list= [
            OrderedDict([
                ('stage3/conv1',     vn_layer.Conv2dBatchReLU(3, 24, 3, 2)),
                ('stage3/downsample2', nn.MaxPool2d(3, 2, 1)),
                ]),


            OrderedDict([
                ('stage4/miniblock1', blx.MiniBlock(24, 144, 2, 2)),
                ('stage4/stage2', blx.Block(144, 144, 3, 3)),
                ]),

            OrderedDict([
                ('stage5/miniblock1', blx.MiniBlock(144, 288, 2, 2)),
                ('stage5/stage2', blx.Block(288, 288, 3, 7)),
                ]),

            OrderedDict([
                ('stage6/miniblock1', blx.MiniBlock(288, 576, 2, 2)),
                ('stage6/stage2', blx.Block(576, 576, 3, 3)),
                # the following is extra
                ]),
        ]

        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layers_list])

    def forward(self, x):
        stem = self.layers[0](x)
        stage4 = self.layers[1](stem)
        stage5 = self.layers[2](stage4)
        stage6 = self.layers[3](stage5)
        features = [stage6, stage5, stage4]
        return features
