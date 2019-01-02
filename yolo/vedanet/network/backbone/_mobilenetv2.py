#
#   Darknet Darknet19 model
#   Copyright EAVISE
#


# modified by mileistone

import os
from collections import OrderedDict
import torch
import torch.nn as nn

from .. import layer as vn_layer
from .brick import mobilenetv2 as bmnv2

__all__ = ['Mobilenetv2']

# default mobilenetv2 1.0
class Mobilenetv2(nn.Module):
    """
    mobilenetv2
    """
    def __init__(self, width_mult=1):
        """ Network initialisation """
        super().__init__()

        # setting of inverted residual blocks
        cfg = [
            # t, c, n, s
            # stage 3
            [
                [1, 16, 1, 1],
                [6, 24, 2, 2],
            ],
            # stage 4
            [
                [6, 32, 3, 2],
            ],
            # stage 5
            [
                [6, 64, 4, 2],
                [6, 96, 3, 1],
            ],
            # stage 6
            [
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ],
        ]

        self.layers = bmnv2.buildMobilenetv2(cfg, width_mult)

    def forward(self, x):
        stem = self.layers[0](x)
        stage4 = self.layers[1](stem)
        #print(stage4.shape)
        #print(self.layers[0], self.layers[1], self.layers[2])
        stage5 = self.layers[2](stage4)
        stage6 = self.layers[3](stage5)
        features = [stage6, stage5, stage4]
        #print(stage4.shape, stage5.shape, stage6.shape)
        return features
