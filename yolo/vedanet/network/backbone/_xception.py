# modified by mileistone

import os
from collections import OrderedDict
import torch
import torch.nn as nn

from .. import layer as vn_layer
from .brick import xception as bx

__all__ = ['Xception']


class Xception(nn.Module):
    """
    """
    def __init__(self):
        """ Network initialisation """
        super().__init__()

        layers_list = [
            # Sequence 0 : input = image tensor
            OrderedDict([
                ('stage3/convbatchrelu1', vn_layer.Conv2dBatchReLU(3, 32, 3, 2)),
                ('stage3/convbatchrelu2', vn_layer.Conv2dBatchReLU(32, 64, 3, 1)),
                ('stage3/miniblock3', bx.MiniBlock(64, 128, 2, 2, False)),
                ]),

            OrderedDict([
                ('stage4/miniblock1', bx.MiniBlock(128, 256, 2, 2)),
                ]),

            OrderedDict([
                ('stage5/miniblock1', bx.MiniBlock(256, 728, 2, 2)),
                ('stage5/stage2', bx.Block(728, 728, 3, 8)),
                ]),

            OrderedDict([
                ('stage6/miniblock1', bx.MiniBlock(728, 1024, 2, 2)),
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
