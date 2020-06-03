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

__all__ = ['TinyYolov3']


class TinyYolov3(nn.Module):
    """
    """
    def __init__(self):
        """ Network initialisation """
        super().__init__()

        # Network
        layer_list = [
            # Sequence 0 : input = image tensor
            # output redutcion 16
            # backbone
            OrderedDict([
                ('0_convbatch',     vn_layer.Conv2dBatchLeaky(3, 16, 3, 1)),
                ('1_max',           nn.MaxPool2d(2, 2)),
                ('2_convbatch',     vn_layer.Conv2dBatchLeaky(16, 32, 3, 1)),
                ('3_max',           nn.MaxPool2d(2, 2)),
                ('4_convbatch',     vn_layer.Conv2dBatchLeaky(32, 64, 3, 1)),
            ]),

            OrderedDict([
                ('5_max',           nn.MaxPool2d(2, 2)),
                ('6_convbatch',    vn_layer.Conv2dBatchLeaky(64, 128, 3, 1)),
            ]),

            OrderedDict([
                ('7_max',          nn.MaxPool2d(2, 2)),
                ('8_convbatch',    vn_layer.Conv2dBatchLeaky(128, 256, 3, 1)),
            ]),

            # Sequence 1 : input = sequence0
            # output redutcion 32
            # backbone
            OrderedDict([
                ('9_max',          nn.MaxPool2d(2, 2)),
                ('10_convbatch',    vn_layer.Conv2dBatchLeaky(256, 512, 3, 1)),
                ('11_max',          nn.MaxPool2d(3, 1, 1)),
                ('12_convbatch',    vn_layer.Conv2dBatchLeaky(512, 1024, 3, 1)),
                ('13_convbatch',    vn_layer.Conv2dBatchLeaky(1024, 256, 1, 1)),
            ]),

        ]

        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, x):
        stem = self.layers[0](x)
        stage4 = self.layers[1](stem)
        stage5 = self.layers[2](stage4)
        stage6 = self.layers[3](stage5)
        features = [stage6, stage5, stage4]
        return features
