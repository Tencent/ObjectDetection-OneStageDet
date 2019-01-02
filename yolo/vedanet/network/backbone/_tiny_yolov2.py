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

__all__ = ['TinyYolov2']


class TinyYolov2(nn.Module):
    """
    """
    def __init__(self):
        """ Network initialisation """
        super().__init__()

        # Network
        layer_list = [
            OrderedDict([
                ('1_convbatch',     vn_layer.Conv2dBatchLeaky(3, 16, 3, 1)),
                ('2_max',           nn.MaxPool2d(2, 2)),
                ('3_convbatch',     vn_layer.Conv2dBatchLeaky(16, 32, 3, 1)),
                ('4_max',           nn.MaxPool2d(2, 2)),
                ('5_convbatch',     vn_layer.Conv2dBatchLeaky(32, 64, 3, 1)),
                ('6_max',           nn.MaxPool2d(2, 2)),
                ('7_convbatch',     vn_layer.Conv2dBatchLeaky(64, 128, 3, 1)),
                ]),
            OrderedDict([
                ('8_max',           nn.MaxPool2d(2, 2)),
                ('9_convbatch',     vn_layer.Conv2dBatchLeaky(128, 256, 3, 1)),
                ]),
            OrderedDict([
                ('10_max',          nn.MaxPool2d(2, 2)),
                ('11_convbatch',    vn_layer.Conv2dBatchLeaky(256, 512, 3, 1)),
                ]),
            OrderedDict([
                ('12_max',          vn_layer.PaddedMaxPool2d(2, 1, (0, 1, 0, 1))),
                ('13_convbatch',    vn_layer.Conv2dBatchLeaky(512, 1024, 3, 1)),
                ]),
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, x):
        stem = self.layers[0](x)
        stage4 = self.layers[1](stem)
        stage5 = self.layers[2](stage4)
        stage6 = self.layers[3](stage5)
        #print(stage5.shape, stage6.shape)
        features = [stage6, stage5, stage4]
        return features
