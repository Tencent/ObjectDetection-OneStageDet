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

__all__ = ['Darknet19']


class Darknet19(nn.Module):
    """ `Darknet19`_ implementation with pytorch.

    Todo:
        - Loss function: L2 (Crossentropyloss in pytorch)

    Args:
        weights_file (str, optional): Path to the saved weights; Default **None**
        input_channels (Number, optional): Number of input channels; Default **3**

    Attributes:
        self.loss (fn): loss function. Usually this is :class:`~lightnet.network.RegionLoss`
        self.postprocess (fn): Postprocessing function. By default this is :class:`~lightnet.data.GetBoundingBoxes`

    .. _Darknet19: https://github.com/pjreddie/darknet/blob/master/cfg/darknet19.cfg
    """
    def __init__(self):
        """ Network initialisation """
        super().__init__()

        # Network
        layer_list = [
            OrderedDict([
                ('1_convbatch',     vn_layer.Conv2dBatchLeaky(3, 32, 3, 1)),
                ('2_max',           nn.MaxPool2d(2, 2)),
                ('3_convbatch',     vn_layer.Conv2dBatchLeaky(32, 64, 3, 1)),
                ('4_max',           nn.MaxPool2d(2, 2)),
                ('5_convbatch',     vn_layer.Conv2dBatchLeaky(64, 128, 3, 1)),
                ('6_convbatch',     vn_layer.Conv2dBatchLeaky(128, 64, 1, 1)),
                ('7_convbatch',     vn_layer.Conv2dBatchLeaky(64, 128, 3, 1)),
                ]),

            OrderedDict([
                ('8_max',           nn.MaxPool2d(2, 2)),
                ('9_convbatch',     vn_layer.Conv2dBatchLeaky(128, 256, 3, 1)),
                ('10_convbatch',    vn_layer.Conv2dBatchLeaky(256, 128, 1, 1)),
                ('11_convbatch',    vn_layer.Conv2dBatchLeaky(128, 256, 3, 1)),
                ]),

            OrderedDict([
                ('12_max',          nn.MaxPool2d(2, 2)),
                ('13_convbatch',    vn_layer.Conv2dBatchLeaky(256, 512, 3, 1)),
                ('14_convbatch',    vn_layer.Conv2dBatchLeaky(512, 256, 1, 1)),
                ('15_convbatch',    vn_layer.Conv2dBatchLeaky(256, 512, 3, 1)),
                ('16_convbatch',    vn_layer.Conv2dBatchLeaky(512, 256, 1, 1)),
                ('17_convbatch',    vn_layer.Conv2dBatchLeaky(256, 512, 3, 1)),
                ]),

            OrderedDict([
                ('18_max',          nn.MaxPool2d(2, 2)),
                ('19_convbatch',    vn_layer.Conv2dBatchLeaky(512, 1024, 3, 1)),
                ('20_convbatch',    vn_layer.Conv2dBatchLeaky(1024, 512, 1, 1)),
                ('21_convbatch',    vn_layer.Conv2dBatchLeaky(512, 1024, 3, 1)),
                ('22_convbatch',    vn_layer.Conv2dBatchLeaky(1024, 512, 1, 1)),
                ('23_convbatch',    vn_layer.Conv2dBatchLeaky(512, 1024, 3, 1)),
                # the following is extra
                ('24_convbatch',    vn_layer.Conv2dBatchLeaky(1024, 1024, 3, 1)),
                ('25_convbatch',    vn_layer.Conv2dBatchLeaky(1024, 1024, 3, 1)),
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
