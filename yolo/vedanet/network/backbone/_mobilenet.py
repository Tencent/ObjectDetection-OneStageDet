import os
from collections import OrderedDict
import torch
import torch.nn as nn

from .. import layer as vn_layer
from .brick import mobilenet as bmnv

__all__ = ['Mobilenet']

# default Mobilenet 1.0
class Mobilenet(nn.Module):
    """
    """
    def __init__(self, alpha=1):
        """ Network initialisation """
        super().__init__()

        # Network
        layer_list = [
            # Sequence 0 : input = image tensor
            OrderedDict([
                ('1_convbatch',     vn_layer.Conv2dBatchReLU(3, int(alpha*32),  3, 2)),
                ('2_convdw',        bmnv.Conv2dDepthWise(int(alpha*32),  int(alpha*64),  3, 1)),
                ('3_convdw',        bmnv.Conv2dDepthWise(int(alpha*64),  int(alpha*128), 3, 2)),
                ('4_convdw',        bmnv.Conv2dDepthWise(int(alpha*128), int(alpha*128), 3, 1)),
                ]),

            OrderedDict([
                ('5_convdw',        bmnv.Conv2dDepthWise(int(alpha*128), int(alpha*256), 3, 2)),
                ('6_convdw',        bmnv.Conv2dDepthWise(int(alpha*256), int(alpha*256), 3, 1)),
                ]),

            OrderedDict([
                ('7_convdw',        bmnv.Conv2dDepthWise(int(alpha*256), int(alpha*512), 3, 2)),
                ('8_convdw',        bmnv.Conv2dDepthWise(int(alpha*512), int(alpha*512), 3, 1)),
                ('9_convdw',        bmnv.Conv2dDepthWise(int(alpha*512), int(alpha*512), 3, 1)),
                ('10_convdw',       bmnv.Conv2dDepthWise(int(alpha*512), int(alpha*512),  3, 1)),
                ('11_convdw',       bmnv.Conv2dDepthWise(int(alpha*512), int(alpha*512),  3, 1)),
                ('12_convdw',       bmnv.Conv2dDepthWise(int(alpha*512), int(alpha*512),  3, 1)),
                ]),

            OrderedDict([
                ('13_convdw',       bmnv.Conv2dDepthWise(int(alpha*512), int(alpha*1024), 3, 2)),
                ('14_convdw',       bmnv.Conv2dDepthWise(int(alpha*1024), int(alpha*1024), 3, 1)),
                # the following is extra
                ('15_convdw',       bmnv.Conv2dDepthWise(int(alpha*1024), int(alpha*1024), 3, 1)),
                ('16_convdw',       bmnv.Conv2dDepthWise(int(alpha*1024), int(alpha*1024), 3, 1)),
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
