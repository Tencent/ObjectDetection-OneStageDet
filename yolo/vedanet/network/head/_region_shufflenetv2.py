import os
from collections import OrderedDict
import torch
import torch.nn as nn

from .. import layer as vn_layer

__all__ = ['RegionShufflenetv2']


class RegionShufflenetv2(nn.Module):
    def __init__(self, num_anchors, num_classes, input_channels=48):
        """ Network initialisation """
        super().__init__()
        layer_list = [
            # Sequence 2 : input = sequence0
            OrderedDict([
                ('1_convbatch',    vn_layer.Conv2dBatchReLU(232, 32, 1, 1)),
                ('2_reorg',        vn_layer.Reorg(2)),
                ]),

            # Sequence 3 : input = sequence2 + sequence1
            OrderedDict([
                ('3_convbatch',    vn_layer.Conv2dBatchReLU((4*32)+464, 464, 3, 1)),
                ('4_conv',         nn.Conv2d(464, num_anchors*(5+num_classes), 1, 1, 0)),
                ]),
            ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, middle_feats):
        outputs = []
        # stage 5
        # Route : layers=-9
        stage6_reorg = self.layers[0](middle_feats[1])
        # stage 6
        stage6 = middle_feats[0]
        # Route : layers=-1, -4
        out = self.layers[1](torch.cat((stage6_reorg, stage6), 1))
        features = [out]
        return features
