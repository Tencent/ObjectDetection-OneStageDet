import os
from collections import OrderedDict
import torch
import torch.nn as nn

from .. import layer as vn_layer

__all__ = ['TinyYolov2']


class TinyYolov2(nn.Module):
    def __init__(self, num_anchors, num_classes):
        """ Network initialisation """
        super().__init__()
        layer_list = [
            OrderedDict([
                ('14_convbatch',    vn_layer.Conv2dBatchLeaky(1024, 1024, 3, 1)),
                ('15_conv',         nn.Conv2d(1024, num_anchors*(5+num_classes), 1, 1, 0)),
                ]),
            ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, middle_feats):
        outputs = []
        # stage 6
        stage6 = middle_feats[0]
        # Route : layers=-1, -4
        out = self.layers[0](stage6)
        features = [out]
        return features
