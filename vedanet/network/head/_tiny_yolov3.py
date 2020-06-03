import os
from collections import OrderedDict
import torch
import torch.nn as nn

from .. import layer as vn_layer

__all__ = ['TinyYolov3']


class TinyYolov3(nn.Module):
    custom_layers = ()
    def __init__(self, num_classes, num_anchors_list):
        """ Network initialisation """
        super().__init__()
        layer_list = [
            # stage 6
            OrderedDict([
                ('14_convbatch',    vn_layer.Conv2dBatchLeaky(256, 512, 3, 1)),
                ('15_conv',         nn.Conv2d(512, num_anchors_list[0]*(5+num_classes), 1, 1, 0)),
            ]),
            # stage 5
            # stage5 / upsample
            OrderedDict([
                ('18_convbatch',    vn_layer.Conv2dBatchLeaky(256, 128, 1, 1)),
                ('19_upsample',     nn.Upsample(scale_factor=2)),
            ]),
            # stage5 / head
            OrderedDict([
                ('21_convbatch',    vn_layer.Conv2dBatchLeaky(256 + 128, 256, 3, 1)),
                ('22_conv',         nn.Conv2d(256, num_anchors_list[0]*(5+num_classes), 1, 1, 0)),
            ]),
            ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, middle_feats):
        outputs = []
        #print(middle_feats[0].shape, middle_feats[1].shape)
        stage6 = self.layers[0](middle_feats[0])
        stage5_upsample = self.layers[1](middle_feats[0])
        stage5 = self.layers[2](torch.cat((middle_feats[1], stage5_upsample), 1))
        features = [stage6, stage5]
        return features
