import os
from collections import OrderedDict
import torch
import torch.nn as nn


from .brick import yolov3 as byolov3

__all__ = ['Yolov3']


class Yolov3(nn.Module):
    custom_layers = (byolov3.Head, byolov3.Head.custom_layers)
    def __init__(self, num_classes, in_channels_list, num_anchors_list):
        """ Network initialisation """
        super().__init__()
        layer_list = [
            # stage 6, largest
            OrderedDict([
                ('1_head', byolov3.Head(in_channels_list[0], num_anchors_list[0], num_classes)),
                ]),

            OrderedDict([
                ('2_head', byolov3.Head(in_channels_list[1], num_anchors_list[1], num_classes)),
                ]),

            # smallest
            OrderedDict([
                ('2_head', byolov3.Head(in_channels_list[2], num_anchors_list[2], num_classes)),
                ]),
            ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, middle_feats):
        outputs = []
        stage6 = self.layers[0](middle_feats[0])
        stage5 = self.layers[1](middle_feats[1])
        stage4 = self.layers[2](middle_feats[2])
        features = [stage6, stage5, stage4]
        return features
