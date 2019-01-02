import os
from collections import OrderedDict
import torch
import torch.nn as nn

from .. import layer as vn_layer
from .brick import squeezenext as bsn

__all__ = ['Squeezenext']

# default 1.0-SqNxt-23v5
# there are some difference with orignal 1.0xSqNxt-23v5 in downsample part,
# where we just use a 3x3 conv with stride 2 to downsample.
class Squeezenext(nn.Module):
    """
    """
    def __init__(self, width_mul=1):
        """ Network initialisation """
        super().__init__()
        

        sqz_chns = [3, 64, 
                        32*width_mul, 64*width_mul, 128*width_mul, 256*width_mul]
        sqz_stage_cfg = [None, None, 
                        2, 4, 14, 1]

        # Network
        layer_list = [
            # Sequence 0 : input = image tensor
            OrderedDict([
                # stage 1
                ('stage1/downsample', vn_layer.Conv2dBatchReLU(sqz_chns[0], sqz_chns[1], 5, 2)),
                # stage 2
                # pass

                # stage 3
                ('stage3/downsample', nn.MaxPool2d(3, 2, 1)),
                ('stage3/trans', vn_layer.Conv2dBatchReLU(sqz_chns[1], sqz_chns[2], 1, 1)),
                ('stage3/squeeze', bsn.Stage(sqz_chns[2], sqz_chns[2], sqz_stage_cfg[2])),
            ]),

                # stage 4
            OrderedDict([
                ('stage4/trans', vn_layer.Conv2dBatchReLU(sqz_chns[2], sqz_chns[3], 3, 2)),
                ('stage4/squeeze', bsn.Stage(sqz_chns[3], sqz_chns[3], sqz_stage_cfg[3])),
            ]),

                # stage 5
            OrderedDict([
                ('stage5/trans', vn_layer.Conv2dBatchReLU(sqz_chns[3], sqz_chns[4], 3, 2)),
                ('stage5/squeeze', bsn.Stage(sqz_chns[4], sqz_chns[4], sqz_stage_cfg[4])),
            ]),

            # Sequence 1 : input = sequence0
            OrderedDict([
                # stage 6
                ('stage6/trans', vn_layer.Conv2dBatchReLU(sqz_chns[4], sqz_chns[5], 3, 2)),
                ('stage6/squeeze', bsn.Stage(sqz_chns[5], sqz_chns[5], sqz_stage_cfg[5])),
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
