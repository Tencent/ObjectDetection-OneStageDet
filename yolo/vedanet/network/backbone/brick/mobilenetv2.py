import torch.nn as nn
from collections import OrderedDict

from ... import layer as vn_layer


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        if abs(expand_ratio - 1) < .01:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def buildInvertedResBlock(residual_setting, input_channel, width_mult):
    # building inverted residual blocks
    features = []
    for t, c, n, s in residual_setting:
        output_channel = int(c * width_mult)
        for i in range(n):
            if i == 0:
                features.append(InvertedResidual(input_channel, output_channel, s, t))
            else:
                features.append(InvertedResidual(input_channel, output_channel, 1, t))
            input_channel = output_channel
    layers = nn.Sequential(*features)
    return layers, input_channel


def buildMobilenetv2(cfg, width_mult):
    """
    """
    # building first layer
    input_channel = int(32 * width_mult)

    #### stage 3
    s3_layer1 = vn_layer.Conv2dBatchReLU(3, input_channel, 3, 2)
    residual_setting = cfg[0]
    s3_layer2, output_channel_stage3 = buildInvertedResBlock(residual_setting, input_channel, 
            width_mult)


    #### stage 4
    residual_setting = cfg[1]
    s4_layer1, output_channel_stage4 = buildInvertedResBlock(residual_setting, output_channel_stage3, 
            width_mult)

    #### stage 5
    residual_setting = cfg[2]
    s5_layer1, output_channel_stage5 = buildInvertedResBlock(residual_setting, output_channel_stage4, 
            width_mult)

    #### stage 6
    residual_setting = cfg[3]
    s6_layer1, output_channel_stage6 = buildInvertedResBlock(residual_setting, output_channel_stage5, 
            width_mult)
    layer_list = [
        # stage 3
        OrderedDict([
            ('stage3/layer1', s3_layer1),
            ('stage3/layer2', s3_layer2),
            ]),
        # stage 4
        OrderedDict([
            ('stage4/layer1', s4_layer1),
            ]),
        # stage 5
        OrderedDict([
            ('stage5/layer1', s5_layer1),
            ]),
        # stage 6
        OrderedDict([
            ('stage6/layer1', s6_layer1),
            ]),
        ]

    layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    return layers
