import torch
import torch.nn as nn

from ... import layer as vn_layer

class SeparableConv2d(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, relu_in_middle=True):
        super().__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.relu_in_middle = relu_in_middle

        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii/2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size/2)

        # Layer
        if relu_in_middle:
            self.layers = nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, self.kernel_size, self.stride, self.padding, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.out_channels),
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, self.kernel_size, self.stride, self.padding, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),

                nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.out_channels),
            )


    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, relu_in_middle={relu_in_middle})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x


class MiniBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, separable_conv_num, start_with_relu=True):
        super().__init__()
        layer_list = []
        
        # start
        if start_with_relu:
            layer_list.append(nn.ReLU(inplace=True))
        layer_list.append(SeparableConv2d(in_channels, out_channels, 3, 1))
        # middle
        for _ in range(separable_conv_num - 1):
            layer_list.extend(
                        [
                            nn.ReLU(inplace=True),
                            SeparableConv2d(out_channels, out_channels, 3, 1),
                        ]
                    )
        # end
        if stride > 1:
            self.shortcut = vn_layer.Conv2dBatchReLU(in_channels, out_channels, 1, 2)
            layer_list.append(nn.MaxPool2d(3, stride, 1))
        else:
            self.shortcut = nn.Sequential()

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        y = self.layers(x) + self.shortcut(x)
        return y


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, block_layer, repeat):
        super().__init__()
        layer_list = []
        layer_list.append(MiniBlock(in_channels, out_channels, 1, block_layer))
        for idx in range(repeat - 1):
            layer = MiniBlock(out_channels, out_channels, 1, block_layer)
            layer_list.append(layer)
        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        y = self.layers(x)
        return y
