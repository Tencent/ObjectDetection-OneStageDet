import torch.nn as nn
from collections import OrderedDict

from ... import layer as vn_layer


class Conv2dDepthWise(nn.Module):
    """ This layer implements the depthwise separable convolution from Mobilenets_.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution

    .. _Mobilenets: https://arxiv.org/pdf/1704.04861.pdf
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv2dDepthWise, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii/2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size/2)

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, self.kernel_size, self.stride, self.padding, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),

            vn_layer.Conv2dBatchReLU(in_channels, out_channels, 1, 1),
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x
