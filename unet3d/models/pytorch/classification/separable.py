from torch import nn as nn
from .resnet import conv1x1x1, conv3x3x3
from .myronenko import MyronenkoConvolutionBlock


class SeparableConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv3d, self).__init__()
        self.depthwise = conv3x3x3(in_channels, in_channels, kernel_size=kernel_size, 
                                groups=in_channels, padding=1)
        self.pointwise = conv1x1x1(in_channels, out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SeparableConvolutionBlock(MyronenkoConvolutionBlock):
    def __init__(self,  in_planes, planes, stride=1, norm_layer=None, norm_groups=8, kernel_size=3):
        super(SeparableConvolutionBlock,self).__init__(in_planes=in_planes, planes=planes, stride=stride,
                                               norm_layer=norm_layer,
                                               norm_groups=norm_groups, kernel_size=kernel_size)
        self.conv = SeparableConv3d(in_planes, planes, kernel_size)


class SeparableResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, norm_layer=None, norm_groups=8, kernel_size=3):
        super(SeparableResidualBlock, self).__init__()
        self.conv1 = SeparableConvolutionBlock(in_planes=in_planes, planes=planes, stride=stride,
                                               norm_layer=norm_layer,
                                               norm_groups=norm_groups, kernel_size=kernel_size)
        self.conv2 = SeparableConvolutionBlock(in_planes=planes, planes=planes, stride=stride, norm_layer=norm_layer,
                                               norm_groups=norm_groups, kernel_size=kernel_size)
        if in_planes != planes:
            self.sample = conv1x1x1(in_planes, planes)
        else:
            self.sample = None

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        if self.sample is not None:
            identity = self.sample(identity)

        x += identity

        return x