from torch import nn as nn
from .resnet import conv1x1x1, conv3x3x3
from .separable import SeparableResidualBlock
from .myronenko import MyronenkoResidualBlock

# adapted from https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class SESeparableResidualBlock(SeparableResidualBlock):
    def __init__(self, in_planes, planes, stride=1, norm_layer=None, norm_groups=8, kernel_size=3, reduction=16):
        super(SESeparableResidualBlock, self).__init__(in_planes, planes, stride, norm_layer, norm_groups, kernel_size)
        self.se = SELayer(planes, reduction)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)

        if self.sample is not None:
            identity = self.sample(identity)

        x += identity

        return x


class SEResidualBlock(MyronenkoResidualBlock):
    def __init__(self, in_planes, planes, stride=1, norm_layer=None, norm_groups=8, kernel_size=3, reduction=16):
        super(SEResidualBlock, self).__init__(in_planes, planes, stride, norm_layer, norm_groups, kernel_size)
        self.se = SELayer(planes, reduction)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)

        if self.sample is not None:
            identity = self.sample(identity)

        x += identity

        return x
