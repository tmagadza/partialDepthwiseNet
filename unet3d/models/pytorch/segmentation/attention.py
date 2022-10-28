import torch
from torch import nn
from torch.nn import functional as F
from functools import partial

from .separable_unet import SeparableUnet, SeparableUnetEncoder, SeparableUnetDecoder
from ..classification.myronenko import MyronenkoLayer, MyronenkoResidualBlock
from ..classification.isensee import conv1x1x1, SeparableResidualBlock, IsenseeNet
# adapted from https://github.com/bnsreenu/python_for_microscopists/blob/master/224_225_226_models.py


class GatingSignal(nn.Module):
    def __init__(self, in_planes, planes, norm=False):
        super(GatingSignal, self).__init__()

        self.conv = conv1x1x1(in_planes, planes)
        self.relu = nn.ReLU(inplace=True)
        if(norm):
            self.norm1 = nn.BatchNorm3d(in_planes)
        else:
            self.norm1 = nn.Sequential()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm1(x)
        x = self.relu(x)

        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_g, out_g, in_x, out_x, upsampling_mode='trilinear', align_corners=False):
        super(AttentionBlock, self).__init__()

        self.phi_g = conv1x1x1(in_g, out_g)
        self.theta_x = conv1x1x1(in_x, out_x, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.psi = conv1x1x1(out_g, 1)
        self.sigmoid = nn.Sigmoid()
        self.upsample = partial(
            F.interpolate, scale_factor=2, mode=upsampling_mode, align_corners=align_corners)

    def forward(self, gating, x):
        gating = self.phi_g(gating)
        att = self.theta_x(x)
        att = self.relu(att + gating)
        att = self.psi(att)
        att = self.sigmoid(att)
        att = self.upsample(att)

        x = att.expand_as(x) * x
        return x


class AttentionSeperableDecoder(nn.Module):
    def __init__(self, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoResidualBlock,
                 upsampling_scale=2, upsampling_mode="trilinear", align_corners=False,
                 layer_widths=None,  use_attention=True, kernel_size=3, debug= True,):
        super(AttentionSeperableDecoder,self).__init__()

        if layer_blocks is None:
            self.layer_blocks = [1, 1, 1, 1]
        else:
            self.layer_blocks = layer_blocks
        self.debug = debug
        self.use_attention = use_attention
        self.base_width = base_width
        self.layer_widths = layer_widths

        self.layers = nn.ModuleList()
        self.pre_upsampling_blocks = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.upsampling_blocks = list()

        for i, n_blocks in enumerate(self.layer_blocks):
            depth = len(self.layer_blocks) - (i + 1)
            in_width, out_width = self.calculate_layer_widths(depth)

            if depth > 2:
                self.layers.append(layer(n_blocks=n_blocks, block=SeparableResidualBlock, in_planes=in_width, planes=out_width,
                                         kernel_size=kernel_size))
            else:
                self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=in_width, planes=out_width,
                                         kernel_size=kernel_size))

            self.pre_upsampling_blocks.append(
                conv1x1x1(in_width, out_width, stride=1))
            self.upsampling_blocks.append(partial(nn.functional.interpolate, scale_factor=upsampling_scale,
                                                  mode=upsampling_mode, align_corners=align_corners))

            if use_attention:
                self.attention.append(AttentionBlock(
                    in_width, out_width, out_width, out_width))
            else:
                self.attention.append(nn.Sequential())

    def calculate_layer_widths(self, depth):
        out_width = self.base_width * (2**depth)
        in_width = out_width * 2

        if self.debug:
            print("Decoder {}:".format(depth), in_width, out_width)
        return in_width, out_width

    def forward(self, inputs):
        x = inputs[0].clone()
        for i, (pre, up, att, lay) in enumerate(zip(self.pre_upsampling_blocks, self.upsampling_blocks, self.attention, self.layers)):
            if self.use_attention:
                a = att(x, inputs[i + 1])
            x = pre(x)
            x = up(x)

            if self.use_attention:
                x = torch.cat((x, a), 1)
            else:
                x = torch.cat((x, inputs[i + 1]), 1)

            x = lay(x)

        return x


class AttentionUnet(IsenseeNet):
    def __init__(self,  *args, encoder_class=SeparableUnetEncoder, decoder_class=AttentionSeperableDecoder, **kwargs):
        super().__init__(*args, encoder_class=encoder_class,
                         decoder_class=decoder_class, **kwargs)

if __name__ == '__main__':
    input = torch.randn((1,4,128,128,128))
    model = AttentionUnet()

    pred = model(input)
