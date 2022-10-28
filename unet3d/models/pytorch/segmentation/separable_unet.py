from torch import nn as nn
from .unet import *
from ..classification.separable import SeparableResidualBlock
from ..classification.myronenko import *
from ..classification import resnet
from functools import partial

from ..classification.isensee import HookBasedFeatureExtractor
from torch.autograd import Variable


class SeparableUnetEncoder(nn.Module):
    def __init__(self,n_features, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoResidualBlock,
                 feature_dilation=2, downsampling_stride=2, dropout=0.2, layer_widths=None, kernel_size=3):
        super(SeparableUnetEncoder, self).__init__()
        
        if layer_blocks is None:
            layer_blocks = [2, 2, 2, 2, 2]

        self.layers = nn.ModuleList()
        self.downsampling_convolutions = nn.ModuleList()

        in_width = n_features

        for i, n_blocks in enumerate(layer_blocks):
            if layer_widths is not None:
                out_width = layer_widths[i]
            else:
                out_width = base_width * (feature_dilation ** i)
            if dropout and i == 0:
                layer_dropout = dropout
            else:
                layer_dropout = None

            if i <= len(layer_blocks) - 4:
                self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=in_width, planes=out_width,
                                        dropout=layer_dropout, kernel_size=kernel_size))
            else:
                self.layers.append(layer(n_blocks=n_blocks, block=SeparableResidualBlock, in_planes=in_width, planes=out_width,
                                        dropout=layer_dropout, kernel_size=kernel_size))
            if i != len(layer_blocks) - 1:
                self.downsampling_convolutions.append(conv3x3x3(out_width, out_width, stride=downsampling_stride,
                                                                kernel_size=kernel_size))
            print("Encoder {}:".format(i), in_width, out_width)
            in_width = out_width

    def forward(self, x):
        outputs = list()
        for layer, downsampling in zip(self.layers[:-1], self.downsampling_convolutions):
            x = layer(x)
            outputs.insert(0, x)
            x = downsampling(x)
        x = self.layers[-1](x)
        outputs.insert(0, x)
        return outputs


class SeparableUnetDecoder(nn.Module):
    def __init__(self, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoResidualBlock,
                 upsampling_scale=2, feature_reduction_scale=2, upsampling_mode="trilinear", align_corners=False,
                 layer_widths=None, use_transposed_convolutions=False, kernel_size=3):
        super(SeparableUnetDecoder, self).__init__()
        self.use_transposed_convolutions = use_transposed_convolutions
        if layer_blocks is None:
            self.layer_blocks = [1, 1, 1, 1]
        else:
            self.layer_blocks = layer_blocks
        self.layers = nn.ModuleList()
        self.pre_upsampling_blocks = nn.ModuleList()
        if use_transposed_convolutions:
            self.upsampling_blocks = nn.ModuleList()
        else:
            self.upsampling_blocks = list()
        self.base_width = base_width
        self.feature_reduction_scale = feature_reduction_scale
        self.layer_widths = layer_widths
        for i, n_blocks in enumerate(self.layer_blocks):
            depth = len(self.layer_blocks) - (i + 1)
            in_width, out_width = self.calculate_layer_widths(depth)

            if depth != 0:
                if depth > 3:

                    self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=in_width, planes=in_width,
                                            kernel_size=kernel_size))
                else:
                    self.layers.append(layer(n_blocks=n_blocks, block=SeparableResidualBlock, in_planes=in_width, planes=in_width,
                                            kernel_size=kernel_size))

                if self.use_transposed_convolutions:
                    self.pre_upsampling_blocks.append(nn.Sequential())
                    self.upsampling_blocks.append(nn.ConvTranspose3d(in_width, out_width, kernel_size=kernel_size,
                                                                     stride=upsampling_scale, padding=1))
                else:
                    self.pre_upsampling_blocks.append(resnet.conv1x1x1(in_width, out_width, stride=1))
                    self.upsampling_blocks.append(partial(nn.functional.interpolate, scale_factor=upsampling_scale,
                                                          mode=upsampling_mode, align_corners=align_corners))
            else:
                self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=in_width, planes=out_width,
                                         kernel_size=kernel_size))
    
    def _calculate_layer_widths(self, depth):
        if self.layer_widths is not None:
            out_width = self.layer_widths[depth]
            in_width = self.layer_widths[depth + 1]
        else:
            if depth > 0:
                out_width = int(self.base_width * (self.feature_reduction_scale ** (depth - 1)))
                in_width = out_width * self.feature_reduction_scale
            else:
                out_width = self.base_width
                in_width = self.base_width
        return in_width, out_width

    def calculate_layer_widths(self, depth):
        in_width, out_width = self._calculate_layer_widths(depth=depth)
        if depth != len(self.layer_blocks) - 1:
            in_width *= 2
        print("Decoder {}:".format(depth), in_width, out_width)
        return in_width, out_width

    def forward(self, inputs):
        x = inputs[0]
        for i, (pre, up, lay) in enumerate(zip(self.pre_upsampling_blocks, self.upsampling_blocks, self.layers[:-1])):
            x = lay(x)
            x = pre(x)
            x = up(x)
            x = torch.cat((x, inputs[i + 1]), 1)
        x = self.layers[-1](x)
        return x


class SeparableUnet(UNet):
    def __init__(self, *args, encoder_class=SeparableUnetEncoder, decoder_class=SeparableUnetDecoder, **kwargs):
        super(SeparableUnet, self).__init__(
            *args, encoder_class=encoder_class, decoder_class=decoder_class, **kwargs)
    def get_feature_maps(self, input, layer_name, upscale):
        feature_extractor = HookBasedFeatureExtractor(self, layer_name, upscale)
        return feature_extractor.forward(input)


if __name__ == '__main__':
    unet = SeparableUnet(2)

    print (unet)