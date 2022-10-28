from torch import nn as nn
from .resnet import conv1x1x1, conv3x3x3
from .isensee import IsenseeConvolutionalBlock, IsenseeLayer
import torch


class NNUNetEncoder(nn.Module):
    def __init__(self, n_features, base_width=32, layer=IsenseeLayer, block=IsenseeConvolutionalBlock,
                 layer_blocks=None, feature_dilation=2, downsampling_stride=2, max_features=320, debug=False):
        super().__init__()
        if layer_blocks is None:
            self.layer_blocks = [2, 2, 2, 2, 2, 2]

        self.debug = False
        self.features = list()
        self.base_width = base_width
        self.layers = nn.ModuleList()
        self.downsampling_convolutions = nn.ModuleList()
        in_width = n_features
        for i, n_blocks in enumerate(self.layer_blocks):
            out_width = base_width * (feature_dilation ** i)

            if out_width > max_features:
                out_width = max_features

            self.layers.append(layer(n_blocks, block, in_width, out_width))

            if i != len(self.layer_blocks) - 1:
                self.downsampling_convolutions.append(
                    nn.Conv3d(out_width, out_width, downsampling_stride,downsampling_stride))
            if self.debug:
                print("Encoder {}:".format(i), in_width, out_width)
            self.features.append((in_width, out_width))
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


class MirroredDecoder(nn.Module):
    def __init__(self, encoder, layer=IsenseeLayer, block=IsenseeConvolutionalBlock, kernel_size=3,  upsampling_stride=2, debug=False):
        super().__init__()
        self.debug = debug
        self.layers = nn.ModuleList()
        self.transpose_convolutions = nn.ModuleList()
        self.layer_blocks = encoder.layer_blocks[:-1]
        self.encoder = encoder
        in_width, out_width = self.calculate_widths(-1)
        for i, n_blocks in enumerate(self.layer_blocks):

            self.transpose_convolutions.append(nn.ConvTranspose3d(
                in_width - out_width, in_width - out_width, kernel_size=upsampling_stride, stride=upsampling_stride))
            self.layers.append(
                layer(n_blocks, block, in_width, out_width))

            depth = len(self.layer_blocks) - (i + 1)
            if self.debug:  
                print("Decoder {}:".format(depth), in_width, out_width)

            in_width, out_width = self.calculate_widths(depth)

    def calculate_widths(self, depth):
        in_width, out_width = self.encoder.features[depth]
        return in_width + out_width, in_width

    def forward(self, outputs):
        x = outputs[0]
        for i,  (upsample, layer) in enumerate(zip(self.transpose_convolutions, self.layers)):
            x = upsample(x)
            x = torch.cat((x, outputs[i + 1]), 1)
            x = layer(x)
        return x


class NNUNet(nn.Module):
    def __init__(self, *args, n_features=4, n_outputs =1, base_width= 32, activation=None,   **kwargs):
        super().__init__()
        self.base_width = base_width
        self.encoder = NNUNetEncoder(n_features, **kwargs)
        self.decoder = MirroredDecoder(self.encoder, **kwargs)

        self.set_final_convolution(n_outputs)
        self.set_activation(activation=activation)


    def set_final_convolution(self, n_outputs):
        self.final_convolution = conv1x1x1(in_planes=self.base_width, out_planes=n_outputs, stride=1)

    def set_activation(self, activation):
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation = nn.Softmax(dim=1)
        else:
            self.activation = None

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_convolution(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
