
import torch
from torch.autograd import Variable

from torch import nn as nn
from .resnet import conv3x3x3, conv1x1x1
from . import *
from ..initialization import InitWeights_He, InitWeights_XavierUniform
from functools import partial

# copied from https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/networks/utils.py


class HookBasedFeatureExtractor(nn.Module):
    def __init__(self, submodule, layername, upscale=False):
        super(HookBasedFeatureExtractor, self).__init__()

        self.submodule = submodule
        self.submodule.eval()
        self.layername = layername
        self.outputs_size = None
        self.outputs = None
        self.inputs = None
        self.inputs_size = None
        self.upscale = upscale

    def get_input_array(self, m, i, o):
        if isinstance(i, tuple):
            if isinstance(i[0], list):
                i = i[0]
            self.inputs = [i[index].data.clone() for index in range(len(i))]
            self.inputs_size = [input.size() for input in self.inputs]
        else:
            self.inputs = i.data.clone()
            self.inputs_size = self.input.size()
        print('Input Array Size: ', self.inputs_size)

    def get_output_array(self, m, i, o):
        if isinstance(o, list):
            self.outputs = [o[index].data.clone() for index in range(len(o))]
            self.outputs_size = [output.size() for output in self.outputs]
        else:
            self.outputs = o.clone()
            self.outputs_size = self.outputs.size()
        print('Output Array Size: ', self.outputs_size)

    def rescale_output_array(self, newsize):
        us = nn.Upsample(size=newsize[2:], mode='bilinear')
        if isinstance(self.outputs, list):
            for index in range(len(self.outputs)):
                self.outputs[index] = us(self.outputs[index]).data()
        else:
            self.outputs = us(self.outputs).data()

    def forward(self, x):
        target_layer = self.submodule._modules.get(self.layername)

        # Collect the output tensor
        h_inp = target_layer.register_forward_hook(self.get_input_array)
        h_out = target_layer.register_forward_hook(self.get_output_array)
        self.submodule(x)
        h_inp.remove()
        h_out.remove()

        # Rescale the feature-map if it's required
        if self.upscale:
            self.rescale_output_array(x.size())

        return self.inputs, self.outputs


class IsenseeConvolutionalBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, kernel_size=3):
        super().__init__()
        self.conv = conv3x3x3(in_planes, planes, stride,
                              kernel_size=kernel_size)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.norm = nn.InstanceNorm3d(planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.lrelu(x)
        return x


class IsenseeLayer(nn.Module):
    def __init__(self, n_blocks, block, in_planes, planes, kernel_size=3):
        super().__init__()
        self.block = block
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(
                block(in_planes, planes, kernel_size=kernel_size))
            in_planes = planes

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
        return x


class IsenseeEncoder(nn.Module):
    def __init__(self, n_features, base_width=30, debug=False, layer=IsenseeLayer, block=IsenseeConvolutionalBlock,
                 feature_dilation=2, downsampling_stride=2, kernel_size=3 ):
        super().__init__()
        self.debug = debug
        layer_blocks = [2, 2, 2, 2, 1]
        self.layers = nn.ModuleList()
        self.downsampling_convolutions = nn.ModuleList()
        in_width = n_features
        for i, n_blocks in enumerate(layer_blocks):
            out_width = base_width * (feature_dilation ** i)
            self.layers.append(layer(n_blocks=n_blocks, block=block,
                               in_planes=in_width, planes=out_width, kernel_size=kernel_size))
            if i != len(layer_blocks) - 1:
                self.downsampling_convolutions.append(
                    nn.MaxPool3d(kernel_size, stride=downsampling_stride, padding=1))
            if self.debug:
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


class IsenseeDecoder(nn.Module):
    def __init__(self, base_width=30, layer_blocks=None, layer=IsenseeLayer, block=IsenseeConvolutionalBlock,
                 upsampling_scale=2, feature_reduction_scale=2, upsampling_mode="trilinear", align_corners=False,
                 layer_widths=None, use_transposed_convolutions=False, debug=False, kernal_size=3):
        super().__init__()
        self.debug = debug
        self.base_width = base_width
        self.feature_reduction_scale = feature_reduction_scale
        if layer_blocks is None:
            self.layer_blocks = [1, 1, 1, 1, 1]
        self.layers = nn.ModuleList()
        self.pre_upsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = list()
        for i, n_blocks in enumerate(self.layer_blocks):
            depth = len(self.layer_blocks) - (i + 1)
            in_width, out_width = self.calculate_layer_widths(depth)

            if depth > 0:
                self.pre_upsampling_blocks.append(
                    block(in_width, out_width, kernel_size=kernal_size))

                self.upsampling_blocks.append(partial(nn.functional.interpolate, scale_factor=upsampling_scale,
                                                      mode=upsampling_mode, align_corners=align_corners))

                self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=in_width, planes=out_width,
                                         kernel_size=kernal_size))
            else:
                self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=in_width, planes=out_width,
                                         kernel_size=kernal_size))

    def calculate_layer_widths(self, depth):
        in_width = self.base_width * \
            (self.feature_reduction_scale ** (depth))
        out_width = in_width // 2

        if (depth == 0):
            in_width = self.base_width
            out_width = self.base_width

        if self.debug:
            print("Decoder {}:".format(depth), in_width, out_width)

        return in_width, out_width

    def forward(self, inputs):
        x = inputs[0]
        for i,  (pre, up, lay) in enumerate(zip(self.pre_upsampling_blocks, self.upsampling_blocks, self.layers[:-1])):
            x = pre(x)
            x = up(x)
            x = torch.cat((x, inputs[i + 1]), 1)
            x = lay(x)
        x = self.layers[-1](x)
        return x


class IsenseeNet(nn.Module):
    def __init__(self, *args, n_features=4, base_width=32, encoder_class=IsenseeEncoder, decoder_class=IsenseeDecoder,
                 n_outputs=3, activation="sigmoid", encoder_kwargs={}, decoder_kwargs={}, weightInitializer=InitWeights_He(1e-2), debug=False, **kwargs):
        super().__init__()
        for key in ['block']:
            if key in encoder_kwargs:
                encoder_kwargs[key] = eval(encoder_kwargs[key])

            if key in decoder_kwargs:
                decoder_kwargs[key] = eval(decoder_kwargs[key])

        self.weightInitializer = weightInitializer
        self.encoder = encoder_class(n_features, base_width=base_width, **encoder_kwargs)
        self.decoder = decoder_class(base_width, **decoder_kwargs)
        self.base_width = base_width

        self.set_final_convolution(n_outputs)
        self.set_activation(activation=activation)

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)

    def set_final_convolution(self, n_outputs):
        self.final_convolution = conv1x1x1(
            in_planes=self.base_width, out_planes=n_outputs, stride=1)

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

    def get_feature_maps(self, input, layer_name, upscale):
        feature_extractor = HookBasedFeatureExtractor(
            self, layer_name, upscale)
        return feature_extractor.forward(Variable(input))
