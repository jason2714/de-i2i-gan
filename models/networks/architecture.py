import logging
import torch
from torch import nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from .normalization import SPADE


def get_act_layer(act_str):
    act_layer = nn.Identity()
    if act_str == 'leaky_relu':
        act_layer = nn.LeakyReLU(0.2, inplace=True)
    elif act_str == 'relu':
        act_layer = nn.ReLU(inplace=True)
    elif act_str == 'sigmoid':
        act_layer = nn.Sigmoid()
    elif act_str == 'tanh':
        act_layer = nn.Tanh()
    elif act_str is None:
        logging.info('create conv block without activation layer')
    else:
        raise NameError(f'activation layer named {act_str} not defined')
    return act_layer


class DeConvBlock(nn.Module):
    def __init__(self, f_in, f_out,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=0,
                 padding_mode='zeros',
                 bias=False,
                 up_scale=True,
                 norm_layer=None,
                 act_layer=None,
                 use_spectral=False,
                 add_noise=False):
        """
            valid_padding_strings = {'same', 'valid', int}
            valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
            valid_activation_strings = {'leaky_relu', 'relu', 'sigmoid', 'tanh'}
        """
        super(DeConvBlock, self).__init__()
        blocks = []
        if up_scale:
            blocks.append(nn.Upsample(scale_factor=2))
        # print(type(f_in), type(f_out))
        blocks.append(nn.Conv2d(f_in, f_out,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                padding_mode=padding_mode,
                                bias=bias))

        # adaptive noise insert
        if add_noise:
            blocks.append(NoiseInjection())
        # add normalization layer
        if norm_layer is not None:
            blocks.append(norm_layer(f_out))

        # add activation layer
        blocks.append(get_act_layer(act_layer))

        if use_spectral:
            for idx, layer in enumerate(blocks):
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    blocks[idx] = spectral_norm(layer)
        self.de_conv_block = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.de_conv_block(x)
        return out


class ConvBlock(nn.Module):
    def __init__(self, f_in, f_out,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=0,
                 padding_mode='zeros',
                 bias=False,
                 norm_layer=None,
                 act_layer=None,
                 use_spectral=False,
                 add_noise=False):
        """
            valid_padding_strings = {'same', 'valid'}
            valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
            valid_activation_strings = {'leaky_relu', 'relu', 'sigmoid', 'tanh'}
        """
        super(ConvBlock, self).__init__()
        blocks = [nn.Conv2d(f_in, f_out,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            padding_mode=padding_mode,
                            bias=bias)]

        # adaptive noise insert
        if add_noise:
            blocks.append(NoiseInjection())

        # add normalization layer
        if norm_layer is not None:
            blocks.append(norm_layer(f_out))

        # add activation layer
        blocks.append(get_act_layer(act_layer))

        if use_spectral:
            for idx, layer in enumerate(blocks):
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    blocks[idx] = spectral_norm(layer)

        self.conv_block = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.conv_block(x)
        return out


class ResBlock(nn.Module):
    def __init__(self, f_in, f_out,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=0,
                 padding_mode='zeros',
                 bias=False,
                 norm_layer=nn.InstanceNorm2d,
                 act_layer='relu',
                 use_spectral=False,
                 add_noise=False):
        """
            valid_padding_strings = {'same', 'valid'}
            valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
            valid_activation_strings = {'leaky_relu', 'relu', 'sigmoid', 'tanh'}
        """
        super(ResBlock, self).__init__()
        blocks = [ConvBlock(f_in, f_out,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            padding_mode=padding_mode,
                            bias=bias,
                            norm_layer=norm_layer,
                            act_layer=act_layer,
                            use_spectral=use_spectral,
                            add_noise=add_noise),
                  ConvBlock(f_out, f_out,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            padding_mode=padding_mode,
                            bias=bias,
                            norm_layer=norm_layer,
                            act_layer=None,
                            use_spectral=use_spectral,
                            add_noise=add_noise)]
        self.res_block = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.res_block(x)
        return out + x


class SPADEResBlock(nn.Module):
    def __init__(self, label_nc, f_in, f_out,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=0,
                 padding_mode='zeros',
                 bias=False,
                 up_scale=False,
                 norm_layer=nn.InstanceNorm2d,
                 act_layer='relu',
                 use_spectral=False,
                 add_noise=False):
        """
            valid_padding_strings = {'same', 'valid', int}
            valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
            valid_activation_strings = {'leaky_relu', 'relu', 'sigmoid', 'tanh'}
        """
        super(SPADEResBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.up_scale = up_scale
        self.add_noise = add_noise
        self.noise_0 = NoiseInjection()
        self.noise_1 = NoiseInjection()

        f_mid = min(f_in, f_out)
        self.spade_norm_0 = SPADE(label_nc, f_in,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  norm_layer=norm_layer)
        self.spade_norm_1 = SPADE(label_nc, f_mid,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  norm_layer=norm_layer)
        self.act_0 = get_act_layer(act_layer)
        self.act_1 = get_act_layer(act_layer)
        self.conv_0 = nn.Conv2d(f_in, f_mid,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                padding_mode=padding_mode,
                                bias=bias)
        self.conv_1 = nn.Conv2d(f_mid, f_out,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                padding_mode=padding_mode,
                                bias=bias)
        if use_spectral:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)

    def forward(self, x, seg):
        if self.up_scale:
            x = self.up(x)
        x = self.spade_norm_0(x, seg)
        x = self.act_0(x)
        x = self.conv_0(x)
        if self.add_noise:
            x = self.noise_0(x)
        x = self.spade_norm_1(x, seg)
        x = self.act_1(x)
        x = self.conv_1(x)
        if self.add_noise:
            x = self.noise_1(x)
        return x


class NoiseInjection(nn.Module):
    def __init__(self, weight_type='constant', nc=None):
        super(NoiseInjection, self).__init__()
        if weight_type == 'constant':
            self.weight = nn.Parameter(torch.zeros(1))
        elif weight_type == 'vector':
            assert nc is not None, "num_channel shouldn't be None"
            self.weight = nn.Parameter(torch.zeros(nc))
        else:
            raise NameError(f'weight type named {weight_type} not defined')

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        self.weight = torch.reshape(self.weight, (1, -1, 1, 1))
        return image + self.weight * noise

