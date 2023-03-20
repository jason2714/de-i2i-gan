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

    def forward(self, x, seg=None):
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
                 use_spectral=False):
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

    def forward(self, x, seg=None):
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
                 use_spectral=False):
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
                            use_spectral=use_spectral),
                  ConvBlock(f_out, f_out,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            padding_mode=padding_mode,
                            bias=bias,
                            norm_layer=norm_layer,
                            act_layer=None,
                            use_spectral=use_spectral)]
        self.res_block = nn.Sequential(*blocks)

    def forward(self, x, seg=None):
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
        if add_noise:
            self.noise_0 = NoiseInjection()
            self.noise_1 = NoiseInjection()
        else:
            self.noise_0 = nn.Identity()
            self.noise_1 = nn.Identity()

        f_mid = min(f_in, f_out)
        self.spade_norm_0 = SPADE(label_nc, f_in,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  norm_layer=norm_layer)
        self.spade_norm_1 = SPADE(label_nc, f_mid,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  norm_layer=norm_layer)
        self.spade_norm_s = SPADE(label_nc, f_in,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  norm_layer=norm_layer)
        self.act = get_act_layer(act_layer)
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
        self.conv_s = nn.Conv2d(f_in, f_out,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                padding_mode=padding_mode,
                                bias=bias)
        if use_spectral:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            self.conv_s = spectral_norm(self.conv_s)

    def forward(self, x, seg):
        if self.up_scale:
            x = self.up(x)
        x_s = self.shortcut(x, seg)
        x = self.noise_0(self.conv_0(self.act(self.spade_norm_0(x, seg))))
        x = self.noise_1(self.conv_1(self.act(self.spade_norm_1(x, seg))))

        return x + x_s

    def shortcut(self, x, seg):
        if self.up_scale:
            x_s = self.conv_s(self.spade_norm_s(x, seg))
        else:
            x_s = x
        return x_s


class SPADEConvBlock(nn.Module):
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
        super(SPADEConvBlock, self).__init__()

        # whether to up sample input image
        if up_scale:
            self.up = nn.Upsample(scale_factor=2)
        else:
            self.up = nn.Identity()

        # whether to inject noise
        if add_noise:
            self.noise = NoiseInjection()
        else:
            self.noise = nn.Identity()

        self.spade_norm = SPADE(label_nc, f_in,
                                kernel_size=(3, 3),
                                padding=padding,
                                norm_layer=norm_layer)
        self.conv = nn.Conv2d(f_in, f_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              padding_mode=padding_mode,
                              bias=bias)

        # add activation layer
        self.act = get_act_layer(act_layer)

        if use_spectral:
            self.conv = spectral_norm(self.conv)

    def forward(self, x, seg):
        x = self.up(x)
        x = self.spade_norm(x, seg)
        out = self.noise(self.conv(self.act(x)))
        return out


class NoiseInjection(nn.Module):
    def __init__(self, weight_type='constant', nc=None):
        super(NoiseInjection, self).__init__()
        if weight_type == 'constant':
            self.weight = nn.Parameter(torch.zeros(1, 1, 1, 1))
        elif weight_type == 'vector':
            assert nc is not None, "num_channel shouldn't be None"
            self.weight = nn.Parameter(torch.zeros(1, nc, 1, 1))
        else:
            raise NameError(f'weight type named {weight_type} not defined')

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        return image + self.weight * noise


class UnetBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, label_nc, crt_dim,
                 submodule=None,
                 skip_conn=False,
                 innermost=False,
                 kernel_sizes=((4, 4), (3, 3)),
                 strides=((2, 2), (1, 1)),
                 paddings=(1, 1),
                 padding_mode='zeros',
                 bias=False,
                 norm_layers=(nn.BatchNorm2d, nn.InstanceNorm2d),
                 act_layers=('leaky_relu', 'relu'),
                 use_spectral=False,
                 add_noise=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            crt_dim (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            norm_layer          -- normalization layer
        """
        super(UnetBlock, self).__init__()

        self.skip_conn = skip_conn
        self.down_conv = ConvBlock(crt_dim, crt_dim * 2,
                                   kernel_size=kernel_sizes[0],
                                   stride=strides[0],
                                   padding=paddings[0],
                                   padding_mode=padding_mode,
                                   bias=bias,
                                   norm_layer=norm_layers[0],
                                   act_layer=act_layers[0],
                                   use_spectral=use_spectral)
        # conv_blk.append(SPADEConvBlock(label_nc, crt_dim, crt_dim * 2,
        #                                kernel_size=(4, 4),
        #                                stride=(2, 2),
        #                                padding=1,
        #                                padding_mode='reflect',
        #                                up_scale=False,
        #                                norm_layer=nn.BatchNorm2d,
        #                                act_layer='leaky_relu',
        #                                use_spectral=use_spectral,
        #                                add_noise=add_noise))

        if innermost or not skip_conn:
            inner_dim = crt_dim * 2
        else:
            inner_dim = crt_dim * 4
        self.up_conv = SPADEConvBlock(label_nc, inner_dim, crt_dim,
                                      kernel_size=kernel_sizes[1],
                                      stride=strides[1],
                                      padding=paddings[1],
                                      padding_mode=padding_mode,
                                      up_scale=True,
                                      norm_layer=norm_layers[1],
                                      act_layer=act_layers[1],
                                      use_spectral=use_spectral,
                                      add_noise=add_noise)
        self.submodule = submodule

    def forward(self, x, seg):
        feat = self.down_conv(x, seg)
        if self.submodule is not None:
            feat = self.submodule(feat, seg)
        out = self.up_conv(feat, seg)
        if self.skip_conn:  # add skip connections
            # TODO fix the nan bug
            return torch.cat([x, out], 1)
        else:
            return out


class ResnetSubModule(nn.Module):
    def __init__(self, label_nc, crt_dim, num_res=6, use_spectral=True, add_noise=True):
        super().__init__()
        enc_res_blk = []
        dec_res_blk = []
        for i in range(num_res // 2):
            enc_res_blk.append(ResBlock(crt_dim, crt_dim,
                                        kernel_size=(3, 3),
                                        stride=(1, 1),
                                        padding='same',
                                        padding_mode='reflect',
                                        norm_layer=nn.BatchNorm2d,
                                        act_layer='leaky_relu',
                                        use_spectral=use_spectral))
            # enc_res_blk.append(SPADEResBlock(label_nc, crt_dim, crt_dim,
            #                                  kernel_size=(3, 3),
            #                                  stride=(1, 1),
            #                                  padding='same',
            #                                  padding_mode='reflect',
            #                                  up_scale=False,
            #                                  norm_layer=nn.BatchNorm2d,
            #                                  act_layer='leaky_relu',
            #                                  use_spectral=use_spectral,
            #                                  add_noise=add_noise))
        # decoder
        for i in range(num_res // 2, num_res):
            dec_res_blk.append(SPADEResBlock(label_nc, crt_dim, crt_dim,
                                             kernel_size=(3, 3),
                                             stride=(1, 1),
                                             padding='same',
                                             padding_mode='reflect',
                                             up_scale=False,
                                             norm_layer=nn.InstanceNorm2d,
                                             act_layer='relu',
                                             use_spectral=use_spectral,
                                             add_noise=add_noise))
        self.enc_res_blk = nn.Sequential(*enc_res_blk)
        self.dec_res_blk = nn.Sequential(*dec_res_blk)

    def forward(self, feat, seg):
        for enc_res_blk in self.enc_res_blk:
            feat = enc_res_blk(feat, seg)
        for dec_res_blk in self.dec_res_blk:
            feat = dec_res_blk(feat, seg)
        return feat
