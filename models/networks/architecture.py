import logging
import math

import torch
from torch import nn
# from torch.nn.utils.parametrizations import spectral_norm
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from .normalization import SPADE, SEAN, AdaIN


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


class NormConvBlock(nn.Module):
    def __init__(self, style_norm_block_type, hidden_nc,
                 label_nc, f_in, f_out,
                 embed_nc=None,
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
        super(NormConvBlock, self).__init__()

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

        self.style_norm_block_type = style_norm_block_type
        if style_norm_block_type == 'spade':
            self.norm = SPADE(label_nc, f_in, hidden_nc=hidden_nc,
                              kernel_size=(3, 3),
                              padding=padding,
                              norm_layer=norm_layer)
        elif style_norm_block_type == 'sean':
            assert embed_nc is not None, 'embed_nc must be specified for SEAN'
            self.norm = SEAN(embed_nc, f_in, label_nc,
                             hidden_nc=hidden_nc, norm_layer=norm_layer)
        elif style_norm_block_type == 'adain':
            self.norm = AdaIN(f_in, hidden_nc=hidden_nc,
                              norm_layer=norm_layer, denorm_type='linear')
        else:
            raise ValueError('Unknown style norm block type: {}'.format(style_norm_block_type))
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

    def forward(self, x, labels, style_feat=None):
        x = self.up(x)

        out = self.noise(self.conv(self.act(self.norm_forward(x, labels, style_feat))))
        return out

    def norm_forward(self, x, labels, style_feat=None):
        if self.style_norm_block_type == 'sean':
            x = self.norm(x, labels, style_feat)
        elif self.style_norm_block_type == 'spade':
            x = self.norm(x, labels)
        elif self.style_norm_block_type == 'adain':
            x = self.norm(x, style_feat)
        return x

    def update_alpha(self, epoch, num_epochs):
        self.norm.update_alpha(epoch, num_epochs)


class NormResBlock(nn.Module):
    def __init__(self, style_norm_block_type, hidden_nc,
                 label_nc, f_in, f_out,
                 embed_nc=None,
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
        super(NormResBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.up_scale = up_scale
        if add_noise:
            self.noise_0 = NoiseInjection()
            self.noise_1 = NoiseInjection()
        else:
            self.noise_0 = nn.Identity()
            self.noise_1 = nn.Identity()

        f_mid = min(f_in, f_out)
        self.style_norm_block_type = style_norm_block_type
        if style_norm_block_type == 'spade':
            self.norm_0 = SPADE(label_nc, f_in, hidden_nc=hidden_nc,
                                kernel_size=kernel_size,
                                padding=padding,
                                norm_layer=norm_layer)
            self.norm_1 = SPADE(label_nc, f_mid, hidden_nc=hidden_nc,
                                kernel_size=kernel_size,
                                padding=padding,
                                norm_layer=norm_layer)
            self.norm_s = SPADE(label_nc, f_in, hidden_nc=hidden_nc,
                                kernel_size=kernel_size,
                                padding=padding,
                                norm_layer=norm_layer)
        elif style_norm_block_type == 'sean':
            assert embed_nc is not None, 'embed_nc must be specified for SEAN'
            self.norm_0 = SEAN(embed_nc, f_in, label_nc, hidden_nc=hidden_nc, norm_layer=norm_layer)
            self.norm_1 = SEAN(embed_nc, f_mid, label_nc, hidden_nc=hidden_nc, norm_layer=norm_layer)
            self.norm_s = SEAN(embed_nc, f_in, label_nc, hidden_nc=hidden_nc, norm_layer=norm_layer)
        elif style_norm_block_type == 'adain':
            self.norm_0 = AdaIN(f_in, hidden_nc=hidden_nc, norm_layer=norm_layer, denorm_type='linear')
            self.norm_1 = AdaIN(f_mid, hidden_nc=hidden_nc, norm_layer=norm_layer, denorm_type='linear')
            self.norm_s = AdaIN(f_in, hidden_nc=hidden_nc, norm_layer=norm_layer, denorm_type='linear')
        else:
            raise ValueError('Unknown style norm block type: {}'.format(style_norm_block_type))
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

    def forward(self, x, labels, style_feat=None):
        if self.up_scale:
            x = self.up(x)
        x_s = self.shortcut(x, labels, style_feat)
        x = self.noise_0(self.conv_0(self.act(self.norm_forward(self.norm_0, x, labels, style_feat))))
        x = self.noise_1(self.conv_1(self.act(self.norm_forward(self.norm_1, x, labels, style_feat))))

        return x + x_s

    def shortcut(self, x, labels, style_feat=None):
        if self.up_scale:
            x_s = self.conv_s(self.norm_forward(self.norm_s, x, labels, style_feat))
        else:
            x_s = x
        return x_s

    def update_alpha(self, epoch, num_epochs):
        self.norm_0.update_alpha(epoch, num_epochs)
        self.norm_1.update_alpha(epoch, num_epochs)
        self.norm_s.update_alpha(epoch, num_epochs)

    def norm_forward(self, norm_layer, x, labels, style_feat=None):
        if self.style_norm_block_type == 'sean':
            x = norm_layer(x, labels, style_feat)
        elif self.style_norm_block_type == 'spade':
            x = norm_layer(x, labels)
        elif self.style_norm_block_type == 'adain':
            x = norm_layer(x, style_feat)
        return x


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


class MaskToken(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        # self.mask_token = torch.nn.Parameter(torch.empty(1, opt.input_nc, opt.image_size, opt.image_size)
        #                                      .normal_(mean=0, std=opt.init_variance))
        # self.mask_token = torch.nn.Parameter(torch.empty(1, 1, opt.image_size, opt.image_size)
        #                                      .normal_(mean=0, std=opt.init_variance))
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, opt.image_size, opt.image_size))
        # self.mask_token = torch.nn.Parameter(torch.empty(1, 1, 1, 1).normal_(mean=0, std=opt.init_variance))

    def forward(self, x, masks):
        return x + self.mask_token * (1 - masks)


class StyleEncoder(torch.nn.Module):
    def __init__(self, embed_nc, hidden_nc):
        super().__init__()
        self.mlp_shared = nn.Sequential(nn.Linear(embed_nc, hidden_nc),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_nc, hidden_nc),
                                        nn.ReLU(inplace=True))

    def forward(self, feat):
        if feat.dim() == 3:
            feat = feat.mean(dim=1)
        return self.mlp_shared(feat)


class LatentDecoder(torch.nn.Module):
    def __init__(self, label_nc, hidden_nc, latent_dim):
        super().__init__()
        self.mlp_latent = nn.Sequential(nn.Linear(latent_dim, hidden_nc // 2),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_nc // 2, hidden_nc),
                                        nn.ReLU(inplace=True))
        self.noise_dim = latent_dim - label_nc

    def forward(self, labels):
        if labels.dim() == 4:
            labels = labels.view(labels.size(0), -1)
        noise = torch.randn(labels.size(0), self.noise_dim).to(labels.device)
        latent = torch.cat([labels, noise], dim=1)
        return self.mlp_latent(latent)


# TODO fix skip_conn
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
        self.up_conv = NormConvBlock(label_nc, inner_dim, crt_dim,
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
        # decoder
        for i in range(num_res // 2, num_res):
            dec_res_blk.append(NormResBlock(label_nc, crt_dim, crt_dim,
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

# class SPADEConvBlock(nn.Module):
#     def __init__(self, label_nc, f_in, f_out,
#                  kernel_size=(3, 3),
#                  stride=(1, 1),
#                  padding=0,
#                  padding_mode='zeros',
#                  bias=False,
#                  up_scale=False,
#                  norm_layer=nn.InstanceNorm2d,
#                  act_layer='relu',
#                  use_spectral=False,
#                  add_noise=False):
#         """
#             valid_padding_strings = {'same', 'valid', int}
#             valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
#             valid_activation_strings = {'leaky_relu', 'relu', 'sigmoid', 'tanh'}
#         """
#         super(SPADEConvBlock, self).__init__()
#
#         # whether to up sample input image
#         if up_scale:
#             self.up = nn.Upsample(scale_factor=2)
#         else:
#             self.up = nn.Identity()
#
#         # whether to inject noise
#         if add_noise:
#             self.noise = NoiseInjection()
#         else:
#             self.noise = nn.Identity()
#
#         self.spade_norm = SPADE(label_nc, f_in,
#                                 kernel_size=(3, 3),
#                                 padding=padding,
#                                 norm_layer=norm_layer)
#         self.conv = nn.Conv2d(f_in, f_out,
#                               kernel_size=kernel_size,
#                               stride=stride,
#                               padding=padding,
#                               padding_mode=padding_mode,
#                               bias=bias)
#
#         # add activation layer
#         self.act = get_act_layer(act_layer)
#
#         if use_spectral:
#             self.conv = spectral_norm(self.conv)
#
#     def forward(self, x, seg):
#         x = self.up(x)
#         x = self.spade_norm(x, seg)
#         out = self.noise(self.conv(self.act(x)))
#         return out
#
#
# class SEANConvBlock(nn.Module):
#     def __init__(self, embed_nc, label_nc, f_in, f_out,
#                  kernel_size=(3, 3),
#                  stride=(1, 1),
#                  padding=0,
#                  padding_mode='zeros',
#                  bias=False,
#                  up_scale=False,
#                  norm_layer=nn.InstanceNorm2d,
#                  act_layer='relu',
#                  use_spectral=False,
#                  add_noise=False):
#         """
#             valid_padding_strings = {'same', 'valid', int}
#             valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
#             valid_activation_strings = {'leaky_relu', 'relu', 'sigmoid', 'tanh'}
#         """
#         super(SEANConvBlock, self).__init__()
#
#         # whether to up sample input image
#         if up_scale:
#             self.up = nn.Upsample(scale_factor=2)
#         else:
#             self.up = nn.Identity()
#
#         # whether to inject noise
#         if add_noise:
#             self.noise = NoiseInjection()
#         else:
#             self.noise = nn.Identity()
#
#         self.sean_norm = SEAN(embed_nc, f_in, label_nc, norm_layer=norm_layer)
#         self.conv = nn.Conv2d(f_in, f_out,
#                               kernel_size=kernel_size,
#                               stride=stride,
#                               padding=padding,
#                               padding_mode=padding_mode,
#                               bias=bias)
#
#         # add activation layer
#         self.act = get_act_layer(act_layer)
#
#         if use_spectral:
#             self.conv = spectral_norm(self.conv)
#
#     def forward(self, x, labels, style_feat=None):
#         x = self.up(x)
#         x = self.sean_norm(x, labels, style_feat)
#         out = self.noise(self.conv(self.act(x)))
#         return out
#
#     def update_alpha(self, epoch, num_epochs):
#         self.sean_norm.update_alpha(epoch, num_epochs)
#
#
# class SPADEResBlock(nn.Module):
#     def __init__(self, label_nc, f_in, f_out,
#                  kernel_size=(3, 3),
#                  stride=(1, 1),
#                  padding=0,
#                  padding_mode='zeros',
#                  bias=False,
#                  up_scale=False,
#                  norm_layer=nn.InstanceNorm2d,
#                  act_layer='relu',
#                  use_spectral=False,
#                  add_noise=False):
#         """
#             valid_padding_strings = {'same', 'valid', int}
#             valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
#             valid_activation_strings = {'leaky_relu', 'relu', 'sigmoid', 'tanh'}
#         """
#         super(SPADEResBlock, self).__init__()
#         self.up = nn.Upsample(scale_factor=2)
#         self.up_scale = up_scale
#         if add_noise:
#             self.noise_0 = NoiseInjection()
#             self.noise_1 = NoiseInjection()
#         else:
#             self.noise_0 = nn.Identity()
#             self.noise_1 = nn.Identity()
#
#         f_mid = min(f_in, f_out)
#         self.spade_norm_0 = SPADE(label_nc, f_in,
#                                   kernel_size=kernel_size,
#                                   padding=padding,
#                                   norm_layer=norm_layer)
#         self.spade_norm_1 = SPADE(label_nc, f_mid,
#                                   kernel_size=kernel_size,
#                                   padding=padding,
#                                   norm_layer=norm_layer)
#         self.spade_norm_s = SPADE(label_nc, f_in,
#                                   kernel_size=kernel_size,
#                                   padding=padding,
#                                   norm_layer=norm_layer)
#         self.act = get_act_layer(act_layer)
#         self.conv_0 = nn.Conv2d(f_in, f_mid,
#                                 kernel_size=kernel_size,
#                                 stride=stride,
#                                 padding=padding,
#                                 padding_mode=padding_mode,
#                                 bias=bias)
#         self.conv_1 = nn.Conv2d(f_mid, f_out,
#                                 kernel_size=kernel_size,
#                                 stride=stride,
#                                 padding=padding,
#                                 padding_mode=padding_mode,
#                                 bias=bias)
#         self.conv_s = nn.Conv2d(f_in, f_out,
#                                 kernel_size=kernel_size,
#                                 stride=stride,
#                                 padding=padding,
#                                 padding_mode=padding_mode,
#                                 bias=bias)
#         if use_spectral:
#             self.conv_0 = spectral_norm(self.conv_0)
#             self.conv_1 = spectral_norm(self.conv_1)
#             self.conv_s = spectral_norm(self.conv_s)
#
#     def forward(self, x, seg):
#         if self.up_scale:
#             x = self.up(x)
#         x_s = self.shortcut(x, seg)
#         x = self.noise_0(self.conv_0(self.act(self.spade_norm_0(x, seg))))
#         x = self.noise_1(self.conv_1(self.act(self.spade_norm_1(x, seg))))
#
#         return x + x_s
#
#     def shortcut(self, x, seg):
#         if self.up_scale:
#             x_s = self.conv_s(self.spade_norm_s(x, seg))
#         else:
#             x_s = x
#         return x_s
#
#
# class SEANResBlock(nn.Module):
#     def __init__(self, embed_nc, label_nc, f_in, f_out,
#                  kernel_size=(3, 3),
#                  stride=(1, 1),
#                  padding=0,
#                  padding_mode='zeros',
#                  bias=False,
#                  up_scale=False,
#                  norm_layer=nn.InstanceNorm2d,
#                  act_layer='relu',
#                  use_spectral=False,
#                  add_noise=False):
#         """
#             valid_padding_strings = {'same', 'valid', int}
#             valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
#             valid_activation_strings = {'leaky_relu', 'relu', 'sigmoid', 'tanh'}
#         """
#         super(SEANResBlock, self).__init__()
#         self.up = nn.Upsample(scale_factor=2)
#         self.up_scale = up_scale
#         if add_noise:
#             self.noise_0 = NoiseInjection()
#             self.noise_1 = NoiseInjection()
#         else:
#             self.noise_0 = nn.Identity()
#             self.noise_1 = nn.Identity()
#
#         f_mid = min(f_in, f_out)
#         self.sean_norm_0 = SEAN(embed_nc, f_in, label_nc, norm_layer=norm_layer)
#         self.sean_norm_1 = SEAN(embed_nc, f_mid, label_nc, norm_layer=norm_layer)
#         self.sean_norm_s = SEAN(embed_nc, f_in, label_nc, norm_layer=norm_layer)
#         self.act = get_act_layer(act_layer)
#         self.conv_0 = nn.Conv2d(f_in, f_mid,
#                                 kernel_size=kernel_size,
#                                 stride=stride,
#                                 padding=padding,
#                                 padding_mode=padding_mode,
#                                 bias=bias)
#         self.conv_1 = nn.Conv2d(f_mid, f_out,
#                                 kernel_size=kernel_size,
#                                 stride=stride,
#                                 padding=padding,
#                                 padding_mode=padding_mode,
#                                 bias=bias)
#         self.conv_s = nn.Conv2d(f_in, f_out,
#                                 kernel_size=kernel_size,
#                                 stride=stride,
#                                 padding=padding,
#                                 padding_mode=padding_mode,
#                                 bias=bias)
#         if use_spectral:
#             self.conv_0 = spectral_norm(self.conv_0)
#             self.conv_1 = spectral_norm(self.conv_1)
#             self.conv_s = spectral_norm(self.conv_s)
#
#     def forward(self, x, labels, style_feat=None):
#         if self.up_scale:
#             x = self.up(x)
#         x_s = self.shortcut(x, labels, style_feat)
#         x = self.noise_0(self.conv_0(self.act(self.sean_norm_0(x, labels, style_feat))))
#         x = self.noise_1(self.conv_1(self.act(self.sean_norm_1(x, labels, style_feat))))
#
#         return x + x_s
#
#     def shortcut(self, x, labels, style_feat=None):
#         if self.up_scale:
#             x_s = self.conv_s(self.sean_norm_s(x, labels, style_feat))
#         else:
#             x_s = x
#         return x_s
#
#     def update_alpha(self, epoch, num_epochs):
#         self.sean_norm_0.update_alpha(epoch, num_epochs)
#         self.sean_norm_1.update_alpha(epoch, num_epochs)
#         self.sean_norm_s.update_alpha(epoch, num_epochs)
