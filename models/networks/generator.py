from models.networks.base_network import BaseNetwork
from models.networks.architecture import DeConvBlock, ConvBlock, ResBlock, \
    SPADEResBlock, SPADEConvBlock, UnetBlock, ResnetSubModule
import torch
from torch import nn
import math


class WGanGenerator(BaseNetwork):
    def __init__(self, num_layers, ngf=64, noise_dim=100):
        super().__init__()
        self.noise_dim = noise_dim
        model = [nn.Upsample(scale_factor=2)]  # noise_dim x 2 x 2

        current_dim = ngf * (2 ** num_layers)
        model += [DeConvBlock(noise_dim, current_dim,
                              kernel_size=(4, 4),
                              padding='same',
                              norm_layer=nn.BatchNorm2d,
                              act_layer='relu')]  # 512 x 4 x 4
        for i in range(num_layers):
            model += [DeConvBlock(current_dim,
                                  current_dim // 2,
                                  kernel_size=(4, 4),
                                  stride=(1, 1),
                                  padding='same',
                                  norm_layer=nn.BatchNorm2d,
                                  act_layer='relu')]
            current_dim //= 2
        model += [nn.Upsample(scale_factor=2),
                  nn.Conv2d(current_dim, 3,
                            kernel_size=(4, 4),
                            stride=(1, 1),
                            padding='same',
                            bias=False),
                  nn.Tanh()]
        self.noise_to_image = nn.Sequential(*model)

    def forward(self, x):
        assert isinstance(x, (torch.Tensor, int)), "x must be batch_size (Int) or noise (Torch.Tensor)"
        if isinstance(x, torch.Tensor):
            noise = x
        else:
            noise = torch.rand(x, self.noise_dim, 1, 1)
        noise = noise.to(self.device)
        fake_data = self.noise_to_image(noise)
        return fake_data


class DefectGanGenerator(BaseNetwork):
    def __init__(self, label_nc, input_nc=3, num_scales=2, num_res=6, ngf=64,
                 use_spectral=True, add_noise=True, cycle_gan=False, skip_conn=False):
        """
            image to image translation network

        """
        super().__init__()
        # DONE ADD SPADE, SPATIAL_DISTRIBUTION_MAP, ADAPTIVE_NOISE
        assert (num_res & 1) == 0, 'num_res must be even'
        self.cycle_gan = cycle_gan
        self.label_nc = label_nc
        self.skip_conn = skip_conn

        # stem
        crt_dim = ngf
        self.stem = ConvBlock(input_nc, crt_dim,
                              kernel_size=(7, 7),
                              padding='same',
                              padding_mode='reflect',
                              norm_layer=nn.BatchNorm2d,
                              act_layer='leaky_relu',
                              use_spectral=use_spectral)

        if skip_conn:
            # original kernel size is 7
            crt_dim *= (2 ** num_scales)
            unet_blk = ResnetSubModule(label_nc, crt_dim,
                                       num_res=num_res, use_spectral=use_spectral, add_noise=add_noise)
            for i in range(num_scales):
                crt_dim //= 2
                unet_blk = UnetBlock(label_nc, crt_dim,
                                     submodule=unet_blk,
                                     skip_conn=skip_conn,
                                     innermost=(i == 0),
                                     kernel_sizes=((4, 4), (3, 3)),
                                     strides=((2, 2), (1, 1)),
                                     paddings=(1, 'same'),
                                     padding_mode='reflect',
                                     norm_layers=(nn.BatchNorm2d, nn.InstanceNorm2d),
                                     act_layers=('leaky_relu', 'relu'),
                                     use_spectral=use_spectral,
                                     add_noise=add_noise)
            self.unet_blk = unet_blk
            crt_dim *= 2
        else:
            # original blocks
            self.enc_blk = []
            self.dec_blk = []
            self.enc_res_blk = []
            self.dec_res_blk = []
            conv_blk = []
            de_conv_blk = []
            enc_res_blk = []
            dec_res_blk = []

            for i in range(num_scales):
                conv_blk.append(ConvBlock(crt_dim, crt_dim * 2,
                                          kernel_size=(4, 4),
                                          stride=(2, 2),
                                          padding=1,
                                          padding_mode='reflect',
                                          norm_layer=nn.BatchNorm2d,
                                          act_layer='leaky_relu',
                                          use_spectral=use_spectral))
                crt_dim *= 2

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

            for i in range(num_scales):
                de_conv_blk.append(SPADEConvBlock(label_nc, crt_dim, crt_dim // 2,
                                                  kernel_size=(3, 3),
                                                  stride=(1, 1),
                                                  padding='same',
                                                  padding_mode='reflect',
                                                  up_scale=True,
                                                  norm_layer=nn.InstanceNorm2d,
                                                  act_layer='relu',
                                                  use_spectral=use_spectral,
                                                  add_noise=add_noise))
                crt_dim //= 2

            self.enc_blk = nn.Sequential(*conv_blk)
            self.enc_res_blk = nn.Sequential(*enc_res_blk)
            self.dec_res_blk = nn.Sequential(*dec_res_blk)
            self.dec_blk = nn.Sequential(*de_conv_blk)

        # head
        self.foreground_head = DeConvBlock(crt_dim, 3,
                                           kernel_size=(3, 3),
                                           padding='same',
                                           padding_mode='reflect',
                                           up_scale=False,
                                           norm_layer=None,
                                           act_layer='tanh',
                                           use_spectral=False,
                                           add_noise=False)
        self.distribution_head = DeConvBlock(crt_dim, 1,
                                             kernel_size=(3, 3),
                                             padding='same',
                                             padding_mode='reflect',
                                             up_scale=False,
                                             norm_layer=None,
                                             act_layer='sigmoid',
                                             use_spectral=False,
                                             add_noise=False)

    def forward(self, x, labels):
        assert isinstance(x, torch.Tensor), "x must be Original Images: Torch.Tensor"
        # expand labels' shape to the same as data
        # x, labels = x.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

        feat = self.stem(x)

        if self.skip_conn:
            feat = self.unet_blk(feat, labels)
        else:
            # original
            for enc_blk in self.enc_blk:
                feat = enc_blk(feat, labels)

            # inner residual block section
            for enc_res_blk in self.enc_res_blk:
                feat = enc_res_blk(feat, labels)
            for dec_res_blk in self.dec_res_blk:
                feat = dec_res_blk(feat, labels)
            # inner residual block section

            for dec_blk in self.dec_blk:
                feat = dec_blk(feat, labels)

        foreground = self.foreground_head(feat)
        spatial_prob = self.distribution_head(feat)
        output = x * (1 - spatial_prob) + foreground * spatial_prob

        if self.cycle_gan:
            return foreground, spatial_prob
        else:
            return output, spatial_prob
