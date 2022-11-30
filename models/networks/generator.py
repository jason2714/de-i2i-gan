from models.networks.base_network import BaseNetwork
from models.networks.architecture import DeConvBlock, ConvBlock, ResBlock, SPADEResBlock
import torch
from torch import nn
import math


class WGanGenerator(BaseNetwork):
    def __init__(self, num_layers, ngf=64, noise_dim=100):
        super().__init__()
        self.noise_dim = noise_dim
        model = [nn.Upsample(scale_factor=2)]  # noise_dim x 2 x 2

        current_dim = ngf * (2 ** num_layers)
        model += [DeConvBlock(noise_dim, current_dim, kernel_size=(4, 4), padding='same')]  # 512 x 4 x 4
        for i in range(num_layers):
            model += [DeConvBlock(current_dim,
                                  current_dim // 2,
                                  kernel_size=(4, 4),
                                  stride=(1, 1),
                                  padding='same')]
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
    def __init__(self, label_nc, num_scales=2, num_res=6, ngf=64, input_dim=3, use_spectral=True):
        """
            image to image translation network

        """
        super().__init__()
        # TODO SPADE, SPATIAL CONTROL MAP, ADAPTIVE NOISE
        assert (num_res & 1) == 0, 'num_res must be even'
        crt_dim = ngf
        # self.noise_dim = noise_dim
        self.enc_blk = []
        self.dec_blk = []
        conv_blk = []
        de_conv_blk = []
        enc_res_blk = []
        dec_res_blk = []
        # encoder
        stem = ConvBlock(input_dim, crt_dim,
                         kernel_size=(7, 7),
                         padding='same',
                         padding_mode='reflect',
                         norm_layer=nn.BatchNorm2d,
                         act_layer='leaky_relu',
                         use_spectral=use_spectral)
        for i in range(num_scales):
            conv_blk.append(ConvBlock(crt_dim, crt_dim * 2,
                                      kernel_size=(4, 4),
                                      stride=(2, 2),
                                      padding='same',
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

        # decoder
        for i in range(num_res // 2, num_res):
            # dec_res_blk.append(ResBlock(crt_dim, crt_dim,
            #                             kernel_size=(3, 3),
            #                             stride=(1, 1),
            #                             padding='same',
            #                             padding_mode='reflect',
            #                             norm_layer=nn.InstanceNorm2d,
            #                             act_layer='relu',
            #                             use_spectral=use_spectral))
            dec_res_blk.append(SPADEResBlock(label_nc, crt_dim, crt_dim,
                                             kernel_size=(3, 3),
                                             stride=(1, 1),
                                             padding='same',
                                             padding_mode='reflect',
                                             up_scale=False,
                                             norm_layer=nn.InstanceNorm2d,
                                             act_layer='relu'))
        for i in range(num_scales):
            # de_conv_blk.append(DeConvBlock(crt_dim, crt_dim // 2,
            #                                kernel_size=(4, 4),
            #                                stride=(1, 1),
            #                                padding='same',
            #                                padding_mode='reflect',
            #                                norm_layer=nn.InstanceNorm2d,
            #                                act_layer='relu',
            #                                use_spectral=use_spectral))
            de_conv_blk.append(SPADEResBlock(label_nc, crt_dim, crt_dim // 2,
                                             kernel_size=(3, 3),
                                             stride=(1, 1),
                                             padding='same',
                                             padding_mode='reflect',
                                             up_scale=True,
                                             norm_layer=nn.InstanceNorm2d,
                                             act_layer='relu'))
            crt_dim //= 2
        # original kernel size is 7
        head = DeConvBlock(crt_dim, 3,
                           kernel_size=(3, 3),
                           padding='same',
                           padding_mode='reflect',
                           norm_layer=None,
                           act_layer='tanh',
                           use_spectral=False)

        # for skip and mix
        # skip_conv_blk.append(ConvBlock(crt_dim * 2, crt_dim,
        #                               kernel_size=(3, 3),
        #                               stride=(1, 1),
        #                               padding='same',
        #                               padding_mode='reflect',
        #                               norm_layer=nn.InstanceNorm2d,
        #                               act_layer='relu',
        #                               use_spectral=False))
        # mix_conv_blk = []
        # skip_conv_blk = []
        self.enc_blk = [stem, *conv_blk, *enc_res_blk]
        self.dec_blk = [*dec_res_blk, *de_conv_blk, head]

    def forward(self, x, label):
        assert isinstance(x, torch.Tensor), "x must be Original Images: Torch.Tensor"
        for enc_blk in self.enc_blk:
            x = enc_blk(x)
        for dec_blk in self.dec_blk:
            x = dec_blk(x, label)
        return x
