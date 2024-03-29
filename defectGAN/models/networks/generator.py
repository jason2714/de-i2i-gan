from collections import defaultdict

from models.networks.base_network import BaseNetwork
from models.networks.architecture import DeConvBlock, ConvBlock, ResBlock, UnetBlock, ResnetSubModule, SEAN, \
    NormConvBlock, NormResBlock
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
    def __init__(self, opt):
        """
            image to image translation network

        """
        super().__init__()
        assert (opt.num_res & 1) == 0, 'num_res must be even'
        self.opt = opt
        self.cycle_gan = opt.cycle_gan
        self.label_nc = opt.label_nc
        self.skip_conn = opt.skip_conn

        # stem
        crt_dim = opt.ngf
        self.stem = ConvBlock(opt.input_nc, crt_dim,
                              kernel_size=(7, 7),
                              padding='same',
                              padding_mode='reflect',
                              norm_layer=nn.BatchNorm2d,
                              act_layer='leaky_relu',
                              use_spectral=opt.use_spectral)

        if opt.skip_conn:
            # original kernel size is 7
            crt_dim *= (2 ** opt.num_scales)
            unet_blk = ResnetSubModule(opt.label_nc, crt_dim,
                                       num_res=opt.num_res, use_spectral=opt.use_spectral, add_noise=opt.add_noise)
            for i in range(opt.num_scales):
                crt_dim //= 2
                unet_blk = UnetBlock(opt.label_nc, crt_dim,
                                     submodule=unet_blk,
                                     skip_conn=opt.skip_conn,
                                     innermost=(i == 0),
                                     kernel_sizes=((4, 4), (3, 3)),
                                     strides=((2, 2), (1, 1)),
                                     paddings=(1, 'same'),
                                     padding_mode='reflect',
                                     norm_layers=(nn.BatchNorm2d, nn.InstanceNorm2d),
                                     act_layers=('leaky_relu', 'relu'),
                                     use_spectral=opt.use_spectral,
                                     add_noise=opt.add_noise)
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

            for i in range(opt.num_scales):
                conv_blk.append(ConvBlock(crt_dim, crt_dim * 2,
                                          kernel_size=(4, 4),
                                          stride=(2, 2),
                                          padding=1,
                                          padding_mode='reflect',
                                          norm_layer=nn.BatchNorm2d,
                                          act_layer='leaky_relu',
                                          use_spectral=opt.use_spectral))
                crt_dim *= 2

            for i in range(opt.num_res // 2):
                enc_res_blk.append(ResBlock(crt_dim, crt_dim,
                                            kernel_size=(3, 3),
                                            stride=(1, 1),
                                            padding='same',
                                            padding_mode='reflect',
                                            norm_layer=nn.BatchNorm2d,
                                            act_layer='leaky_relu',
                                            use_spectral=opt.use_spectral))
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
            for i in range(opt.num_res // 2, opt.num_res):
                dec_res_blk.append(NormResBlock(opt.style_norm_block_type, opt.hidden_nc,
                                                opt.label_nc, crt_dim, crt_dim,
                                                style_distill=opt.style_distill,
                                                embed_nc=opt.embed_nc,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding='same',
                                                padding_mode='reflect',
                                                up_scale=False,
                                                norm_layer=nn.InstanceNorm2d,
                                                act_layer='relu',
                                                use_spectral=opt.use_spectral,
                                                add_noise=opt.add_noise))
                # if opt.style_norm_block_type == 'spade':
                #     dec_res_blk.append(SPADEResBlock(opt.label_nc, crt_dim, crt_dim,
                #                                      kernel_size=(3, 3),
                #                                      stride=(1, 1),
                #                                      padding='same',
                #                                      padding_mode='reflect',
                #                                      up_scale=False,
                #                                      norm_layer=nn.InstanceNorm2d,
                #                                      act_layer='relu',
                #                                      use_spectral=opt.use_spectral,
                #                                      add_noise=opt.add_noise))
                # elif opt.style_norm_block_type == 'sean':
                #     dec_res_blk.append(SEANResBlock(opt.embed_nc, opt.label_nc, crt_dim, crt_dim,
                #                                     kernel_size=(3, 3),
                #                                     stride=(1, 1),
                #                                     padding='same',
                #                                     padding_mode='reflect',
                #                                     up_scale=False,
                #                                     norm_layer=nn.InstanceNorm2d,
                #                                     act_layer='relu',
                #                                     use_spectral=opt.use_spectral,
                #                                     add_noise=opt.add_noise))
                # else:
                #     raise NotImplementedError(f'style norm block type [{opt.style_norm_block_type}] is not implemented')

            for i in range(opt.num_scales):
                de_conv_blk.append(NormConvBlock(opt.style_norm_block_type, opt.hidden_nc,
                                                 opt.label_nc, crt_dim, crt_dim // 2,
                                                 style_distill=opt.style_distill,
                                                 embed_nc=opt.embed_nc,
                                                 kernel_size=(3, 3),
                                                 stride=(1, 1),
                                                 padding='same',
                                                 padding_mode='reflect',
                                                 up_scale=True,
                                                 norm_layer=nn.InstanceNorm2d,
                                                 act_layer='relu',
                                                 use_spectral=opt.use_spectral,
                                                 add_noise=opt.add_noise))
                # if opt.style_norm_block_type == 'spade':
                #     de_conv_blk.append(SPADEConvBlock(opt.label_nc, crt_dim, crt_dim // 2,
                #                                       kernel_size=(3, 3),
                #                                       stride=(1, 1),
                #                                       padding='same',
                #                                       padding_mode='reflect',
                #                                       up_scale=True,
                #                                       norm_layer=nn.InstanceNorm2d,
                #                                       act_layer='relu',
                #                                       use_spectral=opt.use_spectral,
                #                                       add_noise=opt.add_noise))
                # elif opt.style_norm_block_type == 'sean':
                #     de_conv_blk.append(SEANConvBlock(opt.embed_nc, opt.label_nc, crt_dim, crt_dim // 2,
                #                                      kernel_size=(3, 3),
                #                                      stride=(1, 1),
                #                                      padding='same',
                #                                      padding_mode='reflect',
                #                                      up_scale=True,
                #                                      norm_layer=nn.InstanceNorm2d,
                #                                      act_layer='relu',
                #                                      use_spectral=opt.use_spectral,
                #                                      add_noise=opt.add_noise))
                # else:
                #     raise NotImplementedError(f'style norm block type [{opt.style_norm_block_type}] is not implemented')
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

    def forward(self, x, labels, style_feat=None):
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
                feat = dec_res_blk(feat, labels, style_feat)
            # inner residual block section

            for dec_blk in self.dec_blk:
                feat = dec_blk(feat, labels, style_feat)
        if feat.isnan().any():
            feat.nan_to_num_()
        foreground = self.foreground_head(feat)
        spatial_prob = self.distribution_head(feat)
        output = x * (1 - spatial_prob) + foreground * spatial_prob

        if self.cycle_gan:
            return foreground, spatial_prob
        else:
            return output, spatial_prob

    def update_per_epoch(self, epoch):
        super(DefectGanGenerator, self).update_per_epoch(epoch)
        alpha = (1 + math.cos(math.pi * epoch / self.opt.num_epochs)) / 2
        if self.opt.style_norm_block_type == 'sean':
            if self.opt.sean_alpha is None:
                self.set_sean_alpha(alpha)
            if self.opt.use_running_stats:
                self.update_stats()

    def set_sean_alpha(self, alpha):
        for attr_name, attr_value in self.named_modules():
            if isinstance(attr_value, SEAN):
                attr_value.set_alpha(alpha)

    def enable_sean_distill_loss(self, enable_distill_loss):
        for attr_name, attr_value in self.named_modules():
            if isinstance(attr_value, SEAN):
                attr_value.distill_loss = enable_distill_loss

    def get_sean_distill_loss(self):
        distill_losses = {'latent': [], 'embed': []}
        for attr_name, attr_value in self.named_modules():
            if isinstance(attr_value, SEAN):
                if attr_value.distill_loss:
                    for loss_type in distill_losses.keys():
                        distill_losses[loss_type] += attr_value.distill_loss[loss_type]
        for loss_type in distill_losses.keys():
            if distill_losses[loss_type]:
                distill_losses[loss_type] = torch.stack(distill_losses[loss_type]).mean()
        return distill_losses

    def update_stats(self):
        for attr_name, attr_value in self.named_modules():
            if isinstance(attr_value, SEAN):
                attr_value.update_stats()

    @property
    def track_running_stats(self):
        for attr_name, attr_value in self.named_modules():
            if isinstance(attr_value, SEAN):
                return attr_value.track_running_stats

    @track_running_stats.setter
    def track_running_stats(self, track_running_stats):
        for attr_name, attr_value in self.named_modules():
            if isinstance(attr_value, SEAN):
                attr_value.track_running_stats = track_running_stats

    @property
    def inference_running_stats(self):
        for attr_name, attr_value in self.named_modules():
            if isinstance(attr_value, SEAN):
                return attr_value.inference_running_stats

    @inference_running_stats.setter
    def inference_running_stats(self, inference_running_stats):
        for attr_name, attr_value in self.named_modules():
            if isinstance(attr_value, SEAN):
                attr_value.inference_running_stats = inference_running_stats


class StarGANv2Generator(BaseNetwork):
    def __init__(self, opt):
        """
            image to image translation network
        """
        super().__init__()
        assert (opt.num_res & 1) == 0, 'num_res must be even'
        self.opt = opt
        self.label_nc = opt.label_nc

        # stem
        crt_dim = opt.ngf
        max_dim = 512
        self.stem = ConvBlock(opt.input_nc, crt_dim,
                              kernel_size=(7, 7),
                              padding='same',
                              padding_mode='reflect',
                              norm_layer=nn.InstanceNorm2d,
                              act_layer='leaky_relu',
                              use_spectral=opt.use_spectral)

        # original blocks
        self.down_enc_blk = nn.ModuleList()
        self.up_dec_blk = nn.ModuleList()
        self.enc_res_blk = nn.ModuleList()
        self.dec_res_blk = nn.ModuleList()
        for i in range(opt.num_scales):
            new_dim = min(crt_dim * 2, max_dim)
            self.down_enc_blk.append(ResBlock(crt_dim, new_dim,
                                              kernel_size=(3, 3),
                                              stride=(1, 1),
                                              padding='same',
                                              padding_mode='reflect',
                                              norm_layer=nn.InstanceNorm2d,
                                              act_layer='leaky_relu',
                                              use_spectral=opt.use_spectral,
                                              down_scale=True))
            self.up_dec_blk.append(NormConvBlock(opt.style_norm_block_type, opt.hidden_nc,
                                                 opt.label_nc, new_dim, crt_dim,
                                                 embed_nc=opt.embed_nc,
                                                 kernel_size=(3, 3),
                                                 stride=(1, 1),
                                                 padding='same',
                                                 padding_mode='reflect',
                                                 up_scale=True,
                                                 norm_layer=nn.InstanceNorm2d,
                                                 act_layer='relu',
                                                 use_spectral=opt.use_spectral,
                                                 add_noise=opt.add_noise))
            crt_dim = new_dim
        for i in range(opt.num_res // 2):
            self.enc_res_blk.append(ResBlock(crt_dim, crt_dim,
                                             kernel_size=(3, 3),
                                             stride=(1, 1),
                                             padding='same',
                                             padding_mode='reflect',
                                             norm_layer=nn.InstanceNorm2d,
                                             act_layer='leaky_relu',
                                             use_spectral=opt.use_spectral))
            self.dec_res_blk.append(NormResBlock(opt.style_norm_block_type, opt.hidden_nc,
                                                 opt.label_nc, crt_dim, crt_dim,
                                                 embed_nc=opt.embed_nc,
                                                 kernel_size=(3, 3),
                                                 stride=(1, 1),
                                                 padding='same',
                                                 padding_mode='reflect',
                                                 up_scale=False,
                                                 norm_layer=nn.InstanceNorm2d,
                                                 act_layer='relu',
                                                 use_spectral=opt.use_spectral,
                                                 add_noise=opt.add_noise))
            # head
            self.head = DeConvBlock(opt.ngf, 3,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    padding_mode='reflect',
                                    up_scale=False,
                                    norm_layer=None,
                                    act_layer='tanh',
                                    use_spectral=False,
                                    add_noise=False)

    def forward(self, x, labels, style_feat=None):
        assert isinstance(x, torch.Tensor), "x must be Original Images: Torch.Tensor"
        # expand labels' shape to the same as data
        feat = self.stem(x)
        for enc_blk in self.down_enc_blk:
            feat = enc_blk(feat, labels)
        # inner residual block section
        for enc_res_blk in self.enc_res_blk:
            feat = enc_res_blk(feat, labels)
        for dec_res_blk in self.dec_res_blk:
            feat = dec_res_blk(feat, labels, style_feat)
        # inner residual block section
        for dec_blk in self.up_dec_blk:
            feat = dec_blk(feat, labels, style_feat)

        if feat.isnan().any():
            feat.nan_to_num_()
        output = self.foreground_head(feat)
        return output
