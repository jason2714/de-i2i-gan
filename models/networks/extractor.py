import math

import torch

from models.networks.architecture import ConvBlock, ResBlock
from models.networks.base_network import BaseNetwork
from models.networks.architecture import EmbedEncoder, LatentDecoder
from torch import nn


# class StyleExtractor(BaseNetwork):
#     def __init__(self, opt):
#         super(StyleExtractor, self).__init__()
#         self.opt = opt
#         self.latent_decoder = LatentDecoder(opt.label_nc, opt.hidden_nc, opt.latent_dim)
#         self.embed_encoder = EmbedEncoder(opt.embed_nc, opt.hidden_nc)
#         self.alpha = 1.
#
#     def set_alpha(self, alpha):
#         self.alpha = alpha
#
#     def forward(self, labels, feat=None):
#         latent_feat = self.latent_decoder(labels)
#         if feat is None:
#             mix_feat = latent_feat
#         else:
#             enc_feat = self.style_encoder(feat)
#
#             mix_feat = enc_feat * self.alpha + latent_feat * (1 - self.alpha)
#
#             # replace style embed with latent code if style embed is all zeros
#             mask_indices = (feat == 0).all(dim=1).view(-1, 1)
#             mix_feat = mix_feat * ~mask_indices + latent_feat * mask_indices
#         return mix_feat

class StyleExtractor(BaseNetwork):
    def __init__(self, opt):
        super(StyleExtractor, self).__init__()
        assert opt.image_size in (64, 128, 256, 512, 1024), "image size should be one of [64, 128, 256, 512, 1024]"
        num_blocks = int(math.log2(opt.image_size)) - 3
        crt_dim = opt.ndf
        max_dim = 256
        self.sean_alpha = opt.sean_alpha
        self.noise_dim = opt.latent_dim - opt.label_nc
        if opt.sean_alpha == 0:
            layers = [nn.Linear(opt.latent_dim, max_dim), nn.ReLU(inplace=True)]
            for _ in range(3):
                layers += [nn.Linear(max_dim, max_dim), nn.ReLU(inplace=True)]
            layers.append(nn.Linear(max_dim, opt.hidden_nc))
            self.shared = nn.Sequential(*layers)
        elif opt.sean_alpha == 1:
            blocks = [ConvBlock(opt.input_nc, crt_dim,
                                kernel_size=(7, 7),
                                stride=(2, 2),
                                padding=3,
                                padding_mode='reflect',
                                norm_layer=None,
                                act_layer='leaky_relu',
                                use_spectral=False)]
            for _ in range(num_blocks):
                new_dim = min(crt_dim * 2, max_dim)
                blocks.append(ResBlock(crt_dim, new_dim,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding='same',
                                       padding_mode='reflect',
                                       norm_layer=nn.InstanceNorm2d,
                                       act_layer='leaky_relu',
                                       use_spectral=False,
                                       down_scale=True))
                crt_dim = new_dim

            blocks.append(ConvBlock(crt_dim, opt.hidden_nc,
                                    kernel_size=(4, 4),
                                    stride=(1, 1),
                                    padding=0,
                                    norm_layer=None,
                                    act_layer=None,
                                    use_spectral=False))
            self.shared = nn.Sequential(*blocks)
        else:
            raise NotImplementedError("sean_alpha should be 0 or 1")
        # self.unshared = nn.ModuleList()
        # for _ in range(label_nc):
        #     self.unshared.append(nn.Linear(crt_dim, hidden_nc))

    def forward(self, x, labels):
        if self.sean_alpha == 0:
            noise = torch.randn(labels.size(0), self.noise_dim).to(x.device)
            latent = torch.cat([labels, noise], dim=1)
            out = self.shared(latent)
        elif self.sean_alpha == 1:
            out = self.shared(x)
        else:
            raise NotImplementedError("sean_alpha should be 0 or 1")
        return out
