import torch
from torch import nn
import torch.nn.functional as F
import math
from utils.util import calc_mean_std


class SPADE(nn.Module):
    def __init__(self, label_nc, norm_nc, hidden_nc=128, kernel_size=(3, 3), padding='same', norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.param_free_norm = norm_layer(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.

        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, hidden_nc,
                                                  kernel_size=kernel_size,
                                                  padding=padding),
                                        nn.ReLU(inplace=True))
        self.mlp_gamma = nn.Conv2d(hidden_nc, norm_nc, kernel_size=kernel_size, padding=padding)
        self.mlp_beta = nn.Conv2d(hidden_nc, norm_nc, kernel_size=kernel_size, padding=padding)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class AdaIN(nn.Module):
    def __init__(self, norm_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, denorm_type='linear'):
        super().__init__()

        self.denorm_type = denorm_type

        # The dimension of the intermediate embedding space. Yes, hardcoded.

        self.param_free_norm = norm_layer(norm_nc, affine=False)
        if self.denorm_type == 'linear':
            assert hidden_nc is not None, 'hidden_nc should be set when denorm_type is linear.'
            self.hidden_nc = hidden_nc
            self.mlp_gamma = nn.Linear(hidden_nc, norm_nc)
            self.mlp_beta = nn.Linear(hidden_nc, norm_nc)

    def forward(self, x, style_feat):
        N, C = x.size()[:2]
        assert style_feat.size(1) == C, 'The channel of style feature is not equal to input feature.'

        normalized = self.param_free_norm(x)

        if self.denorm_type == 'linear':
            gamma = self.mlp_gamma(style_feat).view(N, C, 1, 1)
            beta = self.mlp_beta(style_feat).view(N, C, 1, 1)
        elif self.denorm_type == 'stat':
            beta, gamma = calc_mean_std(style_feat)
            gamma -= 1
        else:
            raise NotImplementedError('Not implemented denorm type: {}'.format(self.denorm_type))
        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SEAN(nn.Module):
    def __init__(self, embed_nc, norm_nc, label_nc, hidden_nc=128,
                 latent_dim=16, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.latent_dim = latent_dim
        self.noise_dim = self.latent_dim - label_nc
        self.alpha = 1.
        self.param_free_norm = norm_layer(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.

        self.mlp_shared = nn.Sequential(nn.Linear(embed_nc, hidden_nc), nn.ReLU(inplace=True))
        self.mlp_gamma = nn.Linear(hidden_nc, norm_nc)
        self.mlp_beta = nn.Linear(hidden_nc, norm_nc)

        self.mlp_latent = nn.Sequential(nn.Linear(self.latent_dim, hidden_nc),
                                        nn.ReLU(inplace=True))
        # self.mlp_latent = nn.Sequential(nn.Linear(label_nc + self.latent_dim, hidden_nc // 2),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Linear(hidden_nc // 2, hidden_nc),
        #                                 nn.ReLU(inplace=True))
        # self.mlp_latent = nn.Sequential(nn.Linear(label_nc, hidden_nc),
        #                                 nn.ReLU(inplace=True))
        # self.mlp_latent = nn.Sequential(nn.Linear(label_nc, hidden_nc // 4),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Linear(hidden_nc // 4, hidden_nc // 2),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Linear(hidden_nc // 2, hidden_nc),
        #                                 nn.ReLU(inplace=True))

    def set_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, x, labels, feat=None):
        N, C = x.size()[:2]
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # use noise and labels to produce latent code
        noise = torch.randn(N, self.noise_dim).to(x.device)
        if labels.dim() == 4:
            labels = labels.view(N, -1)
        latent = torch.cat([noise, labels], dim=1)
        latent_code = self.mlp_latent(latent)
        # latent_code = self.mlp_latent(labels)

        # Part 2. produce scaling and bias conditioned on semantic map
        if feat is None:
            mix_feat = latent_code
        else:
            if feat.dim() == 3:
                feat = feat.mean(dim=1)
            enc_feat = self.mlp_shared(feat)
            mix_feat = enc_feat * self.alpha + latent_code * (1 - self.alpha)
            # mix_feat = enc_feat

            # replace style embed with latent code if style embed is all zeros
            mask_indices = (feat == 0).all(dim=1).view(-1, 1)
            mix_feat = mix_feat * ~mask_indices + latent_code * mask_indices

        gamma = self.mlp_gamma(mix_feat).view(N, C, 1, 1)
        beta = self.mlp_beta(mix_feat).view(N, C, 1, 1)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

# class MSEAN(nn.Module):
#     def __init__(self, embed_nc, norm_nc, label_nc,
#                  latent_dim=10, norm_layer=nn.BatchNorm2d, use_embed_only=False):
#         super().__init__()
#
#         self.latent_dim = latent_dim
#         self.alpha = 1.
#         self.use_embed_only = use_embed_only
#         self.param_free_norm = norm_layer(norm_nc, affine=False)
#         # The dimension of the intermediate embedding space. Yes, hardcoded.
#         hidden_nc = 128
#
#         self.mlp_shared = nn.Linear(embed_nc, norm_nc)
#
#         self.mlp_latent = nn.Sequential(nn.Linear(label_nc + latent_dim, hidden_nc // 2),
#                                         nn.ReLU(inplace=True),
#                                         nn.Linear(hidden_nc // 2, hidden_nc),
#                                         nn.ReLU(inplace=True))
#         self.eps = 1e-5
#
#     def update_alpha(self, epoch, num_epochs):
#         """
#         Update alpha for mixing latent code and semantic map
#         Must be called after each epoch
#         """
#         if not self.use_embed_only:
#             self.alpha = (1 + math.cos(math.pi * epoch / num_epochs)) / 2
#
#     def forward(self, x, labels, feats=None):
#         # Part 1. generate parameter-free normalized activations
#         normalized = self.param_free_norm(x)
#         enc_feats = self.mlp_shared(feats)
#         beta, gamma = calc_embed_mean_std(enc_feats, eps=self.eps)
#
#         # apply scale and bias
#         out = normalized * gamma + beta
#
#         return out
