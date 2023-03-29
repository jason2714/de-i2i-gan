import torch
from torch import nn
import torch.nn.functional as F


class SPADE(nn.Module):
    def __init__(self, label_nc, norm_nc, kernel_size=(3, 3), padding='same', norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.param_free_norm = norm_layer(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden,
                                                  kernel_size=kernel_size,
                                                  padding=padding),
                                        nn.ReLU(inplace=True))
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=padding)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=padding)

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


class SEAN(nn.Module):
    def __init__(self, embed_nc, norm_nc, label_nc, latent_dim=10, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.latent_dim = latent_dim
        self.alpha = 1.
        self.param_free_norm = norm_layer(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        self.mlp_shared = nn.Sequential(nn.Linear(embed_nc, nhidden), nn.ReLU(inplace=True))
        self.mlp_gamma = nn.Linear(nhidden, norm_nc)
        self.mlp_beta = nn.Linear(nhidden, norm_nc)

        self.mlp_latent = nn.Sequential(nn.Linear(label_nc + latent_dim, nhidden // 2),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(nhidden // 2, nhidden),
                                        nn.ReLU(inplace=True))

    def forward(self, x, labels, feat=None, alpha=None):
        N, C = x.size()[:2]
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # use noise and labels to produce latent code
        noise = torch.randn(N, self.latent_dim).to(x.device)
        latent = torch.cat([noise, labels], dim=1)
        latent_code = self.mlp_latent(latent)

        # Part 2. produce scaling and bias conditioned on semantic map
        if feat is None:
            mix_feat = latent_code
        else:
            feat = feat.view(N, -1)
            tmp_alpha = self.alpha if alpha is None else alpha
            mix_feat = feat * tmp_alpha + latent_code * (1 - tmp_alpha)
        actv = self.mlp_shared(mix_feat)
        gamma = self.mlp_gamma(actv).view(N, C, 1, 1)
        beta = self.mlp_beta(actv).view(N, C, 1, 1)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out
