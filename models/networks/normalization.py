from torch import nn
from .architecture import ConvBlock
import torch.nn.functional as F


class SPADE(nn.Module):
    def __init__(self, label_nc, norm_nc, kernel_size=(3, 3), padding='same', norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.param_free_norm = norm_layer(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        self.mlp_shared = ConvBlock(label_nc, nhidden,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    norm_layer=None,
                                    act_layer='relu')
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
