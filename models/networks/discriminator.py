import numpy as np
import torch
from torch import nn
from models.networks.architecture import ConvBlock
from models.networks.base_network import BaseNetwork


class WGanDiscriminator(BaseNetwork):
    def __init__(self, num_layers, ndf=64):
        super().__init__()
        current_dim = ndf
        stem = []
        backbone = []
        classifier = []
        stem += [ConvBlock(3, ndf,
                           kernel_size=(7, 7),
                           stride=(2, 2),
                           padding=3,
                           padding_mode='reflect',
                           norm_layer=nn.BatchNorm2d,
                           act_layer='relu'),
                 nn.MaxPool2d(kernel_size=(3, 3),
                              stride=(2, 2),
                              padding=1)]
        for i in range(num_layers):
            backbone += [ConvBlock(current_dim, current_dim * 2,
                                   kernel_size=(3, 3),
                                   stride=(2, 2),
                                   padding=1,
                                   bias=False,
                                   norm_layer=nn.BatchNorm2d,
                                   act_layer='relu')]
            current_dim *= 2
        backbone += [nn.AdaptiveAvgPool2d(1)]
        classifier += [nn.Linear(current_dim, 1)]
        self.feature_extractor = nn.Sequential(*(stem + backbone))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.to(self.device)
        features = self.feature_extractor(x)
        logits = self.classifier(features.view(batch_size, -1))
        return logits


class DefectGanDiscriminator(BaseNetwork):
    def __init__(self, label_nc, image_size=128, input_nc=3, num_layers=5, ndf=64, use_spectral=True):
        """
            image to image translation network

        """
        super().__init__()
        self.enc_blk = []
        conv_blk = []
        crt_dim = ndf

        stem = ConvBlock(input_nc, crt_dim,
                         kernel_size=(4, 4),
                         stride=(2, 2),
                         padding=1,
                         padding_mode='reflect',
                         norm_layer=None,
                         act_layer='leaky_relu',
                         use_spectral=use_spectral)
        for i in range(num_layers):
            conv_blk.append(ConvBlock(crt_dim, crt_dim * 2,
                                      kernel_size=(4, 4),
                                      stride=(2, 2),
                                      padding=1,
                                      padding_mode='reflect',
                                      norm_layer=None,
                                      act_layer='leaky_relu',
                                      use_spectral=use_spectral))
            crt_dim *= 2
        kernel_size = image_size // np.power(2, num_layers + 1)
        self.enc_blk = nn.Sequential(stem, *conv_blk)
        self.cls_clf = ConvBlock(crt_dim, label_nc,
                                 kernel_size=(kernel_size, kernel_size),
                                 norm_layer=None,
                                 act_layer=None)
        self.src_clf = ConvBlock(crt_dim, 1,
                                 kernel_size=(3, 3),
                                 stride=(1, 1),
                                 padding='same',
                                 padding_mode='reflect',
                                 norm_layer=None,
                                 act_layer=None)

    def forward(self, x):
        assert isinstance(x, torch.Tensor), "x must be Original Images: Torch.Tensor"
        # x = x.to(self.device, non_blocking=True)
        feat = self.enc_blk(x)
        src_logits = self.src_clf(feat)
        cls_logits = self.cls_clf(feat)
        return src_logits, cls_logits.reshape((cls_logits.size(0), cls_logits.size(1)))


class ViTClassifier(BaseNetwork):
    def __init__(self, label_nc):
        super().__init__()
        self.clf = nn.Linear(768, label_nc)

    def forward(self, x):
        return self.clf(x)
