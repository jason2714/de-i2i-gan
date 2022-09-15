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
                           padding_mode='reflect'),
                 nn.MaxPool2d(kernel_size=(3, 3),
                              stride=(2, 2),
                              padding=1)]
        for i in range(num_layers):
            backbone += [ConvBlock(current_dim, current_dim * 2,
                                   kernel_size=(3, 3),
                                   stride=(2, 2),
                                   padding=1,
                                   bias=False)]
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
