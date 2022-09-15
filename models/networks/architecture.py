from torch import nn
import torch.nn.utils.spectral_norm as spectral_norm


class DeConvBlock(nn.Module):
    def __init__(self, f_in, f_out,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=0,
                 padding_mode='zeros',
                 bias=False,
                 up_scale=True):
        super(DeConvBlock, self).__init__()
        blocks = []
        if up_scale:
            blocks += [nn.Upsample(scale_factor=2)]
        # print(type(f_in), type(f_out))
        blocks += [nn.Conv2d(f_in, f_out,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
                             padding_mode=padding_mode,
                             bias=bias),
                   nn.InstanceNorm2d(f_out),
                   nn.ReLU(inplace=True)]
        self.de_conv_block = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.de_conv_block(x)
        return out


class ConvBlock(nn.Module):
    def __init__(self, f_in, f_out,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=0,
                 padding_mode='zeros',
                 bias=False):
        super(ConvBlock, self).__init__()
        blocks = []
        blocks += [nn.Conv2d(f_in, f_out,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
                             padding_mode=padding_mode,
                             bias=bias),
                   nn.BatchNorm2d(f_out),
                   nn.LeakyReLU(0.2, inplace=True)]
        self.conv_block = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.conv_block(x)
        return out
