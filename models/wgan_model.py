from models.networks.generator import WGanGenerator
from models.networks.discriminator import WGanDiscriminator
from models.base_model import BaseModel
import torch
import math


class WGanModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        image_size = opt.image_size
        assert image_size & (image_size - 1) == 0, 'Image size must be a power of 2'
        num_layers = int(math.log(image_size, 2)) - 3
        self.netG = WGanGenerator(num_layers=num_layers, ngf=opt.ngf, noise_dim=opt.noise_dim).to(opt.device)
        self.netD = WGanDiscriminator(num_layers=num_layers, ndf=opt.ndf).to(opt.device)
        self.clipping_limit = opt.clipping_limit

    def weight_clipping(self):
        # Clamp parameters to a range [-c, c], c=self.clipping_limit
        for p in self.netD.parameters():
            p.data.clamp_(-self.clipping_limit, self.clipping_limit)





