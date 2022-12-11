from models.networks.generator import DefectGanGenerator
from models.networks.discriminator import DefectGanDiscriminator
from models.base_model import BaseModel
import torch
import math


class DefectGanModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        image_size = opt.image_size
        assert image_size & (image_size - 1) == 0, 'Image size must be a power of 2'
        self.netG = DefectGanGenerator(label_nc=opt.label_nc,
                                       input_nc=opt.input_nc,
                                       num_scales=opt.num_scales,
                                       num_res=opt.num_res,
                                       ngf=opt.ngf,
                                       use_spectral=opt.use_spectral,
                                       add_noise=opt.add_noise).to(opt.device)
        self.netD = DefectGanDiscriminator(label_nc=opt.label_nc,
                                           image_size=opt.image_size,
                                           input_nc=opt.input_nc,
                                           num_layers=opt.num_layers,
                                           ndf=opt.ndf,
                                           use_spectral=opt.use_spectral).to(opt.device)

    def _compute_generator_loss(self, bg_data, df_labels, df_data):
        seg = df_labels.expand_as(bg_data)

        # normal -> defect -> normal
        fake_defects, df_prob = self.netG(bg_data, seg)
        recover_normals, re_df_prob = self.netG(fake_defects, -seg)

        # defect -> normal -> defect
        fake_normals, nm_prob = self.netG(df_data, -seg)
        recover_defects, re_nm_prob = self.netG(fake_normals, seg)

        # discriminator
        fake_defects_logits = self.netD(fake_defects)
        fake_normals_logits = self.netD(fake_normals)

        recover_defects_logits = self.netD(recover_defects)
        recover_normals_logits = self.netD(recover_normals)

        real_defects_logits = self.netD(df_data)
        real_normals_logits = self.netD(bg_data)
        # TODO calculate generator loss

    def _compute_discriminator_loss(self, bg_data, df_labels, df_data):
        # generator
        seg = df_labels.expand_as(bg_data)
        with torch.no_grad():
            # normal -> defect
            fake_defects, df_prob = self.netG(bg_data, seg)
            # defect -> normal
            fake_normals, nm_prob = self.netG(df_data, -seg)

        # discriminator
        fake_defects_logits = self.netD(fake_defects.detach_())
        fake_normals_logits = self.netD(fake_normals.detach_())
        real_defects_logits = self.netD(df_data)
        real_normals_logits = self.netD(bg_data)
        # TODO calculate patchGAN loss

    def _generate_fake(self, data, labels):
        seg = labels.expand_as(data)

    def __call__(self, mode, data, labels, df_data=None):
        if mode == 'generator':
            self._compute_generator_loss(data, labels, df_data)
            return g_loss, generated
        elif mode == 'discriminator':
            self._compute_discriminator_loss(data, labels, df_data)
            return d_loss
        # elif mode == 'encode_only':
        #     z, mu, logvar = self.encode_z(real_image)
        #     return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                self._generate_fake(data, labels)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")
