from models.networks.generator import DefectGanGenerator
from models.networks.discriminator import DefectGanDiscriminator
from models.base_model import BaseModel
import torch
import math
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, l1_loss, mse_loss


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
                                       add_noise=opt.add_noise).to(opt.device, non_blocking=True)
        self.netD = DefectGanDiscriminator(label_nc=opt.label_nc,
                                           image_size=opt.image_size,
                                           input_nc=opt.input_nc,
                                           num_layers=opt.num_layers,
                                           ndf=opt.ndf,
                                           use_spectral=opt.use_spectral).to(opt.device, non_blocking=True)

    def __call__(self, mode, data, labels, df_data=None):
        if mode == 'generator':
            return self._compute_generator_loss(data, labels, df_data)
        elif mode == 'discriminator':
            return self._compute_discriminator_loss(data, labels, df_data)
        # elif mode == 'encode_only':
        #     z, mu, logvar = self.encode_z(real_image)
        #     return mu, logvar
        elif mode == 'inference':
            return self._generate_fake(data, labels)
        else:
            raise ValueError("|mode| is invalid")

    def _compute_generator_loss(self, bg_data, df_labels, df_data):
        bg_data, df_labels, df_data = bg_data.to(self.netG.device, non_blocking=True), \
                                      df_labels.to(self.netG.device, non_blocking=True), \
                                      df_data.to(self.netG.device, non_blocking=True)
        seg = df_labels.reshape(*df_labels.size(), 1, 1)

        # normal -> defect -> normal
        fake_defects, df_prob = self.netG(bg_data, seg)
        recover_normals, rec_df_prob = self.netG(fake_defects, -seg)

        # defect -> normal -> defect
        fake_normals, nm_prob = self.netG(df_data, -seg)
        recover_defects, rec_nm_prob = self.netG(fake_normals, seg)

        # discriminator
        fake_defects_src, fake_defects_cls = self.netD(fake_defects)
        fake_normals_src, fake_normals_cls = self.netD(fake_normals)

        # recover_defects_logits = self.netD(recover_defects)
        # recover_normals_logits = self.netD(recover_normals)
        #
        # real_defects_logits = self.netD(df_data)
        # real_normals_logits = self.netD(bg_data)

        # gan loss
        fake_labels = torch.ones_like(fake_defects_src, dtype=torch.float).to(self.netD.device, non_blocking=True)
        gan_loss = {'fake_defect': self._cal_loss(fake_defects_src, fake_labels, 'bce'),
                    'fake_normal': self._cal_loss(fake_normals_src, fake_labels, 'bce')}

        # clf loss
        nm_labels = torch.zeros_like(fake_normals_cls, dtype=torch.float).to(self.netD.device, non_blocking=True)
        nm_labels[:, 0] = 1
        clf_loss = {'fake_defect': self._cal_loss(fake_defects_cls, df_labels, 'bce'),
                    'fake_normal': self._cal_loss(fake_normals_cls, nm_labels, 'bce')}

        # rec loss
        rec_loss = {'defect': self._cal_loss(recover_defects, df_data, 'l1'),
                    'normal': self._cal_loss(recover_normals, bg_data, 'l1')}

        # sd_cyc loss
        sd_cyc_loss = {'defect': self._cal_loss(df_prob, rec_df_prob, 'l1'),
                       'normal': self._cal_loss(nm_prob, rec_nm_prob, 'l1')}

        # sd_con loss
        con_labels = torch.zeros_like(df_prob, dtype=torch.float).to(self.netG.device, non_blocking=True)
        sd_con_loss = {'defect': self._cal_loss(df_prob, con_labels, 'l1'),
                       'normal': self._cal_loss(nm_prob, con_labels, 'l1'),
                       'rec_defect': self._cal_loss(rec_df_prob, con_labels, 'l1'),
                       'rec_normal': self._cal_loss(rec_nm_prob, con_labels, 'l1')}
        return torch.stack(list(gan_loss.values())).mean(), \
               torch.stack(list(clf_loss.values())).mean(), \
               torch.stack(list(rec_loss.values())).mean(), \
               torch.stack(list(sd_cyc_loss.values())).mean(), \
               torch.stack(list(sd_con_loss.values())).mean()

    def _compute_discriminator_loss(self, bg_data, df_labels, df_data):
        bg_data, df_labels, df_data = bg_data.to(self.netG.device, non_blocking=True), \
                                      df_labels.to(self.netG.device, non_blocking=True), \
                                      df_data.to(self.netG.device, non_blocking=True)
        # generator
        seg = df_labels.reshape(*df_labels.size(), 1, 1)
        with torch.no_grad():
            # normal -> defect
            fake_defects, df_prob = self.netG(bg_data, seg)
            # defect -> normal
            fake_normals, nm_prob = self.netG(df_data, -seg)

        fake_defects.requires_grad = True
        fake_normals.requires_grad = True

        # discriminator
        fake_defects_src, _ = self.netD(fake_defects.detach_())
        fake_normals_src, _ = self.netD(fake_normals.detach_())
        real_defects_src, real_defects_cls = self.netD(df_data)
        real_normals_src, real_normals_cls = self.netD(bg_data)

        # gan loss
        real_labels = torch.ones_like(real_defects_src, dtype=torch.float).to(self.netD.device, non_blocking=True)
        fake_labels = torch.zeros_like(fake_defects_src, dtype=torch.float).to(self.netD.device, non_blocking=True)
        gan_loss = {'fake_defect': self._cal_loss(fake_defects_src, fake_labels, 'bce'),
                    'fake_normal': self._cal_loss(fake_normals_src, fake_labels, 'bce'),
                    'real_defect': self._cal_loss(real_defects_src, real_labels, 'bce'),
                    'real_normal': self._cal_loss(real_normals_src, real_labels, 'bce')}

        # clf loss
        # problem
        nm_labels = torch.zeros_like(real_normals_cls, dtype=torch.float).to(self.netD.device, non_blocking=True)
        nm_labels[:, 0] = 1
        clf_loss = {'real_defect': self._cal_loss(real_defects_cls, df_labels, 'bce'),
                    'real_normal': self._cal_loss(real_normals_cls, nm_labels, 'bce')}
        # exit()
        return torch.stack(list(gan_loss.values())).mean(), \
               torch.stack(list(clf_loss.values())).mean()

    @torch.no_grad()
    def _generate_fake(self, data, labels):
        data, labels = data.to(self.netG.device), labels.to(self.netG.device, non_blocking=True)
        seg = labels.reshape(*labels.size(), 1, 1)
        outputs, _ = self.netG(data, seg)
        return outputs

    def _cal_loss(self, logits, targets, loss_type):
        """Compute loss
            input type for cce and bce is unnormalized logits"""
        if loss_type in ('bce', 'bce_logits'):
            # print(logits.size(), targets.size())
            # print(logits[:4], targets[:4])
            # print(binary_cross_entropy_with_logits(logits, targets))
            return binary_cross_entropy_with_logits(logits, targets)
        elif loss_type in ('cce', 'cce_logits'):
            return cross_entropy(logits, targets)
        elif loss_type == 'l1':
            return l1_loss(logits, targets)
        elif loss_type in ('l2', 'mse'):
            return mse_loss(logits, targets)
        else:
            raise ValueError(f"loss_type: {loss_type} is invalid")
