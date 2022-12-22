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
                                       add_noise=opt.add_noise).to(opt.device)
        self.netD = DefectGanDiscriminator(label_nc=opt.label_nc,
                                           image_size=opt.image_size,
                                           input_nc=opt.input_nc,
                                           num_layers=opt.num_layers,
                                           ndf=opt.ndf,
                                           use_spectral=opt.use_spectral).to(opt.device)

    def __call__(self, mode, data, labels, df_data=None):
        if mode == 'generator':
            return self._compute_generator_loss(data, labels, df_data)
        elif mode == 'discriminator':
            return self._compute_discriminator_loss(data, labels, df_data)
        # elif mode == 'encode_only':
        #     z, mu, logvar = self.encode_z(real_image)
        #     return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                return self._generate_fake(data, labels)
        else:
            raise ValueError("|mode| is invalid")

    def _compute_generator_loss(self, bg_data, df_labels, df_data):
        seg = df_labels.expand_as(bg_data)

        # normal -> defect -> normal
        fake_defects, df_prob = self.netG(bg_data, seg)
        recover_normals, rec_df_prob = self.netG(fake_defects, -seg)

        # defect -> normal -> defect
        fake_normals, nm_prob = self.netG(df_data, -seg)
        recover_defects, rec_nm_prob = self.netG(fake_normals, seg)

        # discriminator
        fake_defects_logits = self.netD(fake_defects)
        fake_normals_logits = self.netD(fake_normals)

        # recover_defects_logits = self.netD(recover_defects)
        # recover_normals_logits = self.netD(recover_normals)
        #
        # real_defects_logits = self.netD(df_data)
        # real_normals_logits = self.netD(bg_data)

        # gan loss
        fake_labels = torch.ones_like(fake_defects_logits, dtype=torch.float).to(self.netD.device)
        gan_loss = {'fake_defect': self._cal_gan_loss(fake_defects_logits, fake_labels),
                    'fake_normal': self._cal_gan_loss(fake_normals_logits, fake_labels)}

        # clf loss
        nm_labels = torch.ones_like(fake_normals_logits, dtype=torch.float).to(self.netD.device)
        clf_loss = {'fake_defect': self._cal_clf_loss(fake_defects_logits, df_labels),
                    'fake_normal': self._cal_clf_loss(fake_normals_logits, nm_labels)}

        # rec loss
        rec_loss = {'defect': self._cal_diff_loss(recover_defects, df_data),
                    'normal': self._cal_diff_loss(recover_normals, bg_data)}

        # sd_cyc loss
        sd_cyc_loss = {'defect': self._cal_diff_loss(df_prob, rec_df_prob),
                       'normal': self._cal_diff_loss(nm_prob, rec_nm_prob)}

        # sd_con loss
        con_labels = torch.zeros_like(df_prob, dtype=torch.float).to(self.netG.device)
        sd_con_loss = {'defect': self._cal_diff_loss(df_prob, con_labels),
                       'normal': self._cal_diff_loss(nm_prob, con_labels),
                       'rec_defect': self._cal_diff_loss(rec_df_prob, con_labels),
                       'rec_normal': self._cal_diff_loss(rec_nm_prob, con_labels)}
        return gan_loss, clf_loss, rec_loss, sd_cyc_loss, sd_con_loss

    def _compute_discriminator_loss(self, bg_data, df_labels, df_data):
        # generator
        seg = df_labels.expand_as(bg_data)
        with torch.no_grad():
            # normal -> defect
            fake_defects, df_prob = self.netG(bg_data, seg)
            # defect -> normal
            fake_normals, nm_prob = self.netG(df_data, -seg)

        fake_defects.requires_grad = True
        fake_normals.requires_grad = True

        # discriminator
        fake_defects_logits = self.netD(fake_defects.detach_())
        fake_normals_logits = self.netD(fake_normals.detach_())
        real_defects_logits = self.netD(df_data)
        real_normals_logits = self.netD(bg_data)

        # gan loss
        real_labels = torch.ones_like(fake_defects_logits, dtype=torch.float).to(self.netD.device)
        fake_labels = torch.zeros_like(real_defects_logits, dtype=torch.float).to(self.netD.device)
        gan_loss = {'fake_defect': self._cal_gan_loss(fake_defects_logits, fake_labels),
                    'fake_normal': self._cal_gan_loss(fake_normals_logits, fake_labels),
                    'real_defect': self._cal_gan_loss(real_defects_logits, real_labels),
                    'real_normal': self._cal_gan_loss(real_normals_logits, real_labels)}

        # clf loss
        nm_labels = torch.ones_like(real_normals_logits, dtype=torch.float).to(self.netD.device)
        clf_loss = {'real_defect': self._cal_clf_loss(real_defects_logits, df_labels),
                    'real_normal': self._cal_clf_loss(real_normals_logits, nm_labels)}

        return gan_loss, clf_loss

    def _generate_fake(self, data, labels):
        seg = labels.expand_as(data)
        outputs, _ = self.netG(data, seg)
        return outputs

    def _cal_clf_loss(self, logits, targets, loss_type='multilabel'):
        """Compute binary or softmax cross entropy loss."""
        if loss_type == 'multilabel':
            return binary_cross_entropy_with_logits(logits, targets)
        elif loss_type == 'multiclass':
            return cross_entropy(logits, targets)
        else:
            raise ValueError(f"loss_type: {loss_type} is invalid")

    def _cal_gan_loss(self, logits, targets, loss_type='original'):
        """Compute binary or softmax cross entropy loss."""
        if loss_type == 'original':
            return binary_cross_entropy_with_logits(logits, targets)
        else:
            raise ValueError(f"loss_type: {loss_type} is invalid")

    def _cal_diff_loss(self, logits, targets, loss_type='l1'):
        if loss_type == 'l1':
            return l1_loss(logits, targets)
        elif loss_type == 'l2':
            return mse_loss(logits, targets)
        else:
            raise ValueError(f"loss_type: {loss_type} is invalid")
