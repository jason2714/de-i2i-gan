from models.networks.generator import DefectGanGenerator
from models.networks.discriminator import DefectGanDiscriminator
from models.base_model import BaseModel
import torch
import math
from torchvision.utils import make_grid
import numpy as np
import cv2


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
                                       add_noise=opt.add_noise,
                                       cycle_gan=opt.cycle_gan,
                                       skip_conn=opt.skip_conn).to(opt.device, non_blocking=True)
        self.netD = DefectGanDiscriminator(label_nc=opt.label_nc,
                                           image_size=opt.image_size,
                                           input_nc=opt.input_nc,
                                           num_layers=opt.num_layers,
                                           ndf=opt.ndf,
                                           use_spectral=opt.use_spectral).to(opt.device, non_blocking=True)

    def __call__(self, mode, data, labels, df_data=None):
        if mode == 'generator':
            self.netD.eval()
            self.netG.train()
            return self._compute_generator_loss(data, labels, df_data)
        elif mode == 'discriminator':
            self.netD.train()
            self.netG.eval()
            return self._compute_discriminator_loss(data, labels, df_data)
        # elif mode == 'encode_only':
        #     z, mu, logvar = self.encode_z(real_image)
        #     return mu, logvar
        elif mode == 'inference':
            self.netD.eval()
            self.netG.eval()
            return self._generate_fake(data, labels)
        elif mode == 'generate_grid':
            self.netD.eval()
            self.netG.eval()
            return self._generate_fake_grids(data, labels)
        else:
            raise ValueError(f"|mode {mode}| is invalid")

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

        if self.opt.cycle_gan:
            return torch.stack(list(gan_loss.values())).mean(), \
                   torch.stack(list(clf_loss.values())).mean(), \
                   torch.stack(list(rec_loss.values())).mean(), \
                   torch.zeros([], requires_grad=False), \
                   torch.zeros([], requires_grad=False)
        else:
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
            fake_defects, _ = self.netG(bg_data, seg)
            # defect -> normal
            fake_normals, _ = self.netG(df_data, -seg)

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
        return torch.stack(list(gan_loss.values())).mean(), \
               torch.stack(list(clf_loss.values())).mean()

    @torch.no_grad()
    def _generate_fake(self, data, labels):
        data, labels = data.to(self.netG.device), labels.to(self.netG.device, non_blocking=True)
        if len(labels.size()) == 2:
            seg = labels.reshape(labels.size(0), labels.size(1), 1, 1)
        elif len(labels.size()) == 4:
            seg = labels
        else:
            raise ValueError(f"|labels dim {len(labels.size())}| is invalid")
        return self.netG(data, seg)

    @torch.no_grad()
    def _generate_fake_grids(self, bg_data, labels):
        nm_images = None
        single_df_images = []
        multi_df_images = None
        init_flag = True
        for data in bg_data:
            single_df_images.append(data / 2 + 0.5)
            data = data.unsqueeze(0).to(self.netG.device)
            df_data, df_prob = self._generate_fake(data.expand(labels.size(0), -1, -1, -1), labels)
            foreground = torch.clamp((df_data - data * (1 - df_prob)) / (df_prob + 1e-8), min=-1, max=1)
            # print(torch.min(foreground), torch.max(foreground))
            df_data = (df_data / 2 + 0.5).detach().cpu()
            foreground = (foreground / 2 + 0.5).detach().cpu()
            for idx, (slice_data, slice_prob, slice_foreground) in enumerate(zip(df_data, df_prob, foreground)):
                if idx == self.opt.label_nc - 1:
                    break
                slice_prob = slice_prob.squeeze(0).detach().cpu()
                # print(torch.min(slice_prob), torch.max(slice_prob))
                heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * slice_prob), cv2.COLORMAP_JET),
                                       cv2.COLOR_BGR2RGB)
                heatmap = torch.from_numpy(heatmap.transpose(2, 0, 1)) / 255.
                # print(torch.min(heatmap), torch.max(heatmap))
                single_df_images += [slice_data, heatmap, slice_foreground]
            # nm_data, nm_prob = self.model('inference', df_data, -labels)
            data = (data / 2 + 0.5).cpu()
            if init_flag:
                # nm_images = torch.cat((data, nm_data[:self.opt.label_nc - 1]), dim=0)
                multi_df_images = torch.cat((data, df_data[self.opt.label_nc - 1:]), dim=0)
                init_flag = False
            else:
                # nm_images = torch.cat((nm_images, data, nm_data[:self.opt.label_nc - 1]), dim=0)
                multi_df_images = torch.cat((multi_df_images, data, df_data[self.opt.label_nc - 1:]), dim=0)
        single_df_images = torch.stack(single_df_images, dim=0)
        df_grid = make_grid(single_df_images, nrow=((self.opt.label_nc - 1) * 3 + 1))
        # nm_grid = make_grid(nm_images, nrow=self.opt.label_nc)
        mtp_df_grid = make_grid(multi_df_images, nrow=(self.opt.num_display_images + 1))
        return df_grid, mtp_df_grid
