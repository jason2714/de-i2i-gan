from models.networks.generator import DefectGanGenerator
from models.networks.discriminator import DefectGanDiscriminator
from models.base_model import BaseModel
import torch
import math
from torchvision.utils import make_grid
import numpy as np
import cv2
from utils.util import generate_mask

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

    def __call__(self, mode, data, labels, df_data=None, img_only=False):
        if mode in ('mae', 'mae_inference'):
            self.netD.eval()
            if mode == 'mae':
                self.netG.train()
                return self._compute_mae_loss(data, labels)
            else:
                self.netG.eval()
                with torch.no_grad():
                    return self._compute_mae_loss(data, labels)
        elif mode == 'generator':
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
            return self._generate_fake_grids(data, labels, img_only)
        elif mode == 'generate_mask_grid':
            self.netD.eval()
            self.netG.eval()
            return self._generate_repair_mask_grid(data, labels)
        else:
            raise ValueError(f"|mode {mode}| is invalid")

    def _compute_mae_loss(self, imgs, labels):
        predicted_imgs, masks = self._repair_mask(imgs, labels)

        # mae l2-loss
        rec_loss = self._cal_loss(predicted_imgs * masks, imgs * masks, 'mse') / self.opt.mask_ratio

        # # discriminator
        # fake_defects_src, fake_defects_cls = self.netD(predicted_imgs)
        return rec_loss

    def _compute_generator_loss(self, bg_data, df_labels, df_data):
        bg_data, df_labels, df_data = bg_data.to(self.netG.device, non_blocking=True), \
                                      df_labels.to(self.netG.device, non_blocking=True), \
                                      df_data.to(self.netG.device, non_blocking=True)
        seg = self._expand_seg(df_labels)

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
        seg = self._expand_seg(labels)
        return self.netG(data, seg)

    @torch.no_grad()
    def _generate_fake_grids(self, bg_data, labels, img_only=False):
        df_images = []
        for data in bg_data:
            df_images.append(data / 2 + 0.5)
            data = data.unsqueeze(0).to(self.netG.device)
            df_data, df_prob = self._generate_fake(data.expand(labels.size(0), -1, -1, -1), labels)
            if img_only:
                df_data = (df_data / 2 + 0.5).detach().cpu()
                for slice_data in df_data:
                    df_images += [slice_data]
            else:
                foreground = torch.clamp((df_data - data * (1 - df_prob)) / (df_prob + 1e-8), min=-1, max=1)
                df_data = (df_data / 2 + 0.5).detach().cpu()
                foreground = (foreground / 2 + 0.5).detach().cpu()
                df_prob = df_prob.detach().cpu()
                for idx, (slice_data, slice_prob, slice_foreground) in enumerate(zip(df_data, df_prob, foreground)):
                    slice_prob = slice_prob.squeeze(0)
                    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * slice_prob), cv2.COLORMAP_JET),
                                           cv2.COLOR_BGR2RGB)
                    heatmap = torch.from_numpy(heatmap.transpose(2, 0, 1)) / 255.
                    df_images += [slice_data, heatmap, slice_foreground]
        df_images = torch.stack(df_images, dim=0)
        nrow = 1 + (labels.size(0) if img_only else 3 * labels.size(0))
        df_grid = make_grid(df_images, nrow=nrow)
        return df_grid

    @torch.no_grad()
    def _generate_repair_mask_grid(self, imgs, labels):
        predicted_imgs, masks = self._repair_mask(imgs, labels)
        predicted_masked_imgs = predicted_imgs * (1 - masks)
        combine_imgs = imgs * masks + predicted_masked_imgs

        # generate grid
        grid_imgs = torch.stack([imgs, combine_imgs, predicted_masked_imgs, predicted_imgs], dim=1)
        grid_imgs = grid_imgs.transpose(0, 1).reshape(-1, *imgs.size()[1:])
        nrow = 4

        return make_grid(grid_imgs, nrow=nrow)

    def _repair_mask(self, imgs, labels):
        imgs = imgs.to(self.netG.device, non_blocking=True)
        seg = self._expand_seg(torch.zeros_like(labels))
        masks = generate_mask(imgs.size(), self.opt.patch_size, self.opt.mask_ratio)
        masks = masks.to(self.netG.device, non_blocking=True)
        masked_imgs = imgs * masks

        predicted_imgs, _ = self.netG(masked_imgs, seg)
        return predicted_imgs, masks

    def _expand_seg(self, labels):
        if len(labels.size()) == 2:
            seg = labels.reshape(labels.size(0), labels.size(1), 1, 1)
        elif len(labels.size()) == 4:
            seg = labels
        else:
            raise ValueError(f"|labels dim {len(labels.size())}| is invalid")
        return seg
