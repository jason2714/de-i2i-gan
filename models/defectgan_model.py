import random

from models.networks.generator import DefectGanGenerator
from models.networks.discriminator import DefectGanDiscriminator
from models.base_model import BaseModel
import torch
import math
from torchvision.utils import make_grid
import numpy as np
import cv2
from utils.util import generate_mask, generate_shifted_mask
from torch import autocast
import torchvision.transforms as transforms
from models.networks.architecture import MaskToken


class DefectGanModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        image_size = opt.image_size
        assert image_size & (image_size - 1) == 0, 'Image size must be a power of 2'
        self.netG = DefectGanGenerator(opt).to(opt.device, non_blocking=True)
        self.netD = DefectGanDiscriminator(opt).to(opt.device, non_blocking=True)
        if self.opt.is_train or hasattr(opt, 'clf_loss_type'):
            assert opt.clf_loss_type is not None, 'clf_loss_type should be initialized in dataset'
            self.clf_loss_type = opt.clf_loss_type

        # learnable mask token
        self.mask_token = MaskToken(opt).to(opt.device, non_blocking=True)

        # style embedding
        if opt.style_norm_block_type == 'sean' and not opt.use_latent_only:
            assert opt.embed_path is not None, 'embed_path should be initialized if style_norm_block_type is sean'
            self.embeddings = torch.load(opt.embed_path)
            for label, embeds in self.embeddings.items():
                self.embeddings[label] = [embed.to(opt.device, non_blocking=True) for embed in embeds]

    def __call__(self, mode, data, labels, df_data=None, img_only=False):
        # for mae
        if mode.startswith('mae'):
            with autocast(device_type='cuda'):
                if mode == 'mae_generator':
                    self.netD.eval()
                    self.netG.train()
                    return self._compute_mae_generator_loss(data, labels)
                elif mode == 'mae_discriminator':
                    self.netD.train()
                    self.netG.eval()
                    return self._compute_mae_discriminator_loss(data, labels)
                elif mode == 'mae_inference':
                    self.netD.eval()
                    self.netG.eval()
                    with torch.no_grad():
                        if self.opt.split_training:
                            rec_loss, gan_loss, _ = self._compute_mae_generator_loss(data, labels)
                            _, clf_loss = self._compute_mae_discriminator_loss(data, labels)
                            return rec_loss, gan_loss, clf_loss
                        else:
                            return self._compute_mae_generator_loss(data, labels)
                elif mode == 'mae_generate_grid':
                    self.netD.eval()
                    self.netG.eval()
                    return self._generate_repair_mask_grid(data, labels)
                else:
                    raise ValueError(f"|mode {mode}| is invalid")
        # for defectgan
        else:
            if mode == 'generator':
                self.netD.eval()
                self.netG.train()
                return self._compute_generator_loss(data, labels, df_data)
            elif mode == 'discriminator':
                self.netD.train()
                self.netG.eval()
                return self._compute_discriminator_loss(data, labels, df_data)
            elif mode == 'inference':
                self.netD.eval()
                self.netG.eval()
                return self._generate_fake(data, labels)
            elif mode == 'generate_grid':
                self.netD.eval()
                self.netG.eval()
                return self._generate_fake_grids(data, labels, img_only)
            elif mode == 'inference_classifier':
                self.netD.eval()
                self.netG.eval()
                return self._compute_clf_loss(data, labels)
            else:
                raise ValueError(f"|mode {mode}| is invalid")

    def _compute_mae_generator_loss(self, imgs, labels):
        imgs, labels = imgs.to(self.netG.device, non_blocking=True), labels.to(self.netG.device, non_blocking=True)

        predicted_imgs, masks = self._repair_mask(imgs, labels)

        # mae l1-loss
        rec_loss = self._cal_loss(predicted_imgs, imgs, 'l1')

        if self.opt.split_training:
            return rec_loss, torch.zeros([], requires_grad=False), torch.zeros([], requires_grad=False)
        else:
            # discriminator
            fake_src, fake_cls = self.netD(predicted_imgs)
            fake_labels = torch.ones_like(fake_src, dtype=torch.float).to(self.netD.device, non_blocking=True)
            gan_loss = self._cal_loss(fake_src, fake_labels, 'bce')
            clf_loss = self._cal_loss(fake_cls, labels, self.clf_loss_type)
            return rec_loss, gan_loss, clf_loss

    def _compute_mae_discriminator_loss(self, imgs, labels):
        imgs, labels = imgs.to(self.netG.device, non_blocking=True), labels.to(self.netG.device, non_blocking=True)

        # real
        real_src, real_cls = self.netD(imgs)
        clf_loss = self._cal_loss(real_cls, labels, self.clf_loss_type)

        # return torch.zeros([], requires_grad=False), clf_loss

        if self.opt.split_training:
            return torch.zeros([], requires_grad=False), clf_loss
        else:
            # fake
            with torch.no_grad():
                predicted_imgs, masks = self._repair_mask(imgs, labels)
            predicted_imgs.requires_grad = True
            fake_src, _ = self.netD(predicted_imgs.detach_())
            fake_labels = torch.zeros_like(fake_src, dtype=torch.float).to(self.netD.device, non_blocking=True)
            real_labels = torch.ones_like(real_src, dtype=torch.float).to(self.netD.device, non_blocking=True)

            # gan loss
            gan_loss = {'fake': self._cal_loss(fake_src, fake_labels, 'bce'),
                        'real': self._cal_loss(real_src, real_labels, 'bce')}
            return torch.stack(list(gan_loss.values())).mean(), clf_loss

    def _compute_generator_loss(self, bg_data, df_labels, df_data):
        bg_data, df_labels, df_data = bg_data.to(self.netG.device, non_blocking=True), \
                                      df_labels.to(self.netG.device, non_blocking=True), \
                                      df_data.to(self.netG.device, non_blocking=True)

        nm_labels = torch.zeros_like(df_labels, dtype=torch.float).to(self.netD.device, non_blocking=True)
        nm_labels[:, 0] = 1
        if self.opt.style_norm_block_type == 'sean':
            nm_label_feat = self._get_style_embeds(nm_labels)
            df_label_feat = self._get_style_embeds(df_labels)
            # normal -> defect -> normal
            fake_defects, df_prob = self.netG(bg_data, df_labels, df_label_feat)
            recover_normals, rec_df_prob = self.netG(fake_defects, nm_labels, nm_label_feat)

            # defect -> normal -> defect
            fake_normals, nm_prob = self.netG(df_data, nm_labels, nm_label_feat)
            recover_defects, rec_nm_prob = self.netG(fake_normals, df_labels, df_label_feat)
        elif self.opt.style_norm_block_type == 'spade':
            nm_label_feat = self._expand_seg(nm_labels)
            df_label_feat = self._expand_seg(df_labels)
            # normal -> defect -> normal
            fake_defects, df_prob = self.netG(bg_data, df_label_feat)
            # recover_normals, rec_df_prob = self.netG(fake_defects, -df_label_feat)
            recover_normals, rec_df_prob = self.netG(fake_defects, nm_label_feat)

            # defect -> normal -> defect
            fake_normals, nm_prob = self.netG(df_data, nm_label_feat)
            # fake_normals, nm_prob = self.netG(df_data, -df_label_feat)
            recover_defects, rec_nm_prob = self.netG(fake_normals, df_label_feat)
        else:
            raise ValueError(f"|style_norm_block_type {self.opt.style_norm_block_type}| is invalid")

        # discriminator
        fake_defects_src, fake_defects_cls = self.netD(fake_defects)
        fake_normals_src, fake_normals_cls = self.netD(fake_normals)

        # gan loss
        fake_labels = torch.ones_like(fake_defects_src, dtype=torch.float).to(self.netD.device, non_blocking=True)
        gan_loss = {'fake_defect': self._cal_loss(fake_defects_src, fake_labels, 'bce'),
                    'fake_normal': self._cal_loss(fake_normals_src, fake_labels, 'bce')}

        # clf loss
        clf_loss = {'fake_defect': self._cal_loss(fake_defects_cls, df_labels, self.clf_loss_type),
                    'fake_normal': self._cal_loss(fake_normals_cls, nm_labels, self.clf_loss_type)}
        # print(clf_loss)

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
        nm_labels = torch.zeros_like(df_labels, dtype=torch.float).to(self.netD.device, non_blocking=True)
        nm_labels[:, 0] = 1
        if self.opt.style_norm_block_type == 'sean':
            nm_label_feat = self._get_style_embeds(nm_labels)
            df_label_feat = self._get_style_embeds(df_labels)
            # generator
            with torch.no_grad():
                # normal -> defect
                fake_defects, _ = self.netG(bg_data, df_labels, df_label_feat)
                # defect -> normal
                fake_normals, _ = self.netG(df_data, nm_labels, nm_label_feat)
        elif self.opt.style_norm_block_type == 'spade':
            nm_label_feat = self._expand_seg(nm_labels)
            df_label_feat = self._expand_seg(df_labels)
            # generator
            with torch.no_grad():
                # normal -> defect
                fake_defects, _ = self.netG(bg_data, df_label_feat)
                # defect -> normal
                # fake_normals, _ = self.netG(df_data, -df_label_feat)
                fake_normals, _ = self.netG(df_data, nm_label_feat)
        else:
            raise ValueError(f"|style_norm_block_type {self.opt.style_norm_block_type}| is invalid")


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
        clf_loss = {'real_defect': self._cal_loss(real_defects_cls, df_labels, self.clf_loss_type),
                    'real_normal': self._cal_loss(real_normals_cls, nm_labels, self.clf_loss_type)}
        # exit()
        return torch.stack(list(gan_loss.values())).mean(), \
               torch.stack(list(clf_loss.values())).mean()

    def _compute_clf_loss(self, imgs, labels):
        imgs, labels = imgs.to(self.netG.device, non_blocking=True), labels.to(self.netG.device, non_blocking=True)

        _, df_logits = self.netD(imgs)

        clf_loss = self._cal_loss(df_logits, labels, self.clf_loss_type)

        return df_logits, clf_loss

    @torch.no_grad()
    def _generate_fake(self, data, labels):
        data, labels = data.to(self.netG.device), labels.to(self.netG.device, non_blocking=True)
        if self.opt.style_norm_block_type == 'sean':
            label_feat = self._get_style_embeds(labels)
            return self.netG(data, labels, label_feat)
        elif self.opt.style_norm_block_type == 'spade':
            label_feat = self._expand_seg(labels)
            return self.netG(data, label_feat)
        else:
            raise ValueError(f"|style_norm_block_type {self.opt.style_norm_block_type}| is invalid")

    @torch.no_grad()
    def _generate_fake_grids(self, bg_data, labels, img_only=False):
        df_images = []
        bg_data = bg_data.to(self.netG.device)
        for data in bg_data:
            data = data.unsqueeze(0)
            df_images.append(data.add(1).div(2))
            df_data, df_prob = self._generate_fake(data.repeat(labels.size(0), 1, 1, 1), labels)
            if img_only:
                df_data.add_(1).div_(2)
                df_images.append(df_data)
            else:
                if self.opt.cycle_gan:
                    foreground = df_data
                else:
                    foreground = df_data.sub(data.mul(1 - df_prob)).div(df_prob.add(1e-8)).clamp_(-1, 1)
                df_data.add_(1).div_(2)
                foreground.add_(1).div_(2)
                new_df_prob = df_prob.repeat(1, 3, 1, 1)
                for idx, slice_prob in enumerate(df_prob.cpu()):
                    slice_prob = slice_prob.squeeze(0)
                    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * slice_prob), cv2.COLORMAP_JET),
                                           cv2.COLOR_BGR2RGB)
                    new_df_prob[idx, :] = torch.from_numpy(heatmap.transpose(2, 0, 1)).to(self.netG.device)
                new_df_prob.div_(255.)
                df_images.append(torch.stack([df_data, new_df_prob, foreground], dim=1).flatten(0, 1))
        df_images = torch.cat([*df_images], dim=0)
        nrow = 1 + (labels.size(0) if img_only else 3 * labels.size(0))
        df_grid = make_grid(df_images, nrow=nrow)
        return df_grid

    @torch.no_grad()
    def _generate_repair_mask_grid(self, imgs, labels):
        imgs, labels = imgs.to(self.netG.device, non_blocking=True), labels.to(self.netG.device, non_blocking=True)
        predicted_imgs, masks = self._repair_mask(imgs, labels)
        masked_imgs = imgs * masks
        predicted_masked_imgs = predicted_imgs * (1 - masks)
        combine_imgs = masked_imgs + predicted_masked_imgs

        # generate grid
        grid_imgs = torch.stack([imgs, combine_imgs, masked_imgs, predicted_imgs, predicted_masked_imgs], dim=0)
        grid_imgs = grid_imgs.transpose(0, 1).reshape(-1, *imgs.size()[1:])
        grid_imgs.add_(1).div_(2)
        nrow = 5

        return make_grid(grid_imgs, nrow=nrow)

    def _repair_mask(self, imgs, labels):
        # generate and apply mask
        masks = generate_shifted_mask(imgs.size(), self.opt.patch_size, self.opt.mask_ratio)
        masks = masks.to(self.opt.device, non_blocking=True)
        masked_imgs = imgs * masks

        # mean of unmasked
        # img_mean = masked_imgs.mean(dim=(2, 3)) / self.opt.mask_ratio
        # img_mean = img_mean.reshape(*img_mean.size()[:2], 1, 1)
        masked_imgs = self.mask_token(masked_imgs, masks)

        if self.opt.style_norm_block_type == 'sean':
            style_feat = self._get_style_embeds(labels)
            predicted_imgs, _ = self.netG(masked_imgs, labels, style_feat)
        elif self.opt.style_norm_block_type == 'spade':
            # seg = self._expand_seg(torch.zeros_like(labels))
            seg = self._expand_seg(labels)
            predicted_imgs, _ = self.netG(masked_imgs, seg)
        else:
            raise ValueError(f"|style_norm_block_type {self.opt.style_norm_block_type}| is invalid")

        return predicted_imgs, masks

    def _expand_seg(self, labels):
        if len(labels.size()) == 2:
            seg = labels.reshape(labels.size(0), labels.size(1), 1, 1)
        elif len(labels.size()) == 4:
            seg = labels
        else:
            raise ValueError(f"|labels dim {len(labels.size())}| is invalid")
        return seg

    def _get_style_embeds(self, labels):
        if self.opt.use_latent_only:
            return None
        embed_list = []
        num_embeds = 1
        for label in labels:
            tuple_label = tuple(label.int().tolist())
            if not self.embeddings[tuple_label]:
                mean_embed = torch.zeros(num_embeds, self.opt.embed_nc).to(self.netG.device)
            else:
                embeds = random.choices(self.embeddings[tuple_label], k=num_embeds)
                # mean_embed = torch.stack(embeds).mean(dim=0)
                mean_embed = torch.stack(embeds)
            embed_list.append(mean_embed)
        return torch.stack(embed_list)

