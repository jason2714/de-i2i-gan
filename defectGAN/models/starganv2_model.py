import random

from models.networks.generator import StarGANv2Generator
from models.networks.discriminator import StarGANv2Discriminator
from models.networks.extractor import StyleExtractor
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
        self.netG = StarGANv2Generator(opt).to(opt.device, non_blocking=True)
        self.netD = StarGANv2Discriminator(opt).to(opt.device, non_blocking=True)
        if self.opt.is_train or hasattr(opt, 'clf_loss_type'):
            assert opt.clf_loss_type is not None, 'clf_loss_type should be initialized in dataset'
            self.clf_loss_type = opt.clf_loss_type

        # learnable mask token
        self.mask_token = MaskToken(opt).to(opt.device, non_blocking=True)

        # style embedding
        if opt.style_norm_block_type == 'sean':
            if self.opt.sean_alpha is not None:
                self.netG.set_sean_alpha(self.opt.sean_alpha)
            if opt.sean_alpha != 0:
                # load embeddings
                if not (opt.phase == 'test' and opt.use_running_stats):
                    assert opt.embed_path is not None, 'embed_path should be initialized ' \
                                                   'if style_norm_block_type is sean and sean_alpha is not 0'
                    self.embeddings = torch.load(opt.embed_path)
                    for label, embeds in self.embeddings.items():
                        self.embeddings[label] = [embed.to(opt.device, non_blocking=True) for embed in embeds]
        elif opt.style_norm_block_type == 'adain':
            self.netE = StyleExtractor(opt).to(opt.device, non_blocking=True)

    def __call__(self, mode, data, labels, df_data=None, img_only=False):
        data, labels = data.to(self.opt.device, non_blocking=True), labels.to(self.opt.device, non_blocking=True)
        if df_data is not None:
            df_data = df_data.to(self.opt.device, non_blocking=True)
        # for mae
        with autocast(device_type='cuda'):
            if mode.startswith('mae'):
                # TODO input df_data
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
                            rec_loss, gan_loss, _ = self._compute_mae_inference_loss(data, labels)
                            _, clf_loss = self._compute_mae_discriminator_loss(data, labels)
                            return rec_loss, gan_loss, clf_loss
                        else:
                            return self._compute_mae_inference_loss(data, labels)
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

        if self.opt.style_norm_block_type == 'sean' and self.opt.style_distill:
            distill_loss = {}
            self.netG.enable_sean_distill_loss(True)
        predicted_imgs, masks = self._repair_mask(imgs, labels)
        if self.opt.style_norm_block_type == 'sean' and self.opt.style_distill:
            distill_loss = self.netG.get_sean_distill_loss()
            self.netG.enable_sean_distill_loss(False)

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
            if self.opt.style_norm_block_type == 'sean' and self.opt.style_distill:
                return rec_loss, gan_loss, clf_loss, distill_loss['latent'], distill_loss['embed']
            return rec_loss, gan_loss, clf_loss

    @torch.no_grad()
    def _compute_mae_inference_loss(self, imgs, labels):

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

        nm_labels, nm_label_feat, df_labels, df_label_feat = self._get_label_and_style_feat(bg_data, df_labels, df_data)

        if self.opt.style_norm_block_type == 'sean':
            if self.opt.style_distill:
                distill_loss = {}
                self.netG.enable_sean_distill_loss(True)
            if self.opt.use_running_stats:
                self.netG.track_running_stats = True

        # normal -> defect -> normal
        fake_defects, df_prob = self.netG(bg_data, df_labels, df_label_feat)
        recover_normals, rec_df_prob = self.netG(fake_defects, nm_labels, nm_label_feat)

        # defect -> normal -> defect
        fake_normals, nm_prob = self.netG(df_data, nm_labels, nm_label_feat)
        recover_defects, rec_nm_prob = self.netG(fake_normals, df_labels, df_label_feat)

        if self.opt.style_norm_block_type == 'sean':
            if self.opt.style_distill:
                distill_loss = self.netG.get_sean_distill_loss()
                self.netG.enable_sean_distill_loss(False)
            if self.opt.use_running_stats:
                self.netG.track_running_stats = False

        # discriminator
        fake_defects_src, fake_defects_cls = self.netD(fake_defects)
        fake_normals_src, fake_normals_cls = self.netD(fake_normals)

        # gan loss
        fake_labels = torch.ones_like(fake_defects_src, dtype=torch.float).to(self.netD.device, non_blocking=True)
        gan_loss = {'fake_defect': self._cal_loss(fake_defects_src, fake_labels, 'bce'),
                    'fake_normal': self._cal_loss(fake_normals_src, fake_labels, 'bce')}

        # clf loss
        clf_loss = {
            'fake_defect': self._cal_loss(fake_defects_cls, df_labels.view_as(fake_defects_cls), self.clf_loss_type),
            'fake_normal': self._cal_loss(fake_normals_cls, nm_labels.view_as(fake_normals_cls), self.clf_loss_type)}
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
            if self.opt.style_norm_block_type == 'sean' and self.opt.style_distill:
                return torch.stack(list(gan_loss.values())).mean(), \
                    torch.stack(list(clf_loss.values())).mean(), \
                    torch.stack(list(rec_loss.values())).mean(), \
                    torch.stack(list(sd_cyc_loss.values())).mean(), \
                    torch.stack(list(sd_con_loss.values())).mean(), \
                    distill_loss['latent'], distill_loss['embed']
            return torch.stack(list(gan_loss.values())).mean(), \
                torch.stack(list(clf_loss.values())).mean(), \
                torch.stack(list(rec_loss.values())).mean(), \
                torch.stack(list(sd_cyc_loss.values())).mean(), \
                torch.stack(list(sd_con_loss.values())).mean()

    def _compute_discriminator_loss(self, bg_data, df_labels, df_data):

        nm_labels, nm_label_feat, df_labels, df_label_feat = self._get_label_and_style_feat(bg_data, df_labels, df_data)

        # generator
        with torch.no_grad():
            # normal -> defect
            fake_defects, _ = self.netG(bg_data, df_labels, df_label_feat)
            # defect -> normal
            fake_normals, _ = self.netG(df_data, nm_labels, nm_label_feat)

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
        clf_loss = {
            'real_defect': self._cal_loss(real_defects_cls, df_labels.view_as(real_defects_cls), self.clf_loss_type),
            'real_normal': self._cal_loss(real_normals_cls, nm_labels.view_as(real_normals_cls), self.clf_loss_type)}
        # exit()
        return torch.stack(list(gan_loss.values())).mean(), \
            torch.stack(list(clf_loss.values())).mean()

    def _compute_clf_loss(self, imgs, labels):

        _, df_logits = self.netD(imgs)

        clf_loss = self._cal_loss(df_logits, labels, self.clf_loss_type)

        return df_logits, clf_loss

    @torch.no_grad()
    def _generate_fake(self, data, labels):
        if self.opt.style_norm_block_type == 'sean':
            label_feat = self._get_style_embeds(labels)
            return self.netG(data, labels, label_feat)
        elif self.opt.style_norm_block_type == 'spade':
            label_feat = self._expand_seg(labels)
            return self.netG(data, label_feat)
        elif self.opt.style_norm_block_type == 'adain':
            style_feat = self.netE(data, labels)
            return self.netG(data, labels, style_feat)
        else:
            raise ValueError(f"|style_norm_block_type {self.opt.style_norm_block_type}| is invalid")

    @torch.no_grad()
    def _generate_fake_grids(self, src_data, labels):
        tgt_images = []
        for data in src_data:
            data = data.unsqueeze(0)
            tgt_images.append(data.add(1).div(2))
            transform_images = self._generate_fake(data.repeat(labels.size(0), 1, 1, 1), labels)
            transform_images.add_(1).div_(2)
            tgt_images.append(transform_images)
        tgt_images = torch.cat([*tgt_images], dim=0)
        nrow = 1 + labels.size(0)
        return make_grid(tgt_images, nrow=nrow)

    @torch.no_grad()
    def _generate_repair_mask_grid(self, imgs, labels):
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
        # masks = generate_mask(imgs.size(), self.opt.patch_size, self.opt.mask_ratio)
        masks = generate_shifted_mask(imgs.size(), self.opt.patch_size, self.opt.mask_ratio)
        masks = masks.to(self.opt.device, non_blocking=True)
        masked_imgs = imgs * masks

        # mean of unmasked
        # img_mean = masked_imgs.mean(dim=(2, 3)) / self.opt.mask_ratio
        # img_mean = img_mean.reshape(*img_mean.size()[:2], 1, 1)
        masked_imgs = self.mask_token(masked_imgs, masks)

        if self.opt.style_norm_block_type == 'sean':
            style_feat = self._get_style_embeds(labels)
            predicted_imgs = self.netG(masked_imgs, labels, style_feat)
        elif self.opt.style_norm_block_type == 'spade':
            # seg = self._expand_seg(torch.zeros_like(labels))
            seg = self._expand_seg(labels)
            predicted_imgs = self.netG(masked_imgs, seg)
        elif self.opt.style_norm_block_type == 'adain':
            style_feat = self.netE(imgs, labels)
            predicted_imgs = self.netG(masked_imgs, labels, style_feat)
        else:
            raise ValueError(f"|style_norm_block_type {self.opt.style_norm_block_type}| is invalid")

        return predicted_imgs, masks

    def _expand_seg(self, labels) :
        if len(labels.size()) == 2:
            seg = labels.reshape(labels.size(0), labels.size(1), 1, 1)
        elif len(labels.size()) == 4:
            seg = labels
        else:
            raise ValueError(f"|labels dim {len(labels.size())}| is invalid")
        return seg

    def _get_style_embeds(self, labels):
        if self.opt.sean_alpha == 0:
            return None
        elif self.opt.use_running_stats and self.netG.inference_running_stats:
            return torch.randn(labels.size(0), self.opt.hidden_nc).to(self.netG.device)
        else:
            embed_list = []
            num_embeds = random.randint(1, self.opt.num_embeds)
            # num_embeds = self.opt.num_embeds
            for label in labels:
                tuple_label = tuple(label.int().tolist())
                if not self.embeddings[tuple_label]:
                    mean_embed = torch.zeros(num_embeds, self.opt.embed_nc).to(self.netG.device)
                else:
                    embeds = random.choices(self.embeddings[tuple_label], k=num_embeds)
                    mean_embed = torch.stack(embeds)
                embed_list.append(mean_embed)
            return torch.stack(embed_list)

    def _get_label_and_style_feat(self, src_data, src_labels, tgt_data, tgt_labels):
        src_label_feat = tgt_label_feat = None
        if self.opt.style_norm_block_type == 'sean':
            src_label_feat = self._get_style_embeds(src_labels)
            tgt_label_feat = self._get_style_embeds(tgt_labels)
        elif self.opt.style_norm_block_type == 'spade':
            src_labels = self._expand_seg(src_labels)
            tgt_labels = self._expand_seg(tgt_labels)
        elif self.opt.style_norm_block_type == 'adain':
            src_label_feat = self.netE(src_data, src_labels)
            tgt_label_feat = self.netE(tgt_data, tgt_labels)
        else:
            raise ValueError(f"|style_norm_block_type {self.opt.style_norm_block_type}| is invalid")
        return src_labels, src_label_feat, tgt_labels, tgt_label_feat
