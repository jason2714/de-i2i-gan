from metrics.fid_score import calculate_fid
from metrics.inception import InceptionV3
from metrics.lpips_score import calculate_lpips_from_model
from trainers.base_trainer import BaseTrainer
import torch
from collections import defaultdict
from tqdm import tqdm
import math
from torch.utils.tensorboard import SummaryWriter
from metrics.fid_score import calculate_fid_from_model

from trainers.base_trainer import BaseTrainer
import numpy as np
import cv2
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from metrics.defectgan_metrics import calculate_metrics_from_model


class DefectGanTrainer(BaseTrainer):
    def __init__(self, opt):
        super().__init__(opt)
        assert len(opt.loss_weight) == 5, f'length of loss weights must be 5, not {len(opt.loss_weight)}'
        self.loss_weights = {'clf_d': opt.loss_weight[0],
                             'clf_g': opt.loss_weight[1],
                             'rec': opt.loss_weight[2],
                             'sd_cyc': opt.loss_weight[3],
                             'sd_con': opt.loss_weight[4]}
        self.loss_types = ['gan', 'clf', 'aux']
        if opt.style_norm_block_type == 'sean' and opt.style_distill:
            self.loss_types.append('distill')
        self._init_losses()
        if opt.phase == 'val':
            self.metrics = {'fid': None,
                            'is': None,
                            'lpips': None}
            self.metric_models = dict()
            print(f'Initialize Inception model for calculating FID score with dims {opt.dims}')
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[opt.dims]
            self.metric_models['inception'] = InceptionV3([block_idx]).to(opt.device, non_blocking=True)
            self.metric_models['inception'].eval()

            print(f'Initialize LPIPS model for calculating LPIPS score')
            self.metric_models['lpips'] = LearnedPerceptualImagePatchSimilarity().to(device=opt.device)

    def _init_lr(self, opt):
        assert len(opt.lr) in (1, 2), f'length of lr must be 1 or 2, not {len(opt.lr)}'
        self.lr = {'D': opt.lr[0],
                   'G': opt.lr[1]} if len(opt.lr) == 2 else opt.lr[0]

    def _init_losses(self):
        self.losses = {loss_type: defaultdict(list)
                       for loss_type in self.loss_types}

    def _write_tf_log(self, writer, epoch, val_loaders):
        # for losses
        for loss_type in self.loss_types:
            writer.add_scalars(f'Losses/{loss_type}', {key: sum(value) / (len(value) + 1e-12)
                                                       for key, value in self.losses[loss_type].items()}, epoch)
        writer.add_scalars(f'Lr', {f'net_{model_name}': self.schedulers[model_name].get_last_lr()[0]
                                   for model_name in self.schedulers.keys()}, epoch)
        # for discriminator outputs
        # writer.add_scalars(f'D(x)', {key: sum(value) / len(value)
        #                              for key, value in self.losses.items()
        #                              if key.startswith('gan_')}, epoch)
        # for generated image
        if epoch % self.opt.save_img_freq == 0:
            bg_data, _, _ = next(val_loaders['background'])
            _, df_labels, _ = next(iter(val_loaders['defects']))
            labels = torch.eye(self.opt.label_nc)[1:]
            df_grid = self.model('generate_grid', bg_data, labels)
            mtp_df_grid = self.model('generate_grid', bg_data, df_labels, img_only=True)
            writer.add_image('Images/Single Defect', df_grid, epoch)
            writer.add_image('Images/Multiple Defects', mtp_df_grid, epoch)

    def train(self, train_loaders, val_loaders=None):
        """
        epoch start with 1, end with num_epochs
        """
        writer = SummaryWriter(self.opt.log_dir / self.opt.name)
        for epoch in range(self.first_epoch, self.opt.num_epochs + 1):
            self._init_losses()
            self._train_epoch(train_loaders, epoch)
            self._write_tf_log(writer, epoch, val_loaders)
            if epoch % self.opt.save_ckpt_freq == 0:
                self.model.save(epoch)
                if self.opt.phase == 'val':
                    self._val_epoch(val_loaders, epoch, writer)
            self._update_per_epoch(epoch)
        writer.close()

    def _train_epoch(self, data_loaders, epoch):
        # lrs = ', '.join([f'lr_{model_name}: {self.schedulers[model_name].get_last_lr()[0]:.5f}'
        #                  for model_name in self.schedulers.keys()])
        pbar = tqdm(data_loaders['defects'], colour='MAGENTA')
        # BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
        for df_data, df_labels, _ in pbar:
            self.iters += 1
            pbar.set_description(f'Epoch: [{epoch}/{self.opt.num_epochs}], '
                                 f'Iter: [{self.iters}/{self.opt.num_iters}], lw_con: {self.loss_weights["sd_con"]}')

            # get bg data and truncate them to the same as batch_size of defect data
            bg_data, bg_labels, _ = next(data_loaders['background'])
            # if bg_data.size(0) < df_data.size(0):
            #     bg_data, bg_labels, _ = next(data_loaders['background'])
            bg_data, bg_labels = bg_data[:df_data.size(0)], bg_labels[:df_data.size(0)]

            self._train_discriminator_once(bg_data, df_labels, df_data)
            if self.iters % self.opt.num_critics == 0:
                self._train_generator_once(bg_data, df_labels, df_data)

            if self.iters % self.opt.save_latest_freq == 0:
                self.model.save('latest')
                np.savetxt(self.iter_record_path, (epoch, self.iters), fmt='%i', delimiter=',')
            pbar.set_postfix(gan_D=f'{sum(self.losses["gan"]["D"]) / (len(self.losses["gan"]["D"]) + 1e-12):.4f}',
                             gan_G=f'{sum(self.losses["gan"]["G"]) / (len(self.losses["gan"]["G"]) + 1e-12):.4f}',
                             clf_D=f'{sum(self.losses["clf"]["D"]) / (len(self.losses["clf"]["D"]) + 1e-12):.4f}',
                             clf_G=f'{sum(self.losses["clf"]["G"]) / (len(self.losses["clf"]["G"]) + 1e-12):.4f}',
                             rec=f'{sum(self.losses["aux"]["rec"]) / (len(self.losses["aux"]["rec"]) + 1e-12):.4f}',
                             sd_cyc=f'{sum(self.losses["aux"]["cyc"]) / (len(self.losses["aux"]["cyc"]) + 1e-12):.4f}',
                             sd_con=f'{sum(self.losses["aux"]["con"]) / (len(self.losses["aux"]["con"]) + 1e-12):.4f}')
            # ,
            # dis_grad=f'{max(self.losses["dis_grad"]):.4f}')

    @torch.no_grad()
    def _val_epoch(self, data_loader, epoch, writer):
        self.metrics = calculate_metrics_from_model(self.opt, self.model,
                                                    data_loader['background'], data_loader['defects'],
                                                    self.metric_models, self.metrics)
        for metric in self.metrics:
            if isinstance(self.metrics[metric], dict):
                writer.add_scalars(f'Metrics/{metric}', self.metrics[metric], epoch)
            else:
                writer.add_scalar(f'Metrics/{metric}', self.metrics[metric], epoch)

        for metric_names in self.metrics:
            print(f'{metric_names}: {self.metrics[metric_names]} at epoch {epoch}')

    def _train_generator_once(self, bg_data, df_labels, df_data):
        self.optimizers['G'].zero_grad()
        if self.opt.style_norm_block_type == 'adain':
            self.optimizers['E'].zero_grad()
        if self.opt.style_norm_block_type == 'sean' and self.opt.style_distill:
            gan_loss, clf_loss, rec_loss, sd_cyc_loss, sd_con_loss, distill_latent_loss, distill_embed_loss = \
                self.model('generator', bg_data, df_labels, df_data)
            self.losses['distill']['embed'].append(distill_embed_loss.item())
            self.losses['distill']['latent'].append(distill_latent_loss.item())
        else:
            gan_loss, clf_loss, rec_loss, sd_cyc_loss, sd_con_loss = \
                self.model('generator', bg_data, df_labels, df_data)
        g_loss = gan_loss + \
                 clf_loss * self.loss_weights['clf_g'] + \
                 rec_loss * self.loss_weights['rec'] + \
                 sd_cyc_loss * self.loss_weights['sd_cyc'] + \
                 sd_con_loss * self.loss_weights['sd_con']
        # self.scaler.scale(g_loss).backward()
        # self.scaler.step(self.optimizers['G'])
        # if self.opt.style_norm_block_type == 'adain':
        #     self.scaler.step(self.optimizers['E'])
        # self.scaler.update()
        g_loss.backward()
        self.optimizers['G'].step()
        if self.opt.style_norm_block_type == 'adain':
            self.optimizers['E'].step()
        self.losses['gan']['G'].append(gan_loss.item())
        self.losses['clf']['G'].append(clf_loss.item())
        self.losses['aux']['rec'].append(rec_loss.item())
        self.losses['aux']['cyc'].append(sd_cyc_loss.item())
        self.losses['aux']['con'].append(sd_con_loss.item())

    def _train_discriminator_once(self, bg_data, df_labels, df_data):
        self.optimizers['D'].zero_grad()
        gan_loss, clf_loss = self.model('discriminator', bg_data, df_labels, df_data)
        d_loss = gan_loss + clf_loss * self.loss_weights['clf_d']
        # self.scaler.scale(d_loss).backward()
        # self.scaler.step(self.optimizers['D'])
        # self.scaler.update()
        d_loss.backward()
        self.optimizers['D'].step()
        self.losses['gan']['D'].append(gan_loss.item())
        self.losses['clf']['D'].append(clf_loss.item())

    def _update_per_epoch(self, epoch=None):
        super()._update_per_epoch(epoch)
        self.model.update_per_epoch(epoch)
        # if sum(self.losses['aux']['con']) / len(self.losses['aux']['con']) < 1e-5:
        #     self.loss_weights['sd_con'] = max(self.loss_weights['sd_con'] / 2, 1)
        # elif sum(self.losses['aux']['con']) / len(self.losses['aux']['con']) >= 0.3:
        #     self.loss_weights['sd_con'] *= 2