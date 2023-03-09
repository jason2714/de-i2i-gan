import torch
from collections import defaultdict
from tqdm import tqdm
import math
from torch.utils.tensorboard import SummaryWriter
from metrics.fid_score import calculate_fid_from_model

from trainers.base_trainer import BaseTrainer
import numpy as np
import cv2


class MAETrainer(BaseTrainer):
    def __init__(self, opt, iters_per_epoch=math.inf, data_types=None):
        super().__init__(opt, iters_per_epoch, data_types)
        self.loss_weights = {'rec': opt.loss_weight[0]}
        self.loss_types = ['rec', 'gan']
        self._init_losses()

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
        # for loss_type, values in self.losses.items():
        #     writer.add_scalar(f'Losses/{loss_type}', sum(values) / (len(values) + 1e-12), epoch)

        # for generated image
        if epoch % self.opt.save_img_freq == 0:
            for data_type in self.data_types:
                data, labels, _ = next(val_loaders[data_type])
                repaired_grid = self.model('mae_generate_grid', data, labels)
                writer.add_image(f'Images/{data_type}', repaired_grid, epoch)

    def train(self, train_loaders, val_loaders=None):
        """
        epoch start with 1, end with num_epochs
        """
        writer = SummaryWriter(self.opt.log_dir / self.opt.name)
        for epoch in range(self.first_epoch, self.opt.num_epochs + 1):
            self._init_losses()
            self._train_epoch(train_loaders, epoch)
            if self.opt.phase == 'val':
                self._val_epoch(val_loaders, epoch)
            self._write_tf_log(writer, epoch, val_loaders)
            if epoch % self.opt.save_ckpt_freq == 0:
                self.model.save(epoch)
        writer.close()

    def _train_epoch(self, data_loaders, epoch):
        pbar = tqdm(data_loaders['fusion'], colour='MAGENTA')
        # BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
        for bg_data, bg_labels, _ in pbar:
            self.iters += 1
            pbar.set_description(f'Epoch: [{epoch}/{self.opt.num_epochs}], '
                                 f'Iter: [{self.iters}/{self.opt.num_iters}]')

            self._train_discriminator_once(bg_data, bg_labels)
            if self.iters % self.opt.num_critics == 0:
                self._train_generator_once(bg_data, bg_labels)

            if self.iters % self.opt.save_latest_freq == 0:
                self.model.save('latest')
                np.savetxt(self.iter_record_path, (epoch, self.iters), fmt='%i', delimiter=',')
            pbar.set_postfix(rec=f'{sum(self.losses["rec"]["train"]) / (len(self.losses["rec"]["train"]) + 1e-12):.4f}',
                             gan_D=f'{sum(self.losses["gan"]["D"]) / (len(self.losses["gan"]["D"]) + 1e-12):.4f}',
                             gan_G=f'{sum(self.losses["gan"]["G"]) / (len(self.losses["gan"]["G"]) + 1e-12):.4f}')

        for model_name in self.schedulers.keys():
            self.schedulers[model_name].step()

    @torch.no_grad()
    def _val_epoch(self, data_loader, epoch):
        pbar = tqdm(data_loader['fusion'], colour='MAGENTA')
        # BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
        for data, labels, _ in pbar:
            pbar.set_description(f'Validating model at epoch {epoch}... ')
            rec_loss, gan_loss = self.model('mae_inference', data, labels)
            self.losses['rec']['val'].append(rec_loss.item())
            self.losses['gan']['val'].append(gan_loss.item())
            pbar.set_postfix(rec=f'{sum(self.losses["rec"]["val"]) / (len(self.losses["rec"]["val"]) + 1e-12):.4f}',
                             gan=f'{sum(self.losses["gan"]["val"]) / (len(self.losses["gan"]["val"]) + 1e-12):.4f}')

    def _train_generator_once(self, data, labels):
        self.optimizers['G'].zero_grad()
        rec_loss, gan_loss = self.model('mae_generator', data, labels)
        g_loss = gan_loss + rec_loss * self.loss_weights['rec']
        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.optimizers['G'])
        self.scaler.update()
        # g_loss.backward()
        # self.optimizers['G'].step()
        self.losses['rec']['train'].append(rec_loss.item())
        self.losses['gan']['G'].append(gan_loss.item())

    def _train_discriminator_once(self, data, labels):
        self.optimizers['D'].zero_grad()
        gan_loss = self.model('mae_discriminator', data, labels)
        d_loss = gan_loss
        self.scaler.scale(d_loss).backward()
        self.scaler.step(self.optimizers['D'])
        self.scaler.update()
        # d_loss.backward()
        # self.optimizers['D'].step()
        self.losses['gan']['D'].append(gan_loss.item())
