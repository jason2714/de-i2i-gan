from metrics.fid_score import calculate_fid
from metrics.inception import InceptionV3
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


class MAETrainer(BaseTrainer):
    def __init__(self, opt, iters_per_epoch=math.inf):
        super().__init__(opt, iters_per_epoch)
        self.loss_types = ['rec']
        self._init_losses()
        # if opt.phase == 'val':
        #     block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[opt.dims]
        #     self.inception_model = InceptionV3([block_idx]).to(opt.device, non_blocking=True)
        #     self.inception_model.eval()

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
            bg_data, bg_labels, _ = next(val_loaders['background_inf'])
            repaired_grid = self.model('generate_mask_grid', bg_data, bg_labels)
            writer.add_image('Images/Masked', repaired_grid, epoch)

    def train(self, train_loaders, val_loaders=None):
        """
        epoch start with 1, end with num_epochs
        """
        # TODO
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
        pbar = tqdm(data_loaders['background'], colour='MAGENTA')
        # BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
        for bg_data, bg_labels, _ in pbar:
            self.iters += 1
            pbar.set_description(f'Epoch: [{epoch}/{self.opt.num_epochs}], '
                                 f'Iter: [{self.iters}/{self.opt.num_iters}]')

            # # get bg data and truncate them to the same as batch_size of defect data
            # df_data, df_labels, _ = next(data_loaders['defects'])

            # self._train_discriminator_once(bg_data, df_labels, df_data)
            self._train_generator_once(bg_data, bg_labels)

            if self.iters % self.opt.save_latest_freq == 0:
                self.model.save('latest')
                np.savetxt(self.iter_record_path, (epoch, self.iters), fmt='%i', delimiter=',')
            pbar.set_postfix(rec=f'{sum(self.losses["rec"]["train"]) / (len(self.losses["rec"]["train"]) + 1e-12):.4f}')
        for model_name in self.schedulers.keys():
            self.schedulers[model_name].step()

    @torch.no_grad()
    def _val_epoch(self, data_loader, epoch):
        mse_losses = []
        pbar = tqdm(data_loader['background'], colour='MAGENTA')
        # BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
        for data, labels, _ in pbar:
            pbar.set_description(f'Validating model at epoch {epoch}... ')
            rec_loss = self.model('mae_inference', data, labels)
            mse_losses.append(rec_loss.item())
            pbar.set_postfix(rec=f'{sum(mse_losses) / (len(mse_losses) + 1e-12):.4f}')
        self.losses['rec']['val'] = sum(mse_losses) / (len(mse_losses) + 1e-12)

    def _train_generator_once(self, data, labels):
        self.optimizers['G'].zero_grad()
        rec_loss = self.model('mae', data, labels)
        rec_loss.backward()
        self.optimizers['G'].step()
        self.losses['rec']['train'].append(rec_loss.item())

    def _train_discriminator_once(self, bg_data, df_labels, df_data):
        self.optimizers['D'].zero_grad()
        # gan_loss, clf_loss = self.model('discriminator', bg_data, df_labels, df_data)
        # d_loss = gan_loss + clf_loss * self.loss_weights['clf_d']
        # d_loss.backward()
        # self.optimizers['D'].step()
        # self.losses['gan']['D'].append(gan_loss.item())
        # self.losses['clf']['D'].append(clf_loss.item())
