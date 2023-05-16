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
    def __init__(self, opt, data_types):
        super().__init__(opt)
        self.loss_weights = {'rec': opt.loss_weight[0],
                             'clf_D': opt.loss_weight[1],
                             'clf_G': opt.loss_weight[2]}
        self.loss_types = ['rec', 'gan', 'clf']
        if opt.style_norm_block_type == 'sean' and opt.style_distill:
            self.loss_types.append('distill')
        self._init_losses()

        # initial attributes for dataset
        self.data_types = data_types

        # add mask_token's param
        self.optimizers['G'].add_param_group({'params': self.model.mask_token.parameters()})

    def _init_lr(self, opt):
        assert len(opt.lr) in (1, 2), f'length of lr must be 1 or 2, not {len(opt.lr)}'
        self.lr = {'D': opt.lr[0],
                   'G': opt.lr[1]} if len(opt.lr) == 2 else opt.lr[0]
        if opt.style_norm_block_type == 'adain':
            self.lr['E'] = opt.lr[1]

    def _init_losses(self):
        self.losses = {loss_type: defaultdict(list)
                       for loss_type in self.loss_types}

    def _write_tf_log(self, writer, epoch, val_loaders):
        # for losses
        for loss_type in self.loss_types:
            writer.add_scalars(f'Losses/{loss_type}', {key: sum(value) / (len(value) + 1e-12)
                                                       for key, value in self.losses[loss_type].items()}, epoch)
        for model_name in self.schedulers.keys():
            writer.add_scalar(f'Lr/{model_name}', self.schedulers[model_name].get_last_lr()[0], epoch)
        # for loss_type, values in self.losses.items():
        #     writer.add_scalar(f'Losses/{loss_type}', sum(values) / (len(values) + 1e-12), epoch)

        # for generated image
        if epoch % self.opt.save_img_freq == 0:
            for data_type in self.data_types:
                data, labels, _ = next(val_loaders[data_type])
                repaired_grid = self.model('mae_generate_grid', data, labels)
                writer.add_image(f'Images/{data_type}', repaired_grid, epoch)

            mask_token = self.model.mask_token.mask_token.detach().clone() \
                .view(self.model.mask_token.mask_token.size(1), self.opt.image_size, self.opt.image_size).cpu()
            mask_token = mask_token - mask_token.min()
            mask_token = mask_token / mask_token.max()
            # heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * mask_token), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            # mask_token = torch.from_numpy(heatmap.transpose(2, 0, 1)).float() / 255
            writer.add_image(f'Images/mask_token', mask_token, epoch)

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
            self._update_per_epoch(epoch)
        writer.close()

    def _train_epoch(self, data_loaders, epoch):
        lrs = ', '.join([f'lr_{model_name}: {self.schedulers[model_name].get_last_lr()[0]:.5f}'
                         for model_name in self.schedulers.keys()])
        pbar = tqdm(data_loaders['fusion'], colour='MAGENTA')
        # BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
        for bg_data, bg_labels, _ in pbar:
            self.iters += 1
            pbar.set_description(f'Epoch: [{epoch}/{self.opt.num_epochs}], '
                                 f'Iter: [{self.iters}/{self.opt.num_iters}], {lrs}')

            self._train_discriminator_once(bg_data, bg_labels)
            if self.iters % self.opt.num_critics == 0:
                self._train_generator_once(bg_data, bg_labels)

            if self.iters % self.opt.save_latest_freq == 0:
                self.model.save('latest')
                np.savetxt(self.iter_record_path, (epoch, self.iters), fmt='%i', delimiter=',')
            pbar.set_postfix(rec=f'{sum(self.losses["rec"]["train"]) / (len(self.losses["rec"]["train"]) + 1e-12):.4f}',
                             gan_D=f'{sum(self.losses["gan"]["D"]) / (len(self.losses["gan"]["D"]) + 1e-12):.4f}',
                             gan_G=f'{sum(self.losses["gan"]["G"]) / (len(self.losses["gan"]["G"]) + 1e-12):.4f}',
                             clf_D=f'{sum(self.losses["clf"]["D"]) / (len(self.losses["clf"]["D"]) + 1e-12):.4f}',
                             clf_G=f'{sum(self.losses["clf"]["G"]) / (len(self.losses["clf"]["G"]) + 1e-12):.4f}')

    @torch.no_grad()
    def _val_epoch(self, data_loader, epoch):
        pbar = tqdm(data_loader['fusion'], colour='MAGENTA')
        # BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
        for data, labels, _ in pbar:
            pbar.set_description(f'Validating model at epoch {epoch}... ')
            rec_loss, gan_loss, clf_loss = self.model('mae_inference', data, labels)
            self.losses['rec']['val'].append(rec_loss.item())
            self.losses['gan']['val'].append(gan_loss.item())
            self.losses['clf']['val'].append(clf_loss.item())
            pbar.set_postfix(rec=f'{sum(self.losses["rec"]["val"]) / (len(self.losses["rec"]["val"]) + 1e-12):.4f}',
                             gan=f'{sum(self.losses["gan"]["val"]) / (len(self.losses["gan"]["val"]) + 1e-12):.4f}',
                             clf=f'{sum(self.losses["clf"]["val"]) / (len(self.losses["clf"]["val"]) + 1e-12):.4f}')

    def _train_generator_once(self, data, labels):
        self.optimizers['G'].zero_grad()
        if self.opt.style_norm_block_type == 'adain':
            self.optimizers['E'].zero_grad()
        if self.opt.style_norm_block_type == 'sean' and self.opt.style_distill:
            rec_loss, gan_loss, clf_loss, distill_latent_loss, distill_embed_loss = \
                self.model('mae_generator', data, labels)
            g_loss = gan_loss + rec_loss * self.loss_weights['rec'] + clf_loss * self.loss_weights['clf_G']
            # g_loss += distill_latent_loss + distill_embed_loss
            self.losses['distill']['embed'].append(distill_embed_loss.item())
            self.losses['distill']['latent'].append(distill_latent_loss.item())
        else:
            rec_loss, gan_loss, clf_loss = self.model('mae_generator', data, labels)
            g_loss = gan_loss + rec_loss * self.loss_weights['rec'] + clf_loss * self.loss_weights['clf_G']
        # self.scaler.scale(g_loss).backward()
        # self.scaler.step(self.optimizers['G'])
        # if self.opt.style_norm_block_type == 'adain':
        #     self.scaler.step(self.optimizers['E'])
        # self.scaler.update()
        # for name, param in self.model.networks['G'].named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad.sum())
        #     else:
        #         print(name, param.grad)
        g_loss.backward()
        # print('-' * 100)
        # for name, param in self.model.networks['G'].named_parameters():
        #     print(name, param.sum())
        # print('-' * 100)
        self.optimizers['G'].step()
        # for name, param in self.model.networks['G'].named_parameters():
        #     print(name, param.sum())
        # exit()
        if self.opt.style_norm_block_type == 'adain':
            self.optimizers['E'].step()
        self.losses['rec']['train'].append(rec_loss.item())
        self.losses['gan']['G'].append(gan_loss.item())
        self.losses['clf']['G'].append(clf_loss.item())

    def _train_discriminator_once(self, data, labels):
        self.optimizers['D'].zero_grad()
        gan_loss, clf_loss = self.model('mae_discriminator', data, labels)
        d_loss = gan_loss + clf_loss * self.loss_weights['clf_D']
        # self.scaler.scale(d_loss).backward()
        # self.scaler.step(self.optimizers['D'])
        # self.scaler.update()
        d_loss.backward()
        self.optimizers['D'].step()
        self.losses['gan']['D'].append(gan_loss.item())
        self.losses['clf']['D'].append(clf_loss.item())
