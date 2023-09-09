import torch
from collections import defaultdict
from tqdm import tqdm
import math
from torch.utils.tensorboard import SummaryWriter
from metrics.fid_score import calculate_fid_from_model

from trainers.base_trainer import BaseTrainer
import numpy as np
import cv2


class ViTTrainer(BaseTrainer):
    def __init__(self, opt):
        super().__init__(opt)
        self.loss_types = ['clf']
        self.metric_types = ['acc']
        self._init_losses()
        self.metrics = {metric_type: defaultdict(float)
                        for metric_type in self.metric_types}

    def _init_lr(self, opt):
        assert len(opt.lr) in (1, 2), f'length of lr must be 1 or 2, not {len(opt.lr)}'
        self.lr = opt.lr[0]

    def _init_losses(self):
        self.losses = {loss_type: defaultdict(list)
                       for loss_type in self.loss_types}

    def _write_tf_log(self, writer, epoch):
        # for losses
        for loss_type in self.loss_types:
            writer.add_scalars(f'Losses/{loss_type}', {key: sum(value) / (len(value) + 1e-12)
                                                       for key, value in self.losses[loss_type].items()}, epoch)
        for model_name in self.schedulers.keys():
            writer.add_scalar(f'Lr/{model_name}', self.schedulers[model_name].get_last_lr()[0], epoch)
        if self.opt.phase == 'val':
            for metric_type in self.metric_types:
                writer.add_scalars(f'Metrics/{metric_type}', {key: value for key, value in
                                                              self.metrics[metric_type].items()}, epoch)

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
            self._write_tf_log(writer, epoch)
            if epoch % self.opt.save_ckpt_freq == 0:
                self.model.save(epoch)
            self._update_per_epoch(epoch)
        writer.close()

    def _train_epoch(self, data_loader, epoch):
        lrs = ', '.join([f'lr_{model_name}: {self.schedulers[model_name].get_last_lr()[0]:.5f}'
                         for model_name in self.schedulers.keys()])
        pbar = tqdm(data_loader, colour='MAGENTA')
        # BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
        num_acc = 0
        num_data = 0
        for data, labels, _ in pbar:
            self.iters += 1
            num_data += self.opt.batch_size
            pbar.set_description(f'Epoch: [{epoch}/{self.opt.num_epochs}], '
                                 f'Iter: [{self.iters}/{self.opt.num_iters}], {lrs}')
            num_acc += self._train_classifier_once(data, labels)

            if self.iters % self.opt.save_latest_freq == 0:
                self.model.save('latest')
                np.savetxt(self.iter_record_path, (epoch, self.iters), fmt='%i', delimiter=',')
            pbar.set_postfix(acc=f'{num_acc / num_data:.3f} ({num_acc}/{num_data})',
                             clf=f'{sum(self.losses["clf"]["train"]) / (len(self.losses["clf"]["train"]) + 1e-12):.4f}')
        self.metrics['acc']['train'] = num_acc / num_data

    @torch.no_grad()
    def _val_epoch(self, data_loader, epoch):
        pbar = tqdm(data_loader, colour='MAGENTA')
        # BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
        num_acc = 0
        num_data = 0
        for data, labels, _ in pbar:
            num_data += self.opt.batch_size
            pbar.set_description(f'Validating model at epoch {epoch}... ')
            logits, clf_loss = self.model('inference', data, labels)
            predictions = (torch.sigmoid(logits) >= 0.5).int().cpu()
            num_acc += (predictions == labels).all(dim=1).sum()

            self.losses['clf']['val'].append(clf_loss.item())
            pbar.set_postfix(acc=f'{num_acc / num_data:.3f} ({num_acc}/{num_data})',
                             clf=f'{sum(self.losses["clf"]["val"]) / (len(self.losses["clf"]["val"]) + 1e-12):.4f}')
        self.metrics['acc']['val'] = num_acc / num_data

    def _train_classifier_once(self, data, labels):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
        logits, clf_loss = self.model('train', data, labels)
        self.scaler.scale(clf_loss).backward()
        # for optimizer in self.optimizers.values():
        #     self.scaler.step(optimizer)
        self.scaler.step(self.optimizers['C'])
        self.scaler.update()
        self.losses['clf']['train'].append(clf_loss.item())

        predictions = (torch.sigmoid(logits) >= 0.5).int().cpu()
        acc = (predictions == labels).all(dim=1).sum()
        return acc
