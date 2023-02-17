from metrics.fid_score import calculate_fid
from metrics.inception import InceptionV3
from trainers.base_trainer import BaseTrainer
import torch
from collections import defaultdict
from tqdm import tqdm
from torchvision.utils import make_grid
import math
from torch.utils.tensorboard import SummaryWriter
from metrics.fid_score import calculate_fid_from_model

from trainers.base_trainer import BaseTrainer
import numpy as np
import cv2


class DefectGanTrainer(BaseTrainer):
    def __init__(self, opt, iters_per_epoch=math.inf):
        super().__init__(opt, iters_per_epoch)
        assert len(opt.loss_weight) == 5, f'length of loss weights must be 5, not {len(opt.loss_weight)}'
        self.loss_weights = {'clf_d': opt.loss_weight[0],
                             'clf_g': opt.loss_weight[1],
                             'rec': opt.loss_weight[2],
                             'sd_cyc': opt.loss_weight[3],
                             'sd_con': opt.loss_weight[4]}
        self.loss_types = ['gan', 'clf', 'aux']
        self._init_losses()
        if opt.phase == 'val':
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[opt.dims]
            self.inception_model = InceptionV3([block_idx]).to(opt.device, non_blocking=True)
            self.inception_model.eval()

    def _init_lr(self, opt):
        assert len(opt.lr) in (1, 2), f'length of lr must be 1 or 2, not {len(opt.lr)}'
        self.lr = {'D': opt.lr[0],
                   'G': opt.lr[1]} if len(opt.lr) == 2 else opt.lr[0]
        # def _cal_dis_grad(self, real_data, fake_data):

    def _init_losses(self):
        self.losses = {loss_type: defaultdict(list)
                       for loss_type in self.loss_types}

    #     alpha = torch.rand(real_data.shape[0], 1, 1, 1).expand_as(real_data).to(real_data.device)
    #     max_data = Variable(alpha * real_data + (1 - alpha) * fake_data, requires_grad=True)
    #     mix_logits = self.model.netD(max_data)
    #     mix_grad = grad(mix_logits, max_data, grad_outputs=torch.ones_like(mix_logits))[0]
    #     # mix_logits.backward(torch.ones_like(mix_logits))
    #     # mix_grad = max_data.grad
    #     return mix_grad.max()
    #     # return mean(fake_grad)

    @torch.no_grad()
    def _generate_fake_grids(self, data_loader):
        nm_images = None
        single_df_images = []
        multi_df_images = None
        init_flag = True
        bg_data, bg_labels, _ = next(data_loader['background'])
        _, df_labels, _ = next(iter(data_loader['defects']))
        labels = torch.cat([torch.eye(self.opt.label_nc)[1:], df_labels], dim=0)
        for data in bg_data:
            single_df_images.append(data / 2 + 0.5)
            data = data.unsqueeze(0).to(self.model.netG.device)
            df_data, df_prob = self.model('inference', data.expand(labels.size(0), -1, -1, -1), labels)
            foreground = torch.clamp((df_data - data * (1 - df_prob)) / (df_prob + 1e-8), min=-1, max=1)
            # print(torch.min(foreground), torch.max(foreground))
            df_data = (df_data / 2 + 0.5).detach().cpu()
            foreground = (foreground / 2 + 0.5).detach().cpu()
            for idx, (slice_data, slice_prob, slice_foreground) in enumerate(zip(df_data, df_prob, foreground)):
                if idx == self.opt.label_nc - 1:
                    break
                slice_prob = slice_prob.squeeze(0).detach().cpu()
                # print(torch.min(slice_prob), torch.max(slice_prob))
                heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * slice_prob), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
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

    def _write_tf_log(self, writer, epoch, val_loaders):
        # for losses
        for loss_type in self.loss_types:
            writer.add_scalars(f'Losses/{loss_type}', {key: sum(value) / (len(value) + 1e-12)
                                                       for key, value in self.losses[loss_type].items()}, epoch)
        # for discriminator outputs
        # writer.add_scalars(f'D(x)', {key: sum(value) / len(value)
        #                              for key, value in self.losses.items()
        #                              if key.startswith('gan_')}, epoch)
        # for generated image
        if epoch % self.opt.save_img_freq == 0:
            df_grid, mtp_df_grid = self._generate_fake_grids(val_loaders)
            writer.add_image('Images/Single Defect', df_grid, epoch)
            # writer.add_image('Images/Single Normal', nm_grid, epoch)
            # writer.add_image('Images/Single Defect Distribution', df_prob_grid, epoch)
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
                    self._val_epoch(val_loaders, epoch)
                    for metric in self.metrics:
                        writer.add_scalar(f'Metrics/{metric}', self.metrics[metric], epoch)
        writer.close()

    def _train_epoch(self, data_loaders, epoch):
        pbar = tqdm(data_loaders['defects'], colour='MAGENTA')
        # BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
        for df_data, df_labels, _ in pbar:
            self.iters += 1
            pbar.set_description(f'Epoch: [{epoch}/{self.opt.num_epochs}], '
                                 f'Iter: [{self.iters}/{self.opt.num_iters}]')

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
        for model_name in self.schedulers.keys():
            self.schedulers[model_name].step()

    @torch.no_grad()
    def _val_epoch(self, data_loader, epoch):
        fid_value = calculate_fid_from_model(self.opt, self.model, self.inception_model, data_loader, 'Validating... ')
        print(f'FID: {fid_value} at epoch {epoch}')
        self.metrics['fid'] = fid_value

    def _train_generator_once(self, bg_data, df_labels, df_data):
        self.optimizers['G'].zero_grad()
        gan_loss, clf_loss, rec_loss, sd_cyc_loss, sd_con_loss = \
            self.model('generator', bg_data, df_labels, df_data)
        g_loss = gan_loss + clf_loss * self.loss_weights['clf_g'] + \
                 rec_loss * self.loss_weights['rec'] + \
                 sd_cyc_loss * self.loss_weights['sd_cyc'] + \
                 sd_con_loss * self.loss_weights['sd_con']
        g_loss.backward()
        self.optimizers['G'].step()
        self.losses['gan']['G'].append(gan_loss.item())
        self.losses['clf']['G'].append(clf_loss.item())
        self.losses['aux']['rec'].append(rec_loss.item())
        self.losses['aux']['cyc'].append(sd_cyc_loss.item())
        self.losses['aux']['con'].append(sd_con_loss.item())

    def _train_discriminator_once(self, bg_data, df_labels, df_data):
        self.optimizers['D'].zero_grad()
        gan_loss, clf_loss = self.model('discriminator', bg_data, df_labels, df_data)
        d_loss = gan_loss + clf_loss * self.loss_weights['clf_d']
        d_loss.backward()
        self.optimizers['D'].step()
        self.losses['gan']['D'].append(gan_loss.item())
        self.losses['clf']['D'].append(clf_loss.item())
