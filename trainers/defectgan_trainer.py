from trainers.base_trainer import BaseTrainer
import torch
from collections import defaultdict
from tqdm import tqdm
from torchvision.utils import make_grid
import math
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import inspect
from models import find_model_using_name
from torch.autograd import grad, Variable

from trainers.base_trainer import BaseTrainer


class DefectGanTrainer(BaseTrainer):
    def __init__(self, opt, iters_per_epoch=math.inf):
        super().__init__(opt, iters_per_epoch)
        assert len(opt.loss_weight) == 5, f'length of loss weights must be 5, not {len(opt.loss_weight)}'
        self.loss_weights = {'clf_d': opt.loss_weight[0],
                             'clf_g': opt.loss_weight[1],
                             'rec': opt.loss_weight[2],
                             'sd_cyc': opt.loss_weight[3],
                             'sd_con': opt.loss_weight[4]}

    def _init_lr(self, opt):
        assert len(opt.lr) in (1, 2), f'length of lr must be 1 or 2, not {len(opt.lr)}'
        self.lr = {'D': opt.lr[0],
                   'G': opt.lr[1]} if len(opt.lr) == 2 else opt.lr
        # def _cal_dis_grad(self, real_data, fake_data):

    #     alpha = torch.rand(real_data.shape[0], 1, 1, 1).expand_as(real_data).to(real_data.device)
    #     max_data = Variable(alpha * real_data + (1 - alpha) * fake_data, requires_grad=True)
    #     mix_logits = self.model.netD(max_data)
    #     mix_grad = grad(mix_logits, max_data, grad_outputs=torch.ones_like(mix_logits))[0]
    #     # mix_logits.backward(torch.ones_like(mix_logits))
    #     # mix_grad = max_data.grad
    #     return mix_grad.max()
    #     # return mean(fake_grad)
    #
    # @torch.no_grad()
    # def _generate_image(self):
    #     # print(f'inside generate_image')
    #     # print(f'fix noise required grad = {self.fix_noise.requires_grad}')
    #     fake_data = self.model.netG(self.fix_noise)
    #     # print(f'fake_data required grad = {fake_data.requires_grad}')
    #     # print(fake_data.min(), fake_djaata.max())
    #     fake_data = (fake_data / 2 + 0.5).detach().cpu()
    #     # print(fake_data.min(), fake_data.max())
    #     # exit()
    #     return fake_data

    def _write_tf_log(self, writer, epoch):
        # for losses
        for key, value in self.losses.items():
            scalar = sum(value) / len(value)
            writer.add_scalar(f'Loss/{key}', scalar, epoch)
        # for discriminator outputs
        writer.add_scalars(f'D(x)', {key: sum(value) / len(value)
                                     for key, value in self.dis_outputs.items()
                                     }, epoch)
        # for generated image
        # generated_images = self._generate_image()
        # nrow = int(math.sqrt(self.opt.num_display_images))
        # image_grid = make_grid(generated_images, nrow=nrow)
        # writer.add_image('Generated Image', image_grid, epoch)

    def train(self, train_loaders, val_loaders=None):
        """
        epoch start with 1, end with num_epochs
        """
        writer = SummaryWriter(self.opt.log_dir / self.opt.name)
        for epoch in range(1, self.opt.num_epochs + 1):
            self.losses.clear()
            self._train_epoch(train_loaders, epoch)
            # if val_loader is not None:
            #     self._val_epoch(val_loaders)
            self._write_tf_log(writer, epoch)
            if epoch % self.opt.save_epoch_freq == 0:
                self.model.save(epoch)
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
            bg_data, bg_labels = bg_data[:df_data.size(0)], bg_labels[:df_data.size(0)]

            self._train_discriminator_once(bg_data, df_labels, df_data)
            if self.iters % self.opt.num_critics == 0:
                self._train_generator_once(bg_data, df_labels, df_data)

            if self.iters % self.opt.save_latest_freq == 0:
                self.model.save('latest')
            pbar.set_postfix(w_dis=f'{-sum(self.losses["gan_D"]) / len(self.losses["gan_D"]):.4f}',
                             g_loss=f'{sum(self.losses["gan_G"]) / (len(self.losses["gan_G"]) + 1e-12):.4f}')
            # ,
            # dis_grad=f'{max(self.losses["dis_grad"]):.4f}')
        for model_name in self.schedulers.keys():
            self.schedulers[model_name].step()

    # @torch.no_grad()
    # def _val_epoch(self, data_loader):
    #     pbar = tqdm(data_loader, colour='MAGENTA')
    #     # BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
    #     for batch_data, labels, _ in pbar:
    #         pbar.set_description('Validating... ')
    #         batch_data = batch_data.to(self.opt.device)
    #         val_logits = self.model.netD(batch_data)
    #         self.dis_outputs['val'] += val_logits.flatten().tolist()

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
        self.losses['gan_G'].append(g_loss.item())

    def _train_discriminator_once(self, bg_data, df_labels, df_data):
        self.optimizers['D'].zero_grad()
        gan_loss, clf_loss = self.model('discriminator', bg_data, df_labels, df_data)
        d_loss = gan_loss + clf_loss * self.loss_weights['clf_d']
        d_loss.backward()
        self.optimizers['D'].step()
        self.losses['gan_D'].append(d_loss.item())
