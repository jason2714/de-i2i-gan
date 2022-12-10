import random

import torch
from models.wgan_model import WGanModel
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


class WGanTrainer(BaseTrainer):
    """
    BaseTrainer receives the options and initialize optimizers, models, losses, iters, D(x), num_epochs
    """

    def __init__(self, opt, iters_per_epoch=math.inf):
        super().__init__(opt, iters_per_epoch)
        self.fix_noise = torch.rand(opt.num_display_images, opt.noise_dim, 1, 1)
        # self.opt.name += f'_{self.model.clipping_limit}'
        # self.fix_noise.requires_grad = True
        # print(f'fix noise required grad = {self.fix_noise.requires_grad}')

    def _cal_dis_grad(self, real_data, fake_data):
        alpha = torch.rand(real_data.shape[0], 1, 1, 1).expand_as(real_data).to(real_data.device)
        max_data = Variable(alpha * real_data + (1 - alpha) * fake_data, requires_grad=True)
        mix_logits = self.model.netD(max_data)
        mix_grad = grad(mix_logits, max_data, grad_outputs=torch.ones_like(mix_logits))[0]
        # mix_logits.backward(torch.ones_like(mix_logits))
        # mix_grad = max_data.grad
        return mix_grad.max()
        # return mean(fake_grad)

    @torch.no_grad()
    def _generate_image(self):
        # print(f'inside generate_image')
        # print(f'fix noise required grad = {self.fix_noise.requires_grad}')
        fake_data = self.model.netG(self.fix_noise)
        # print(f'fake_data required grad = {fake_data.requires_grad}')
        # print(fake_data.min(), fake_data.max())
        fake_data = (fake_data / 2 + 0.5).detach().cpu()
        # print(fake_data.min(), fake_data.max())
        # exit()
        return fake_data

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
        generated_images = self._generate_image()
        # print(generated_images.shape)
        nrow = int(math.sqrt(self.opt.num_display_images))
        image_grid = make_grid(generated_images, nrow=nrow)
        writer.add_image('Generated Image', image_grid, epoch)

    def train(self, train_loader, val_loader=None):
        """
        epoch start with 1, end with num_epochs
        """
        writer = SummaryWriter(self.opt.log_dir / self.opt.name)
        for epoch in range(1, self.opt.num_epochs + 1):
            self.losses.clear()
            self.dis_outputs.clear()
            self._train_epoch(train_loader, epoch)
            if val_loader is not None:
                self._val_epoch(val_loader)
            self._write_tf_log(writer, epoch)
            if epoch % self.opt.save_epoch_freq == 0:
                self.model.save(epoch)
        writer.close()

    def _train_epoch(self, data_loader, epoch):
        pbar = tqdm(data_loader, colour='MAGENTA')
        # BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
        for batch_data in pbar:
            self.iters += 1
            pbar.set_description(f'Epoch: [{epoch}/{self.opt.num_epochs}], '
                                 f'Iter: [{self.iters}/{self.opt.num_iters}]')
            # print(batch_data.min(), batch_data.max())
            batch_data = batch_data.to(self.opt.device)
            self._train_discriminator_once(batch_data)
            if self.iters % self.opt.num_critics == 0:
                self._train_generator_once(batch_data.shape[0])
            if self.iters % self.opt.save_latest_freq == 0:
                self.model.save('latest')
            pbar.set_postfix(w_dis=f'{-sum(self.losses["gan_D"]) / len(self.losses["gan_D"]):.4f}',
                             g_loss=f'{sum(self.losses["gan_G"]) / (len(self.losses["gan_G"]) + 1e-12):.4f}')
            # ,
            # dis_grad=f'{max(self.losses["dis_grad"]):.4f}')
        for model_name in self.schedulers.keys():
            self.schedulers[model_name].step()

    @torch.no_grad()
    def _val_epoch(self, data_loader):
        pbar = tqdm(data_loader, colour='MAGENTA')
        # BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
        for batch_data in pbar:
            pbar.set_description('Validating... ')
            batch_data = batch_data.to(self.opt.device)
            val_logits = self.model.netD(batch_data)
            self.dis_outputs['val'] += val_logits.flatten().tolist()

    def _train_generator_once(self, batch_size):
        self.optimizers['G'].zero_grad()
        fake_data = self.model.netG(batch_size)
        fake_logits = self.model.netD(fake_data)
        g_loss = -fake_logits.mean()
        g_loss.backward()
        self.optimizers['G'].step()
        self.losses['gan_G'].append(g_loss.item())

    def _train_discriminator_once(self, real_data):
        self.model.weight_clipping()
        self.optimizers['D'].zero_grad()
        fake_data = self.model.netG(real_data.shape[0])
        fake_logits = self.model.netD(fake_data.detach_())
        real_logits = self.model.netD(real_data)
        w_distance = real_logits.mean() - fake_logits.mean()
        d_loss = -w_distance
        # self.losses['dis_grad'].append(self._cal_dis_grad(real_data, fake_data))
        # self.optimizers['D'].zero_grad()
        d_loss.backward()
        self.optimizers['D'].step()
        self.losses['gan_D'].append(d_loss.item())
        self.dis_outputs['real'] += real_logits.flatten().tolist()
        self.dis_outputs['fake'] += fake_logits.flatten().tolist()
