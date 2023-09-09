"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import random
from pathlib import Path
import time
import datetime
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from core.model import build_model, SEAN, MaskToken
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher, get_test_style_loader
import core.utils as utils
from core.utils import repair_mask
from metrics.eval import calculate_metrics
from core.utils import get_style_code
from core.diffaug import DiffAugment


class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets, self.nets_ema = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        if args.mode in ('train', 'pretrain'):
            self.optims = Munch()
            for net in self.nets.keys():
                if net in ('fan',):
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net in ('mapping_network', 'feature_extractor') else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)
            if args.pretrain_dir is None:
                self.ckptios = [
                    CheckpointIO(args.checkpoint_dir / '{:06d}_nets.ckpt', data_parallel=True, **self.nets),
                    CheckpointIO(args.checkpoint_dir / '{:06d}_nets_ema.ckpt', data_parallel=True, **self.nets_ema),
                    CheckpointIO(args.checkpoint_dir / '{:06d}_optims.ckpt', **self.optims)]
            else:
                self.ckptios = [
                    CheckpointIO(args.checkpoint_dir / '{:06d}_nets.ckpt',
                                 pretrain_path=args.pretrain_dir / '{:06d}_nets.ckpt', data_parallel=True, **self.nets),
                    CheckpointIO(args.checkpoint_dir / '{:06d}_nets_ema.ckpt',
                                 pretrain_path=args.pretrain_dir / '{:06d}_nets_ema.ckpt', data_parallel=True,
                                 **self.nets_ema),
                    CheckpointIO(args.checkpoint_dir / '{:06d}_optims.ckpt', **self.optims)]
        else:
            self.ckptios = [
                CheckpointIO(args.checkpoint_dir / '{:06d}_nets_ema.ckpt', data_parallel=True, **self.nets_ema)]

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name) and ('feature_extractor' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _load_pretrain_checkpoint(self, step):
        for ckptio in self.ckptios:
            if ckptio.pretrain_path is not None:
                ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def pretrain(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        self.mask_token = MaskToken(args).to(self.device)
        # add mask_token's param
        self.optims.generator.add_param_group({'params': self.mask_token.parameters()})

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.train, loaders.ref, args.latent_dim, 'train', args.norm_type, args.hidden_nc)
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'pretrain', args.norm_type, args.hidden_nc)
        inputs_val = next(fetcher_val)

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            # x_real, y_org = inputs.x_src, inputs.y_src
            # z_trg = inputs.z_trg

            x_real, x_real2, y_org = inputs.x_ref, inputs.x_ref2, inputs.y_ref
            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

            masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

            if args.norm_type == 'adain':
                # train the discriminator
                d_loss, d_losses_latent = compute_mae_d_loss(
                    nets, self.mask_token, args, x_real, y_org, z_trg=z_trg, masks=masks)
                self._reset_grad()
                d_loss.backward()
                optims.discriminator.step()

            d_loss, d_losses_ref = compute_mae_d_loss(
                nets, self.mask_token, args, x_real, y_org, z_trg=None, masks=masks)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            if args.norm_type == 'adain':
                # train the generator
                g_loss, g_losses_latent = compute_mae_g_loss(
                    nets, self.mask_token, args, (x_real, x_real2), y_org, z_trgs=(z_trg, z_trg2), masks=masks)
                self._reset_grad()
                g_loss.backward()
                optims.generator.step()
                optims.mapping_network.step()
                optims.style_encoder.step()

            g_loss, g_losses_ref = compute_mae_g_loss(
                nets, self.mask_token, args, (x_real, x_real2), y_org, z_trgs=None, masks=masks)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            if args.norm_type == 'sean':
                optims.feature_extractor.step()

            # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            if args.norm_type == 'adain':
                moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
                moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

            # print out log info
            if (i + 1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i + 1, args.total_iters)
                all_losses = dict()
                if args.norm_type == 'adain':
                    for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                            ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                        for key, value in loss.items():
                            all_losses[prefix + key] = value
                    all_losses['G/lambda_ds'] = args.lambda_ds
                elif args.norm_type == 'sean':
                    for loss, prefix in zip([d_losses_ref, g_losses_ref],
                                            ['D/ref_', 'G/ref_']):
                        for key, value in loss.items():
                            all_losses[prefix + key] = value
                else:
                    raise NotImplementedError('Norm type [%s] is not found' % args.norm_type)
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

            # generate images for debugging
            if (i + 1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)
                utils.debug_mask_image(nets, args, mask_token=self.mask_token, inputs=inputs_val, step=i + 1)
                utils.debug_mask_image(nets_ema, args, mask_token=self.mask_token, inputs=inputs_val, step=i + 1,
                                       is_ema=True)

            # save model checkpoints
            if (i + 1) % args.save_every == 0:
                self._save_checkpoint(step=i + 1)

            # # compute FID and LPIPS if necessary
            # if (i + 1) % args.eval_every == 0:
            #     calculate_metrics(nets_ema, args, i + 1, mode='latent')
            #     calculate_metrics(nets_ema, args, i + 1, mode='reference')

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train', args.norm_type, args.hidden_nc)
        fetcher_val = InputFetcher(loaders.val, loaders.val2, args.latent_dim, 'val', args.norm_type, args.hidden_nc)
        inputs_val = next(fetcher_val)

        # def count_sean_params(model):
        #     total_params = 0
        #     for module in model.modules():
        #         if isinstance(module, SEAN):
        #             total_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
        #     return total_params
        #
        # num_sean_params = count_sean_params(nets.generator)
        # print(f"Number of parameters in SEAN layers: {num_sean_params}")
        #
        # def count_total_params(model):
        #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
        #
        # num_total_params = count_total_params(nets.generator)
        # print(f"Total number of parameters in the model: {num_total_params}")
        # print(num_sean_params / num_total_params)
        # exit()

        # resume training if necessary
        if args.pretrain_iter is not None:
            self._load_pretrain_checkpoint(args.pretrain_iter)

        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        # inputs_val = next(fetcher)
        # os.makedirs(args.sample_dir, exist_ok=True)
        # self.mask_token = MaskToken(args).to(self.device)
        # utils.debug_mask_image(nets, args, mask_token=self.mask_token, inputs=inputs_val, step=1)
        # utils.debug_mask_image(nets_ema, args, mask_token=self.mask_token, inputs=inputs_val, step=1, is_ema=True)

        # os.makedirs(args.sample_dir, exist_ok=True)
        # utils.debug_image(nets_ema, args, inputs=inputs_val, step=1)
        # exit()

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            x_real, y_org = inputs.x_src, inputs.y_src
            x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

            masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

            if args.norm_type == 'adain':
                # train the discriminator
                d_loss, d_losses_latent = compute_d_loss(
                    nets, args, x_real, y_org, y_trg, z_trg=z_trg, masks=masks)
                self._reset_grad()
                d_loss.backward()
                optims.discriminator.step()

            d_loss, d_losses_ref = compute_d_loss(
                nets, args, x_real, y_org, y_trg, x_ref=x_ref, masks=masks)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            if args.norm_type == 'adain':
                # train the generator
                g_loss, g_losses_latent = compute_g_loss(
                    nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=masks)
                self._reset_grad()
                g_loss.backward()
                optims.generator.step()
                optims.mapping_network.step()
                optims.style_encoder.step()

            nets.feature_extractor.module.track_running_stats = True
            g_loss, g_losses_ref = compute_g_loss(
                nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks)
            nets.feature_extractor.module.track_running_stats = False
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            if args.norm_type == 'sean':
                optims.feature_extractor.step()

            # compute moving average of network parameters
            nets.feature_extractor.module.update_stats()

            # # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            if args.norm_type == 'adain':
                moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
                moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)
            # TODO temp remove
            # else:
            #     moving_average_for_sean_blocks(nets.generator, nets_ema.generator, beta=0.999)
            # decay weight for diversity sensitive loss
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # print out log info
            if (i + 1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i + 1, args.total_iters)
                all_losses = dict()
                if args.norm_type == 'adain':
                    for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                            ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                        for key, value in loss.items():
                            all_losses[prefix + key] = value
                    all_losses['G/lambda_ds'] = args.lambda_ds
                elif args.norm_type == 'sean':
                    for loss, prefix in zip([d_losses_ref, g_losses_ref],
                                            ['D/ref_', 'G/ref_']):
                        for key, value in loss.items():
                            all_losses[prefix + key] = value
                else:
                    raise NotImplementedError('Norm type [%s] is not found' % args.norm_type)
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

            # generate images for debugging
            if (i + 1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)
                utils.debug_image(nets_ema, args, inputs=inputs_val, step=i + 1)

            # save model checkpoints
            if (i + 1) % args.save_every == 0:
                self._save_checkpoint(step=i + 1)

            # compute FID and LPIPS if necessary
            if (i + 1) % args.eval_every == 0:
                calculate_metrics(nets_ema, args, i + 1, mode='latent')
                calculate_metrics(nets_ema, args, i + 1, mode='reference')

    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        args.result_dir.mkdir(exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test', args.norm_type, args.hidden_nc))
        ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test', args.norm_type, args.hidden_nc))

        fname = args.result_dir / 'reference.jpg'
        print('Working on {}...'.format(fname))
        utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)

        # fname = args.result_dir / 'video_ref.mp4'
        # print('Working on {}...'.format(fname))
        # utils.video_ref(nets_ema, args, src.x, ref.x, ref.y, fname)

    @torch.no_grad()
    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')
        calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')

    @torch.no_grad()
    def update_sean_stats(self, loaders):
        args = self.args
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        args = self.args
        nets_ema = self.nets_ema

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train', args.norm_type, args.hidden_nc)

        nets_ema.generator.module.track_running_stats = True
        labels_cnter = {label: 0 for label in range(args.num_domains)}
        while not utils.check_values(labels_cnter, 10000):
            # fetch images and labels
            inputs = next(fetcher)
            x_real, y_org = inputs.x_src, inputs.y_src
            x_ref, y_trg = inputs.x_ref, inputs.y_ref
            for label in y_trg:
                labels_cnter[label.item()] += 1
            print(labels_cnter)

            masks = nets_ema.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

            # adversarial loss
            s_trg = get_style_code(nets_ema, args.norm_type, args.num_embeds, y_trg, x_ref, z_trg=None)
            _ = nets_ema.generator(x_real, s_trg, labels=y_trg, masks=masks)

        nets_ema.generator.module.track_running_stats = False
        # nets_ema.generator.module.set_std_weight(2)
        self.nets_ema.generator.module.update_stats()
        self._save_checkpoint(step=-resume_iter)
        calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')


def compute_mae_d_loss(nets, mask_token, args, x_real, y_org, z_trg=None, masks=None):
    # with real images
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    # with fake images
    with torch.no_grad():
        x_fake, mae_masks, _ = repair_mask(nets, mask_token, args, x_real, y_org, z_trg=z_trg, masks=masks)
    out = nets.discriminator(x_fake, y_org)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())


def compute_mae_g_loss(nets, mask_token, args, x_reals, y_org, z_trgs=None, masks=None):
    x_real, x_real2 = x_reals
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    else:
        z_trg = z_trg2 = None

    x_fake, mae_masks, s_trg = repair_mask(nets, mask_token, args, x_real, y_org, z_trg=z_trg, masks=masks)
    out = nets.discriminator(x_fake, y_org)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = get_style_code(nets, args.norm_type, 1, y_org, x_fake, z_trg=z_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # reconstruction loss
    loss_rec = torch.mean(torch.abs(x_fake - x_real))

    s_real = get_style_code(nets, args.norm_type, 1, y_org, x_real, z_trg=z_trg)
    s_real2 = get_style_code(nets, args.norm_type, 1, y_org, x_real2, z_trg=z_trg2)
    if args.norm_type == 'adain':
        s_real2.detach()
        loss_ds = torch.mean(torch.abs(s_real - s_real2))
    else:
        loss_ds = torch.zeros([], requires_grad=False)
    loss = loss_adv + args.lambda_sty * loss_sty + args.lambda_rec * loss_rec + args.lambda_ds * loss_ds
    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       rec=loss_rec.item(),
                       ds=loss_ds.item())
    # loss = loss_adv + args.lambda_rec * loss_rec
    # return loss, Munch(adv=loss_adv.item(),
    #                    rec=loss_rec.item())


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.requires_grad_()
    # TODO testing
    x_real = DiffAugment(x_real, policy=args.DiffAugment)
    # TODO testing
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    # with fake images
    with torch.no_grad():
        s_trg = get_style_code(nets, args.norm_type, 1, y_trg, x_ref, z_trg)
        x_fake = nets.generator(x_real, s_trg, labels=y_trg, masks=masks)
    # TODO testing
    x_fake = DiffAugment(x_fake, policy=args.DiffAugment)
    # TODO testing
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())


def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    else:
        z_trg = z_trg2 = None
    if x_refs is not None:
        x_ref, x_ref2 = x_refs
    else:
        x_ref = x_ref2 = None

    # adversarial loss
    s_trg = get_style_code(nets, args.norm_type, args.num_embeds, y_trg, x_ref, z_trg)

    x_fake = nets.generator(x_real, s_trg, labels=y_trg, masks=masks)
    # TODO testing
    x_fake = DiffAugment(x_fake, policy=args.DiffAugment)
    # TODO testing
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = get_style_code(nets, args.norm_type, args.num_embeds, y_trg, x_fake, z_trg=None)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    s_trg2 = get_style_code(nets, args.norm_type, args.num_embeds, y_trg, x_ref2, z_trg=z_trg2)
    x_fake2 = nets.generator(x_real, s_trg2, labels=y_trg, masks=masks)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))
    # if args.norm_type == 'adain':
    #     if z_trgs is not None:
    #         s_trg2 = nets.mapping_network(z_trg2, y_trg)
    #     else:
    #         s_trg2 = nets.style_encoder(x_ref2, y_trg)
    #     x_fake2 = nets.generator(x_real, s_trg2, labels=y_trg, masks=masks)
    #     x_fake2 = x_fake2.detach()
    #     loss_ds = torch.mean(torch.abs(x_fake - x_fake2))
    # else:
    #     loss_ds = torch.zeros([], requires_grad=False)

    # cycle-consistency loss
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    s_org = get_style_code(nets, args.norm_type, args.num_embeds, y_org, x_real, z_trg=None)
    x_rec = nets.generator(x_fake, s_org, labels=y_org, masks=masks)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    loss = loss_adv + args.lambda_sty * loss_sty \
           - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc
    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item(),
                       cyc=loss_cyc.item())


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def moving_average_for_sean_blocks(model, model_test, beta=0.999):
    for attr_value, attr_value_test in zip(model.modules(), model_test.modules()):
        if isinstance(attr_value, SEAN) and isinstance(attr_value_test, SEAN):
            for l in attr_value.embeds.keys():
                mean = getattr(attr_value, f'mean_{l}')
                mean_test = getattr(attr_value_test, f'mean_{l}')
                std = getattr(attr_value, f'std_{l}')
                std_test = getattr(attr_value_test, f'std_{l}')
                setattr(attr_value_test, f'mean_{l}', torch.lerp(mean, mean_test, beta))
                setattr(attr_value_test, f'std_{l}', torch.lerp(std, std_test, beta))


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg
