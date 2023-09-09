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
from os.path import join as ospj
import json
import glob
from shutil import copyfile

from torchvision import transforms
from tqdm import tqdm
import ffmpeg

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils

from core.model import SEAN


def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x, ncol, filename):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


@torch.no_grad()
def translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.size()
    s_ref = get_style_code(nets, args.norm_type, args.num_embeds, y_ref, x_ref, z_trg=None)
    s_src = get_style_code(nets, args.norm_type, args.num_embeds, y_src, x_src, z_trg=None)

    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    x_fake = nets.generator(x_src, s_ref, labels=y_ref, masks=masks)

    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    x_rec = nets.generator(x_fake, s_src, labels=y_src, masks=masks)
    x_concat = [x_src, x_ref, x_fake, x_rec]
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)
    del x_concat


@torch.no_grad()
def translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename):
    N, C, H, W = x_src.size()
    latent_dim = z_trg_list[0].size(1)
    x_concat = [x_src]
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

    for i, y_trg in enumerate(y_trg_list):
        if args.norm_type == 'adain':
            z_many = torch.randn(10000, latent_dim).to(x_src.device)
            y_many = torch.LongTensor(10000).to(x_src.device).fill_(y_trg[0])
            s_many = nets.mapping_network(z_many, y_many)
            s_avg = torch.mean(s_many, dim=0, keepdim=True)
            s_avg = s_avg.repeat(N, 1)

            for z_trg in z_trg_list:
                s_trg = nets.mapping_network(z_trg, y_trg)
                s_trg = torch.lerp(s_avg, s_trg, psi)
                x_fake = nets.generator(x_src, s_trg, labels=y_trg, masks=masks)
                x_concat += [x_fake]
        elif args.norm_type == 'sean':
            for z_trg in z_trg_list:
                s_trg = nets.feature_extractor(None, y_trg, None, z_trg)
                x_fake = nets.generator(x_src, s_trg, labels=y_trg, masks=masks)
                x_concat += [x_fake]
        else:
            raise NotImplementedError(f"{args.norm_type} is not implemented")

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)


@torch.no_grad()
def translate_using_reference(nets, args, x_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.size()
    wb = torch.ones(1, C, H, W).to(x_src.device)
    x_src_with_wb = torch.cat([wb, x_src], dim=0)

    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

    s_ref = get_style_code(nets, args.norm_type, args.num_embeds, y_ref, x_ref, z_trg=None)
    if s_ref.dim() == 2:
        s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    elif s_ref.dim() == 3:
        s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1, 1)
    else:
        raise ValueError('Style code dimension should be 2 or 3')
    y_ref_list = y_ref.unsqueeze(1).repeat(1, N)
    # N N E
    x_concat = [x_src_with_wb]
    for i, (s_ref, y_ref) in enumerate(zip(s_ref_list, y_ref_list)):
        x_fake = nets.generator(x_src, s_ref, labels=y_ref, masks=masks)
        x_fake_with_ref = torch.cat([x_ref[i:i + 1], x_fake], dim=0)
        x_concat += [x_fake_with_ref]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N + 1, filename)
    del x_concat


@torch.no_grad()
def translate_using_multiple_reference(nets, args, x_src, x_refs, y_ref, filename):
    N, C, H, W = x_src.size()
    num_ref = 1
    if isinstance(x_refs, (tuple, list)) and x_refs[0].dim() == 4:
        x_ref = torch.stack(x_refs, dim=1)
        num_ref = len(x_refs)
    else:
        x_ref = x_refs
    wb = torch.ones(num_ref, C, H, W).to(x_src.device)
    x_src_with_wb = torch.cat([wb, x_src], dim=0)

    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

    s_ref = get_style_code(nets, args.norm_type, -num_ref, y_ref, x_ref, z_trg=None)
    if s_ref.dim() == 2:
        s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    elif s_ref.dim() == 3:
        s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1, 1)
    else:
        raise ValueError('Style code dimension should be 2 or 3')
    # N N E
    x_concat = [x_src_with_wb]
    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets.generator(x_src, s_ref, labels=y_ref, masks=masks)
        x_fake_with_ref = torch.cat([x_ref[i].reshape(-1, C, H, W), x_fake], dim=0)
        x_concat += [x_fake_with_ref]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N + num_ref, filename)
    del x_concat


@torch.no_grad()
def translate_with_alpha_control_reference(nets, args, x_src, x_refs, y_ref, filename):
    N, C, H, W = x_src.size()
    assert len(x_refs) == 2, 'Only support two reference images'
    num_ref = 2
    x_ref = torch.stack(x_refs, dim=1)

    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

    s_ref = get_style_code(nets, args.norm_type, -num_ref, y_ref, x_ref, z_trg=None)
    x_concat = [x_src, x_ref[:, 0].reshape(N, C, H, W)]
    for mix_alpha in torch.arange(0, 1.1, 0.1):
        nets.generator.module.mix_alpha = torch.tensor([1 - mix_alpha, mix_alpha]) \
            .view(1, 2, 1).repeat(N, 1, 1).to(x_src.device)
        x_fake = nets.generator(x_src, s_ref, labels=y_ref, masks=masks)
        x_concat.append(x_fake)
    nets.generator.module.mix_alpha = None
    x_concat.append(x_ref[:, 1].reshape(N, C, H, W))
    x_concat = torch.stack(x_concat, dim=1)
    save_image(x_concat.reshape(-1, C, H, W), x_concat.size(1), filename)
    del x_concat


@torch.no_grad()
def translate_with_layer_index_control(nets, args, x_src, x_refs, y_ref, filename):
    N, C, H, W = x_src.size()
    assert len(x_refs) == 2, 'Only support two reference images'
    layer_len = nets.generator.module.decoder_len

    num_ref = 2
    x_ref = torch.stack(x_refs, dim=1)

    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

    s_ref = get_style_code(nets, args.norm_type, -num_ref, y_ref, x_ref, z_trg=None)
    x_concat = [x_src, x_ref[:, 0].reshape(N, C, H, W)]
    x_concat.append(x_ref[:, 1].reshape(N, C, H, W))
    layer_split_indices = [(10, ), tuple(range(layer_len)),
                           (0, 1), (0, 1, 2),
                           (2, 3), (2, 3, 4), (3, 4),
                           (3, 4, 5), (4, 5), (5, )]
    for layer_split_index in layer_split_indices:
        x_fake = nets.generator(x_src, s_ref, labels=y_ref, masks=masks, layer_split_index=layer_split_index)
        x_concat.append(x_fake)
    x_concat = torch.stack(x_concat, dim=1)
    save_image(x_concat.reshape(-1, C, H, W), x_concat.size(1), filename)
    del x_concat


@torch.no_grad()
def translate_with_alpha_control_multiple_reference(nets, args, x_src, x_refs, y_ref, filename):
    N, C, H, W = x_src.size()
    assert len(x_refs) == 4, 'Only support four reference images'
    num_ref = 4
    x_ref = torch.stack(x_refs, dim=1)

    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

    s_ref = get_style_code(nets, args.norm_type, -num_ref, y_ref, x_ref, z_trg=None)
    num_batch = 11
    x_concat = [torch.cat((x_refs[0], x_src, torch.ones(num_batch - 3, C, H, W).to(x_src.device), x_refs[1]), dim=0)]
    x_src = x_src.repeat(num_batch, 1, 1, 1)
    s_ref = s_ref.repeat(num_batch, 1, 1)
    y_ref = y_ref.repeat(num_batch, 1)
    normalized_grid_weight = get_normalized_grid_weight(num_batch).to(x_src.device)
    for mix_weights in normalized_grid_weight:
        nets.generator.module.mix_alpha = mix_weights.view(*mix_weights.size()[:2], 1)
        x_fake = nets.generator(x_src, s_ref, labels=y_ref, masks=masks)
        x_concat.append(x_fake)
    nets.generator.module.mix_alpha = None

    x_concat.append(torch.cat([x_refs[2], torch.ones(num_batch - 2, C, H, W).to(x_src.device), x_refs[3]], dim=0))
    x_concat = torch.stack(x_concat, dim=1)
    save_image(x_concat.reshape(-1, C, H, W), x_concat.size(1), filename)
    del x_concat


@torch.no_grad()
def debug_mask_image(nets, args, mask_token, inputs, step, is_ema=False):
    x_real, y_org = inputs.x_src, inputs.y_src
    z_trg = inputs.z_trg
    N, C, H, W = x_real.size()

    if is_ema:
        filename_reference = ospj(args.sample_dir, '%06d_reference_ema.jpg' % (step))
        filename_latent = ospj(args.sample_dir, '%06d_latent_ema.jpg' % (step))
    else:
        filename_reference = ospj(args.sample_dir, '%06d_reference.jpg' % (step))
        filename_latent = ospj(args.sample_dir, '%06d_latent.jpg' % (step))

    if args.norm_type == 'adain':
        x_fake_latent, mae_masks_latent, _ = repair_mask(nets, mask_token, args, x_real, y_org, z_trg=z_trg, masks=None)
        x_concat_latent = torch.stack([x_real, x_fake_latent, x_real * mae_masks_latent], dim=1)
        save_image(x_concat_latent.reshape(-1, C, H, W), x_concat_latent.size(1), filename_latent)

    x_fake, mae_masks, _ = repair_mask(nets, mask_token, args, x_real, y_org, z_trg=None, masks=None)
    x_concat = torch.stack([x_real, x_fake, x_real * mae_masks], dim=1)
    save_image(x_concat.reshape(-1, C, H, W), x_concat.size(1), filename_reference)


@torch.no_grad()
def debug_image(nets, args, inputs, step):
    x_src, y_src = inputs.x_src, inputs.y_src
    x_ref, x_ref2, y_ref = inputs.x_ref, inputs.x_ref2, inputs.y_ref
    device = inputs.x_src.device
    N = inputs.x_src.size(0)

    # filename = ospj(args.sample_dir, '%06d_layer_mix.jpg' % (step))
    # translate_with_layer_index_control(nets, args, x_src, [x_ref, x_ref2], y_ref, filename)
    # exit()

    # translate and reconstruct (reference-guided)
    filename = ospj(args.sample_dir, '%06d_cycle_consistency.jpg' % (step))
    translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename)

    y_trg_list = [torch.tensor(y).repeat(N).to(device)
                  for y in range(min(args.num_domains, 5))]
    # TODO temp remove
    if args.norm_type == 'adain':
        # latent-guided image synthesis
        z_trg_list = torch.randn(args.num_outs_per_domain, 1, args.latent_dim).repeat(1, N, 1).to(device)
        for psi in [0.5, 0.7, 1.0]:
            filename = ospj(args.sample_dir, '%06d_latent_psi_%.1f.jpg' % (step, psi))
            translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename)
    elif args.norm_type == 'sean':
        z_trg_list = torch.randn(args.num_outs_per_domain, 1, args.hidden_nc).repeat(1, N, 1).to(device)
        for weight in [1, 2, 3, 4]:
            filename = ospj(args.sample_dir, '%06d_latent_std_%.1f.jpg' % (step, weight))
            nets.feature_extractor.module.set_std_weight(weight)
            nets.feature_extractor.module.inference_running_stats = True
            translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, weight, filename)
            nets.feature_extractor.module.inference_running_stats = False
        nets.feature_extractor.module.set_std_weight(1)
    else:
        raise NotImplementedError(f"{args.norm_type} is not implemented")

    # reference-guided image synthesis
    filename = ospj(args.sample_dir, '%06d_reference.jpg' % (step))
    translate_using_reference(nets, args, x_src, x_ref, y_ref, filename)
    # if args.norm_type == 'sean':
    #     # filename = ospj(args.sample_dir, '%06d_reference.jpg' % (step))
    #     # translate_using_multiple_reference(nets, args, x_src, [x_ref, x_ref2], y_ref, filename)
    #     if args.num_val_refs >= 2:
    #         filename = ospj(args.sample_dir, '%06d_reference_mix.jpg' % (step))
    #         translate_with_alpha_control_reference(nets, args, x_src, [x_ref, x_ref2], y_ref, filename)
    #         filename = ospj(args.sample_dir, '%06d_layer_mix.jpg' % (step))
    #         translate_with_layer_index_control(nets, args, x_src, [x_ref, x_ref2], y_ref, filename)
    #     if args.num_val_refs >= 4:
    #         x_ref3, x_ref4 = inputs.x_ref3, inputs.x_ref4
    #         for i in range(args.num_domains):
    #             filename = ospj(args.sample_dir, f'{step:06d}_reference_four_mix_{i:1d}.jpg')
    #             indices = ((y_ref == i).nonzero(as_tuple=True)[0])
    #             if len(indices):
    #                 index = indices[torch.randint(0, len(indices), (1,))]
    #                 translate_with_alpha_control_multiple_reference(nets, args, x_src[index: index + 1],
    #                                                                 [x_ref[index: index + 1], x_ref2[index: index + 1],
    #                                                                  x_ref3[index: index + 1],
    #                                                                  x_ref4[index: index + 1]],
    #                                                                 y_ref[index: index + 1], filename)


# ======================= #
# Video-related functions #
# ======================= #


def sigmoid(x, w=1):
    return 1. / (1 + np.exp(-w * x))


def get_alphas(start=-5, end=5, step=0.5, len_tail=10):
    return [0] + [sigmoid(alpha) for alpha in np.arange(start, end, step)] + [1] * len_tail


def interpolate(nets, args, x_src, s_prev, s_next, y_next):
    ''' returns T x C x H x W '''
    B = x_src.size(0)
    frames = []
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    alphas = get_alphas()

    s_ref = torch.cat((s_prev, s_next), dim=1)
    for alpha in alphas:
        mix_alpha = torch.tensor([1 - alpha, alpha]).view(*(s_ref.size()[:2]), 1).to(x_src.device, dtype=torch.float32)
        nets.generator.module.mix_alpha = mix_alpha
        x_fake = nets.generator(x_src, s_ref, labels=y_next, masks=masks)
        entries = torch.cat([x_src.cpu(), x_fake.cpu()], dim=2)
        frame = torchvision.utils.make_grid(entries, nrow=B, padding=0, pad_value=-1).unsqueeze(0)
        frames.append(frame)
    frames = torch.cat(frames)
    return frames


def slide(entries, margin=32):
    """Returns a sliding reference window.
    Args:
        entries: a list containing two reference images, x_prev and x_next, 
                 both of which has a shape (1, 3, 256, 256)
    Returns:
        canvas: output slide of shape (num_frames, 3, 256*2, 256+margin)
    """
    _, C, H, W = entries[0].shape
    alphas = get_alphas()
    T = len(alphas)  # number of frames

    canvas = - torch.ones((T, C, H * 2, W + margin))
    merged = torch.cat(entries, dim=2)  # (1, 3, 512, 256)
    for t, alpha in enumerate(alphas):
        top = int(H * (1 - alpha))  # top, bottom for canvas
        bottom = H * 2
        m_top = 0  # top, bottom for merged
        m_bottom = 2 * H - top
        canvas[t, :, top:bottom, :W] = merged[:, :, m_top:m_bottom, :]
    return canvas


@torch.no_grad()
def video_ref(nets, args, x_src, x_ref, y_ref, fname):
    video = []
    s_ref = get_style_code(nets, args.norm_type, 1, y_ref, x_ref, z_trg=None)
    s_prev = None
    for data_next in tqdm(zip(x_ref, y_ref, s_ref), 'video_ref', len(x_ref)):
        x_next, y_next, s_next = [d.unsqueeze(0) for d in data_next]
        if s_prev is None:
            x_prev, y_prev, s_prev = x_next, y_next, s_next
            continue
        if y_prev != y_next:
            x_prev, y_prev, s_prev = x_next, y_next, s_next
            continue
        interpolated = interpolate(nets, args, x_src, s_prev, s_next, y_next)
        entries = [x_prev, x_next]
        slided = slide(entries)  # (T, C, 256*2, 256)
        frames = torch.cat([slided, interpolated], dim=3).cpu()  # (T, C, 256*2, 256*(batch+1))
        video.append(frames)
        x_prev, y_prev, s_prev = x_next, y_next, s_next

    # append last frame 10 time
    for _ in range(10):
        video.append(frames[-1:])
    video = tensor2ndarray255(torch.cat(video))
    save_video(fname, video)


@torch.no_grad()
def video_latent(nets, args, x_src, y_list, z_list, psi, fname):
    latent_dim = z_list[0].size(1)
    s_list = []
    for i, y_trg in enumerate(y_list):
        z_many = torch.randn(10000, latent_dim).to(x_src.device)
        y_many = torch.LongTensor(10000).to(x_src.device).fill_(y_trg[0])
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(x_src.size(0), 1)

        for z_trg in z_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_trg = torch.lerp(s_avg, s_trg, psi)
            s_list.append(s_trg)

    s_prev = None
    video = []
    # fetch reference images
    for idx_ref, (s_next, y_next) in enumerate(tqdm(zip(s_list, y_list), 'video_latent', len(s_list))):
        if s_prev is None:
            s_prev = s_next
            continue
        if idx_ref % len(z_list) == 0:
            s_prev = s_next
            continue
        frames = interpolate(nets, args, x_src, s_prev, s_next, y_next).cpu()
        video.append(frames)
        s_prev = s_next
    for _ in range(10):
        video.append(frames[-1:])
    video = tensor2ndarray255(torch.cat(video))
    save_video(fname, video)


def save_video(fname, images, output_fps=30, vcodec='libx264', filters=''):
    assert isinstance(images, np.ndarray), "images should be np.array: NHWC"
    num_frames, height, width, channels = images.shape
    stream = ffmpeg.input('pipe:', format='rawvideo',
                          pix_fmt='rgb24', s='{}x{}'.format(width, height))
    stream = ffmpeg.filter(stream, 'setpts', '2*PTS')  # 2*PTS is for slower playback
    stream = ffmpeg.output(stream, filename=fname, pix_fmt='yuv420p', vcodec=vcodec, r=output_fps)
    stream = ffmpeg.overwrite_output(stream)
    process = ffmpeg.run_async(stream, pipe_stdin=True)
    for frame in tqdm(images, desc='writing video to %s' % fname):
        process.stdin.write(frame.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()


def tensor2ndarray255(images):
    images = torch.clamp(images * 0.5 + 0.5, 0, 1)
    return images.cpu().numpy().transpose(0, 2, 3, 1) * 255


def label_to_str(label):
    label_str = [str(idx) for idx, label_value in enumerate(label) if label_value == 1]
    return '-'.join(label_str)


def generate_multilabel_combinations(label_dim):
    binary_values = torch.tensor([0, 1])
    all_combinations = torch.cartesian_prod(*([binary_values] * label_dim))
    return all_combinations


def get_style_code(nets, norm_type, num_embeds, y_trg, x_ref, z_trg):
    if norm_type == 'adain':
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:
            s_trg = nets.style_encoder(x_ref, y_trg)
    elif norm_type == 'sean':
        if z_trg is not None:
            s_trg = z_trg
        else:  # x_ref is not None
            s_trg = nets.feature_extractor(x_ref, y_trg, num_embeds)
            # resize = transforms.Resize((224, 224), antialias=False)
            # if x_ref.dim() == 5:
            #     if num_embeds > 0:
            #         num_embeds = random.randint(1, num_embeds)
            #     else:
            #         num_embeds = -num_embeds
            #     N = x_ref.size(0)
            #     x_ref = resize(x_ref[:, :num_embeds].reshape(-1, *(x_ref.size()[-3:])))
            #     inputs = {'pixel_values': x_ref}
            #     out_vit = nets.feature_extractor(**inputs)
            #     s_trg = out_vit.hidden_states[-1][:, 0, :].reshape(N, num_embeds, -1)
            # elif x_ref.dim() == 4:
            #     x_ref = resize(x_ref)
            #     inputs = {'pixel_values': x_ref}
            #     out_vit = nets.feature_extractor(**inputs)
            #     s_trg = out_vit.hidden_states[-1][:, 0, :].unsqueeze(1)
            # else:
            #     raise NotImplementedError(f'Wrong dimension [{x_ref.dim()}]')
    else:
        raise NotImplementedError(f'Normalization type [{norm_type}] is not found.')
    return s_trg


def get_normalized_grid_weight(num_batch):
    # grid size
    grid_width = num_batch
    grid_height = num_batch

    # corner vectors
    top_left = torch.tensor([1, 0, 0, 0])
    top_right = torch.tensor([0, 0, 1, 0])
    bottom_left = torch.tensor([0, 1, 0, 0])
    bottom_right = torch.tensor([0, 0, 0, 1])

    # create grid of weights for interpolation
    grid_x, grid_y = torch.meshgrid(torch.linspace(0, 1, grid_width), torch.linspace(0, 1, grid_height))

    # interpolate the vectors
    interpolated_grid = (
            (1 - grid_x)[:, :, None] * (1 - grid_y)[:, :, None] * top_left[None, None, :]
            + (1 - grid_x)[:, :, None] * grid_y[:, :, None] * bottom_left[None, None, :]
            + grid_x[:, :, None] * (1 - grid_y)[:, :, None] * top_right[None, None, :]
            + grid_x[:, :, None] * grid_y[:, :, None] * bottom_right[None, None, :]
    )

    # normalize the vectors
    normalized_grid = interpolated_grid / interpolated_grid.sum(dim=-1, keepdim=True)
    return normalized_grid


def check_values(dict, threshold=10000):
    for value in dict.values():
        if value <= threshold:
            return False
    return True


def generate_mask(image_size, patch_size, mask_ratio):
    """
        input image_size: (b, c, h, w)
        output size: (b, 1, h, w)
    """
    height_size = image_size[2] // patch_size
    width_size = image_size[3] // patch_size
    mask = torch.bernoulli(torch.full((image_size[0], 1, height_size, width_size), (1 - mask_ratio)))

    return F.interpolate(mask, scale_factor=patch_size, mode='nearest')


def generate_shifted_mask(image_size, patch_size, mask_ratio):
    """
        input image_size: (b, c, h, w)
        output size: (b, 1, h, w)
    """
    img_h, img_w = image_size[2], image_size[3]
    h_shift = torch.randint(low=0, high=patch_size, size=(1,))
    w_shift = torch.randint(low=0, high=patch_size, size=(1,))
    extend_image_size = (*image_size[:2], img_h + patch_size, img_w + patch_size)
    extend_mask = generate_mask(extend_image_size, patch_size, mask_ratio)
    mask = extend_mask[:, :, h_shift: h_shift + img_h, w_shift: w_shift + img_w]
    return mask


def repair_mask(nets, mask_token, args, x_real, y_org, z_trg=None, masks=None):
    s_trg = get_style_code(nets, args.norm_type, 1, y_org, x_real, z_trg)
    mae_masks = generate_shifted_mask(x_real.size(), args.patch_size, args.mask_ratio).to(x_real.device)
    masked_imgs = mask_token(x_real, mae_masks)

    x_fake = nets.generator(masked_imgs, s_trg, labels=y_org, masks=masks)
    return x_fake, mae_masks, s_trg
