"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import copy
import itertools
import math
import random

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTForImageClassification
from core.wing import FAN
from torchvision import transforms


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s, labels=None):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.register_buffer('filter',
                             torch.tensor([[-1, -1, -1],
                                           [-1, 8., -1],
                                           [-1, -1, -1]]) / w_hpf)

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))


# class SEAN(nn.Module):
#     def __init__(self, embed_nc, norm_nc, label_nc, hidden_nc=128):
#         super().__init__()
#
#         self.label_nc = label_nc
#
#         self.hidden_nc = hidden_nc
#         # self.hidden_nc = norm_nc
#         # hidden_nc = norm_nc
#
#         self.alpha = 1.
#         self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
#         # The dimension of the intermediate embedding space. Yes, hardcoded.
#
#         self.mlp_shared = nn.Sequential(nn.Linear(embed_nc, hidden_nc), nn.ReLU(inplace=True))
#         self.mlp_gamma = nn.Linear(hidden_nc, norm_nc)
#         self.mlp_beta = nn.Linear(hidden_nc, norm_nc)
#         self.label_embedding = nn.Embedding(label_nc, hidden_nc)
#
#         # self.means = nn.ParameterDict()
#         self.create_stats()
#         self.inference_running_stats = False
#         self.track_running_stats = False
#         self.num_embeds_tracked = 10_000
#         self.std_weight = 1
#         self.mix_alpha = None
#
#     def _mix_embeddings(self, embeds):
#         if self.mix_alpha is None:
#             if embeds.dim() == 3:
#                 embeds = embeds.mean(dim=1)
#         else:
#             if embeds.dim() == 3:
#                 assert self.mix_alpha.size(0) == embeds.size(0) and self.mix_alpha.size(1) == embeds.size(1), \
#                     f'embeds size {embeds.size()} and mix_alpha size {self.mix_alpha.size()} do not match'
#                 for i in range(len(self.mix_alpha)):
#                     self.mix_alpha[i] = self.mix_alpha[i] / self.mix_alpha[i].sum()
#                 embeds = torch.sum(embeds * self.mix_alpha, dim=1)
#         return embeds
#
#     def create_stats(self):
#         labels = torch.eye(self.label_nc, dtype=int)
#         for l in labels:
#             self.register_buffer(f'mean_{torch.argmax(l)}', torch.zeros(self.hidden_nc))
#             self.register_buffer(f'std_{torch.argmax(l)}', torch.zeros(self.hidden_nc))
#         self.embeds = {torch.argmax(l).item(): [] for l in labels}
#
#     def update_stats(self):
#         eps = 1e-5
#         for l in self.embeds.keys():
#             if self.embeds[l]:
#                 mean = getattr(self, f'mean_{l}')
#                 std = getattr(self, f'std_{l}')
#                 # for i in range(len(self.embeds[l])):
#                 #     self.embeds[l][i] = self.embeds[l][i].to('cuda:0')
#                 feat = torch.stack(self.embeds[l], dim=0)
#                 _, C = feat.size()
#                 correction = int(len(self.embeds[l]) > 1)
#                 feat_var = feat.var(dim=0, correction=correction) + eps
#                 feat_std = feat_var.sqrt()
#                 feat_mean = feat.mean(dim=0)
#                 mean[:], std[:] = feat_mean, feat_std
#                 self.embeds[l] = self.embeds[l][-self.num_embeds_tracked:]
#
#     def forward(self, x, labels, feat=None):
#         N, C = x.size()[:2]
#         # Part 1. generate parameter-free normalized activations
#         normalized = self.param_free_norm(x)
#
#         # Part 2. produce scaling and bias conditioned on semantic map
#         if self.inference_running_stats:
#             mix_feat = torch.empty(N, self.hidden_nc).to(x.device)
#             # feat = torch.randn(x.size(0), self.hidden_nc).to(x.device)
#             for i, (label, noise_vector) in enumerate(zip(labels, feat)):
#                 mean = getattr(self, f'mean_{label.item()}')
#                 std = getattr(self, f'std_{label.item()}')
#                 mix_feat[i] = noise_vector * std * self.std_weight + mean
#         else:
#             enc_feat = self.mlp_shared(feat)
#             latent_code = self.label_embedding(labels)
#             mix_feat = enc_feat + latent_code.view(latent_code.size(0), 1, -1)
#             # mix_feat = enc_feat
#             mix_feat = self._mix_embeddings(mix_feat)
#             if self.track_running_stats:
#                 for label, single_feat in zip(labels, mix_feat.clone().detach()):
#                     if mix_feat.dim() == 3:
#                         for slice_feat in single_feat:
#                             self.embeds[label.item()].append(slice_feat)
#                     else:
#                         self.embeds[label.item()].append(single_feat)
#         mix_feat = mix_feat.view(-1, self.hidden_nc)
#         gamma = self.mlp_gamma(mix_feat).view(-1, C, 1, 1)
#         beta = self.mlp_beta(mix_feat).view(-1, C, 1, 1)
#
#         # apply scale and bias
#         out = normalized * (1 + gamma) + beta
#
#         return out


class SEAN(nn.Module):
    def __init__(self, embed_nc, norm_nc, label_nc, hidden_nc=128):
        super().__init__()

        self.label_nc = label_nc

        self.hidden_nc = hidden_nc
        self.reduce_rate = 1
        self.norm_nc = norm_nc
        # self.hidden_nc = norm_nc
        # hidden_nc = norm_nc

        self.alpha = 1.
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.

        self.mlp_shared = nn.Sequential(nn.Linear(hidden_nc, norm_nc // self.reduce_rate), nn.ReLU(inplace=True))
        self.mlp_gamma = nn.Linear(norm_nc // self.reduce_rate, norm_nc)
        self.mlp_beta = nn.Linear(norm_nc // self.reduce_rate, norm_nc)
        self.label_embedding = nn.Embedding(label_nc, norm_nc // self.reduce_rate)

    def forward(self, x, labels, feat=None):
        N, C = x.size()[:2]
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        enc_feat = self.mlp_shared(feat).view(-1, self.norm_nc // self.reduce_rate)
        latent_code = self.label_embedding(labels).view(-1, self.norm_nc // self.reduce_rate)
        mix_feat = enc_feat + latent_code
        gamma = self.mlp_gamma(mix_feat).view(-1, C, 1, 1)
        beta = self.mlp_beta(mix_feat).view(-1, C, 1, 1)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SEANResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, embed_nc, label_nc, hidden_nc, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, embed_nc, label_nc, hidden_nc)

    def _build_weights(self, dim_in, dim_out, embed_nc, label_nc, hidden_nc):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = SEAN(embed_nc, dim_in, label_nc, hidden_nc)
        self.norm2 = SEAN(embed_nc, dim_out, label_nc, hidden_nc)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s, labels):
        x = self.norm1(x, labels, feat=s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, labels, feat=s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s, labels):
        out = self._residual(x, s, labels)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1, norm_type='adain',
                 embed_nc=768, label_nc=3, hidden_nc=256):
        super().__init__()
        dim_in = 2 ** 14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))

            if norm_type == 'adain':
                self.decode.insert(
                    0, AdainResBlk(dim_out, dim_in, style_dim,
                                   w_hpf=w_hpf, upsample=True))  # stack-like
            elif norm_type == 'sean':
                self.decode.insert(
                    0, SEANResBlk(dim_out, dim_in, embed_nc, label_nc, hidden_nc,
                                  w_hpf=w_hpf, upsample=True))  # stack-like
            else:
                raise NotImplementedError('norm type [%s] is not found' % norm_type)
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            if norm_type == 'adain':
                self.decode.insert(
                    0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))
            elif norm_type == 'sean':
                self.decode.insert(0, SEANResBlk(dim_out, dim_out, embed_nc, label_nc, hidden_nc, w_hpf=w_hpf))
            else:
                raise NotImplementedError('norm type [%s] is not found' % norm_type)

        if w_hpf > 0:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)
        self.decoder_len = len(self.decode)

    def forward(self, x, s, masks=None, labels=None, layer_split_index=None):
        x = self.from_rgb(x)
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x)
        for idx, block in enumerate(self.decode):
            if layer_split_index is not None:
                if idx in layer_split_index:
                    x = block(x, s[:, 1].unsqueeze(1), labels)
                else:
                    x = block(x, s[:, 0].unsqueeze(1), labels)
            else:
                x = block(x, s, labels)
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                x = x + self.hpf(mask * cache[x.size(2)])
        return self.to_rgb(x)

    def update_stats(self):
        for attr_name, attr_value in self.named_modules():
            if isinstance(attr_value, SEAN):
                attr_value.update_stats()

    def set_std_weight(self, std_weight):
        for attr_name, attr_value in self.named_modules():
            if isinstance(attr_value, SEAN):
                attr_value.std_weight = std_weight

    @property
    def track_running_stats(self):
        for attr_name, attr_value in self.named_modules():
            if isinstance(attr_value, SEAN):
                return attr_value.track_running_stats

    @track_running_stats.setter
    def track_running_stats(self, track_running_stats):
        for attr_name, attr_value in self.named_modules():
            if isinstance(attr_value, SEAN):
                attr_value.track_running_stats = track_running_stats

    @property
    def inference_running_stats(self):
        for attr_name, attr_value in self.named_modules():
            if isinstance(attr_value, SEAN):
                return attr_value.inference_running_stats

    @inference_running_stats.setter
    def inference_running_stats(self, inference_running_stats):
        for attr_name, attr_value in self.named_modules():
            if isinstance(attr_value, SEAN):
                attr_value.inference_running_stats = inference_running_stats

    @property
    def mix_alpha(self):
        for attr_name, attr_value in self.named_modules():
            if isinstance(attr_value, SEAN):
                return attr_value.mix_alpha

    @mix_alpha.setter
    def mix_alpha(self, mix_alpha):
        for attr_name, attr_value in self.named_modules():
            if isinstance(attr_value, SEAN):
                attr_value.mix_alpha = mix_alpha


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2 ** 14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2 ** 14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out


class FeatureExtractor(nn.Module):
    def __init__(self, num_domains, embed_nc=768, hidden_nc=256):
        super().__init__()

        self.hidden_nc = hidden_nc
        self.num_domains = num_domains
        self.extractor = ViTForImageClassification. \
            from_pretrained(f'google/vit-base-patch16-224', output_hidden_states=True).eval()
        self.extractor.requires_grad_(False)
        self.resize = transforms.Resize((224, 224), antialias=False)
        self.unshared = nn.Linear(embed_nc, hidden_nc)
        self.actv = nn.ReLU()
        self._style_codes = None
        self.track_running_stats = False
        self.inference_running_stats = False
        self._num_tracked_style_code = 10_000
        self._std_weight = 1
        self._create_stats()

    def _create_stats(self):
        labels = torch.arange(self.num_domains)
        for l in labels:
            self.register_buffer(f'mean_{l.item()}', torch.zeros(self.hidden_nc))
            self.register_buffer(f'std_{l.item()}', torch.zeros(self.hidden_nc))
        self._style_codes = {l.item(): [] for l in labels}

    def extract_features(self, x_ref, num_embeds):
        """
        :param x_ref: (N, num_embeds, C, H, W) or (N, C, H, W)
        :param num_embeds: int if want a random number from 1 to num_embeds or -int if want a fixed number of num_embeds
        :return: (N, num_embeds, embed_nc) or (N, embed_nc)
        """
        if x_ref.dim() == 5:
            if num_embeds > 0:
                num_embeds = random.randint(1, num_embeds)
            else:
                num_embeds = -num_embeds
            N = x_ref.size(0)
            x_ref = self.resize(x_ref[:, :num_embeds].reshape(-1, *(x_ref.size()[-3:])))
            inputs = {'pixel_values': x_ref}
            out_vit = self.extractor(**inputs)
            s_trg = out_vit.hidden_states[-1][:, 0, :].reshape(N, num_embeds, -1)
        elif x_ref.dim() == 4:
            x_ref = self.resize(x_ref)
            inputs = {'pixel_values': x_ref}
            out_vit = self.extractor(**inputs)
            s_trg = out_vit.hidden_states[-1][:, 0, :].unsqueeze(1)
        else:
            raise NotImplementedError(f'Wrong dimension [{x_ref.dim()}]')
        return s_trg

    def reduce_dimension(self, features):
        domain_features = self.unshared(self.actv(features))
        # (n, hidden_nc) or (n, num_embeds, hidden_nc)
        if domain_features.dim() == 3:
            domain_features = domain_features.mean(dim=1)  # (batch, hidden_nc)
        return domain_features

    def update_stats(self):
        eps = 1e-5
        for l in self._style_codes.keys():
            if self._style_codes[l]:
                mean = getattr(self, f'mean_{l}')
                std = getattr(self, f'std_{l}')
                # for i in range(len(self.embeds[l])):
                #     self.embeds[l][i] = self.embeds[l][i].to('cuda:0')
                feat = torch.stack(self._style_codes[l], dim=0)
                _, C = feat.size()
                correction = int(len(self._style_codes[l]) > 1)
                feat_var = feat.var(dim=0, correction=correction) + eps
                feat_std = feat_var.sqrt()
                feat_mean = feat.mean(dim=0)
                mean[:], std[:] = feat_mean, feat_std
                self._style_codes[l] = self._style_codes[l][-self._num_tracked_style_code:]

    def forward(self, x_ref, y_ref, num_embeds, z_trg=None):
        if self._inference_running_stats:
            style_codes = self.sample_style_codes(z_trg, y_ref)
        else:
            features = self.extract_features(x_ref, num_embeds)
            style_codes = self.reduce_dimension(features)

            # update running stats
            if self._track_running_stats:
                for y, style_code in zip(y_ref, style_codes.clone().detach().cpu()):
                    self._style_codes[y.item()] += [style_code]
        return style_codes

    def sample_style_codes(self, z_trg, y_ref):
        assert z_trg is not None, 'z_trg must be provided when inference_running_stats is True'
        assert z_trg.dim() == 2, 'z_trg must be 2D tensor'
        with torch.no_grad():
            style_codes = torch.zeros_like(z_trg)
            for i, (label, noise_vector) in enumerate(zip(y_ref, z_trg)):
                mean = getattr(self, f'mean_{label.item()}')
                std = getattr(self, f'std_{label.item()}')
                style_codes[i] = noise_vector * std * self._std_weight + mean
            return style_codes

    def set_std_weight(self, std_weight):
        self._std_weight = std_weight

    @property
    def track_running_stats(self):
        return self._track_running_stats

    @track_running_stats.setter
    def track_running_stats(self, track_running_stats):
        self._track_running_stats = track_running_stats

    @property
    def inference_running_stats(self):
        return self._inference_running_stats

    @inference_running_stats.setter
    def inference_running_stats(self, inference_running_stats):
        self._inference_running_stats = inference_running_stats


def build_model(args):
    generator = nn.DataParallel(Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf,
                                          norm_type=args.norm_type, label_nc=args.num_domains,
                                          embed_nc=args.embed_nc, hidden_nc=args.hidden_nc))
    mapping_network = nn.DataParallel(MappingNetwork(args.latent_dim, args.style_dim, args.num_domains))
    style_encoder = nn.DataParallel(StyleEncoder(args.img_size, args.style_dim, args.num_domains))
    discriminator = nn.DataParallel(Discriminator(args.img_size, args.num_domains))
    # feature_extractor = nn.DataParallel(ViTForImageClassification.from_pretrained(f'google/vit-base-patch16-224',
    #                                                                               output_hidden_states=True)).eval()
    # feature_extractor.requires_grad_(False)
    feature_extractor = nn.DataParallel(FeatureExtractor(args.num_domains, args.embed_nc, args.hidden_nc))

    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)

    if args.norm_type == 'adain':
        nets = Munch(generator=generator,
                     mapping_network=mapping_network,
                     style_encoder=style_encoder,
                     discriminator=discriminator)
        nets_ema = Munch(generator=generator_ema,
                         mapping_network=mapping_network_ema,
                         style_encoder=style_encoder_ema)
    else:
        nets = Munch(generator=generator,
                     discriminator=discriminator,
                     feature_extractor=feature_extractor)
        nets_ema = Munch(generator=generator_ema,
                         feature_extractor=feature_extractor)

    if args.w_hpf > 0:
        fan = nn.DataParallel(FAN(fname_pretrained=args.wing_path).eval())
        fan.get_heatmap = fan.module.get_heatmap
        nets.fan = fan
        nets_ema.fan = fan

    return nets, nets_ema


class MaskToken(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.mask_token_type = opt.mask_token_type
        self.mask_ratio = opt.mask_ratio
        if self.mask_token_type in ('zero', 'mean'):
            self.mask_token = 0
        elif self.mask_token_type == 'scalar':
            self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, 1, 1))
        elif self.mask_token_type == 'vector':
            self.mask_token = torch.nn.Parameter(torch.zeros(1, 3, 1, 1))
        elif self.mask_token_type == 'position':
            self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, opt.img_size, opt.img_size))
        elif self.mask_token_type == 'full':
            self.mask_token = torch.nn.Parameter(torch.zeros(1, opt.input_nc, opt.img_size, opt.img_size))
        else:
            raise ValueError('Unknown mask token type: {}'.format(self.mask_token_type))

    def forward(self, imgs, masks):
        masked_imgs = imgs * masks
        if self.mask_token_type == 'mean':
            self._calc_mean_mask_token(masked_imgs)
        return masked_imgs + self.mask_token * (1 - masks)

    def _calc_mean_mask_token(self, masked_imgs):
        img_mean = masked_imgs.mean(dim=(2, 3)) / self.mask_ratio
        self.mask_token = img_mean.reshape(*img_mean.size()[:2], 1, 1)
