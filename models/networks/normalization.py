import torch
from torch import nn
import torch.nn.functional as F
import math
from utils.util import calc_mean_std, calc_embed_mean_std, calc_kl_with_logits, generate_multilabel_combinations, \
    label_to_str
from collections import defaultdict


class SPADE(nn.Module):
    def __init__(self, label_nc, norm_nc, hidden_nc=128, kernel_size=(3, 3), padding='same', norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.param_free_norm = norm_layer(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.

        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, hidden_nc,
                                                  kernel_size=kernel_size,
                                                  padding=padding),
                                        nn.ReLU(inplace=True))
        self.mlp_gamma = nn.Conv2d(hidden_nc, norm_nc, kernel_size=kernel_size, padding=padding)
        self.mlp_beta = nn.Conv2d(hidden_nc, norm_nc, kernel_size=kernel_size, padding=padding)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class AdaIN(nn.Module):
    def __init__(self, norm_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, denorm_type='linear'):
        super().__init__()

        self.denorm_type = denorm_type

        # The dimension of the intermediate embedding space. Yes, hardcoded.

        self.param_free_norm = norm_layer(norm_nc, affine=False)
        if self.denorm_type == 'linear':
            assert hidden_nc is not None, 'hidden_nc should be set when denorm_type is linear.'
            self.hidden_nc = hidden_nc
            self.mlp_gamma = nn.Linear(hidden_nc, norm_nc)
            self.mlp_beta = nn.Linear(hidden_nc, norm_nc)

    def forward(self, x, style_feat):
        N, C = x.size()[:2]
        assert style_feat.size(1) == self.hidden_nc, 'The channel of style feature is not equal to hidden_nc.'
        style_feat = style_feat.view(N, self.hidden_nc)

        normalized = self.param_free_norm(x)

        if self.denorm_type == 'linear':
            gamma = self.mlp_gamma(style_feat).view(N, C, 1, 1)
            beta = self.mlp_beta(style_feat).view(N, C, 1, 1)
        elif self.denorm_type == 'stat':
            beta, gamma = calc_mean_std(style_feat)
            gamma -= 1
        else:
            raise NotImplementedError('Not implemented denorm type: {}'.format(self.denorm_type))
        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SEAN(nn.Module):
    def __init__(self, embed_nc, norm_nc, label_nc, hidden_nc=128,
                 latent_dim=16, norm_layer=nn.BatchNorm2d, style_distill=False):
        super().__init__()

        self.style_distill = style_distill
        if self.style_distill:
            self._distill_loss = None
        self.latent_dim = latent_dim
        self.noise_dim = self.latent_dim - label_nc
        self.label_nc = label_nc
        self.hidden_nc = hidden_nc
        self.alpha = 1.
        self.param_free_norm = norm_layer(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.

        self.mlp_shared = nn.Sequential(nn.Linear(embed_nc, hidden_nc), nn.ReLU(inplace=True))
        self.mlp_gamma = nn.Linear(hidden_nc, norm_nc)
        self.mlp_beta = nn.Linear(hidden_nc, norm_nc)

        self.mlp_latent = nn.Sequential(nn.Linear(self.latent_dim, hidden_nc),
                                        nn.ReLU(inplace=True))
        # self.mlp_latent = nn.Sequential(nn.Linear(self.latent_dim, hidden_nc // 4),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Linear(hidden_nc // 4, hidden_nc // 2),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Linear(hidden_nc // 2, hidden_nc),
        #                                 nn.ReLU(inplace=True))
        # self.mlp_latent = nn.Sequential(nn.Linear(self.latent_dim, hidden_nc // 2),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Linear(hidden_nc // 2, hidden_nc),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Linear(hidden_nc, hidden_nc),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Linear(hidden_nc, hidden_nc),
        #                                 nn.ReLU(inplace=True))
        # self.means = nn.ParameterDict()
        self.create_stats()
        self.inference_running_stats = False
        self.track_running_stats = False
        self.num_embeds_tracked = 10000

    def create_stats(self):
        labels = generate_multilabel_combinations(self.label_nc).tolist()
        for l in labels:
            self.register_buffer('mean_' + label_to_str(l), torch.zeros(self.hidden_nc))
            self.register_buffer('std_' + label_to_str(l), torch.zeros(self.hidden_nc))
        self.embeds = {tuple(l): [] for l in labels}

    def update_stats(self):
        eps = 1e-5
        for l in self.embeds.keys():
            if self.embeds[l]:
                mean = getattr(self, 'mean_' + label_to_str(l))
                std = getattr(self, 'std_' + label_to_str(l))
                feat = torch.stack(self.embeds[l], dim=0)
                _, C = feat.size()
                feat_var = feat.var(dim=0) + eps
                feat_std = feat_var.sqrt()
                feat_mean = feat.mean(dim=0)
                mean[:], std[:] = feat_mean, feat_std
                self.embeds[l] = self.embeds[l][-self.num_embeds_tracked:]

    @property
    def distill_loss(self):
        return self._distill_loss

    @distill_loss.setter
    def distill_loss(self, enable_distill_loss):
        """
            set the distillation loss to None to disable distillation
            set the distillation loss to the empty dict to enable distillation
        """
        if enable_distill_loss:
            self._distill_loss = defaultdict(list)
        else:
            self._distill_loss = None

    def set_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, x, labels, feat=None):
        N, C = x.size()[:2]
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # use noise and labels to produce latent code
        noise = torch.randn(N, self.noise_dim).to(x.device)
        if labels.dim() == 4:
            labels = labels.view(N, -1)
        latent = torch.cat([noise, labels], dim=1)
        latent_code = self.mlp_latent(latent)
        # latent_code = self.mlp_latent(labels)

        # Part 2. produce scaling and bias conditioned on semantic map
        if feat is None:
            mix_feat = latent_code
        elif self.inference_running_stats:
            mix_feat = torch.empty(N, self.hidden_nc).to(x.device)
            for i, (label, noise_vector) in enumerate(zip(labels, feat)):
                tuple_label = tuple(map(lambda x: int(x.item()), label))
                mean = getattr(self, 'mean_' + label_to_str(tuple_label))
                std = getattr(self, 'std_' + label_to_str(tuple_label))
                mix_feat[i] = noise_vector * std * 1 + mean
        else:
            enc_feat = self.mlp_shared(feat)
            if self.track_running_stats:
                for i, (label, single_feat) in enumerate(zip(labels, enc_feat.clone().detach())):
                    tuple_label = tuple(map(lambda x: int(x.item()), label))
                    if enc_feat.dim() == 3:
                        for slice_feat in single_feat:
                            self.embeds[tuple_label].append(slice_feat)
                    else:
                        self.embeds[tuple_label].append(single_feat)
            if enc_feat.dim() == 3:
                enc_feat = enc_feat.mean(dim=1)
            mix_feat = enc_feat * self.alpha + latent_code * (1 - self.alpha)
            # mix_feat = enc_feat

            # replace style embed with latent code if style embed is all zeros
            mask_indices = (enc_feat == 0).all(dim=1).view(-1, 1)
            mix_feat = mix_feat * ~mask_indices + latent_code * mask_indices

            if self.style_distill and self._distill_loss is not None:
                t = 4
                mix_labels = mix_feat.detach()
                distill_latent_loss = calc_kl_with_logits(latent_code, mix_labels, t)
                distill_embed_loss = calc_kl_with_logits(enc_feat, mix_labels, t)
                distill_loss = distill_latent_loss * 0.1 + distill_embed_loss
                distill_loss.backward(retain_graph=True)
                self._distill_loss['latent'].append(distill_latent_loss)
                self._distill_loss['embed'].append(distill_embed_loss)

        gamma = self.mlp_gamma(mix_feat).view(N, C, 1, 1)
        beta = self.mlp_beta(mix_feat).view(N, C, 1, 1)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

# class SEAN(nn.Module):
#     def __init__(self, embed_nc, norm_nc, label_nc, hidden_nc=128,
#                  latent_dim=16, norm_layer=nn.BatchNorm2d, style_distill=False):
#         super().__init__()
#
#         self.latent_dim = latent_dim
#         self.noise_dim = self.latent_dim - label_nc
#         self.alpha = 1.
#         self.param_free_norm = norm_layer(norm_nc, affine=False)
#         # The dimension of the intermediate embedding space. Yes, hardcoded.
#
#         self.mlp_shared = nn.Sequential(nn.Linear(embed_nc, norm_nc), nn.ReLU(inplace=True))
#
#         self.mlp_latent = nn.Sequential(nn.Linear(self.latent_dim, norm_nc),
#                                         nn.ReLU(inplace=True))
#         self.eps = 1e-5
#
#     def set_alpha(self, alpha):
#         self.alpha = alpha
#
#     def forward(self, x, labels, feat=None):
#         N, C = x.size()[:2]
#         num_embed = feat.size(1)
#         # Part 1. generate parameter-free normalized activations
#         normalized = self.param_free_norm(x)
#
#         # use noise and labels to produce latent code
#         noise = torch.randn(N, num_embed, self.noise_dim).to(x.device)
#         labels = labels.view(N, 1, -1).expand(-1, num_embed, -1)
#         latent = torch.cat([noise, labels], dim=2)
#         latent_code = self.mlp_latent(latent)
#         # latent_code = self.mlp_latent(labels)
#
#         # Part 2. produce scaling and bias conditioned on semantic map
#         if feat is None:
#             mix_feat = latent_code
#         else:
#             enc_feat = self.mlp_shared(feat)
#             mix_feat = enc_feat * self.alpha + latent_code * (1 - self.alpha)
#
#             # replace style embed with latent code if style embed is all zeros
#             mask_indices = (feat == 0).all(dim=2).view(N, num_embed, 1)
#             mix_feat = mix_feat * ~mask_indices + latent_code * mask_indices
#
#         beta, gamma = calc_embed_mean_std(mix_feat, eps=self.eps)
#
#         # apply scale and bias
#         out = normalized * (1 + gamma) + beta
#
#         return out


# class MSEAN(nn.Module):
#     def __init__(self, embed_nc, norm_nc, label_nc,
#                  latent_dim=10, norm_layer=nn.BatchNorm2d, use_embed_only=False):
#         super().__init__()
#
#         self.latent_dim = latent_dim
#         self.alpha = 1.
#         self.use_embed_only = use_embed_only
#         self.param_free_norm = norm_layer(norm_nc, affine=False)
#         # The dimension of the intermediate embedding space. Yes, hardcoded.
#         hidden_nc = 128
#
#         self.mlp_shared = nn.Linear(embed_nc, norm_nc)
#
#         self.mlp_latent = nn.Sequential(nn.Linear(label_nc + latent_dim, hidden_nc // 2),
#                                         nn.ReLU(inplace=True),
#                                         nn.Linear(hidden_nc // 2, hidden_nc),
#                                         nn.ReLU(inplace=True))
#         self.eps = 1e-5
#
#     def set_alpha(self, alpha):
#         self.alpha = alpha
#
#     def forward(self, x, labels, feats=None):
#         # Part 1. generate parameter-free normalized activations
#         normalized = self.param_free_norm(x)
#         enc_feats = self.mlp_shared(feats)
#         beta, gamma = calc_embed_mean_std(enc_feats, eps=self.eps)
#
#         # apply scale and bias
#         out = normalized * gamma + beta
#
#         return out
