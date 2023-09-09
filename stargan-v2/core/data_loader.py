"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
import pathlib
from pathlib import Path
from itertools import chain
import os
import random

from munch import Munch
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler, BatchSampler, SequentialSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None, num_ref=2):
        self.num_ref = num_ref
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self, root):
        domains = os.listdir(root)
        labels = []
        root = pathlib.Path(root)
        fnames_list = [[] for _ in range(self.num_ref)]
        for idx, domain in enumerate(sorted(domains)):
            class_dir = root / domain
            cls_fnames = list(class_dir.iterdir())
            fnames_list[0] += cls_fnames
            for i in range(1, self.num_ref):
                fnames_list[i] += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
        return list(zip(*fnames_list)), labels

    def __getitem__(self, index):
        fnames = self.samples[index]
        label = self.targets[index]
        imgs = []
        for fname in fnames:
            img = Image.open(fname).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        return *imgs, label

    def __len__(self):
        return len(self.targets)


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_train_loader(root, which='source', img_size=256,
                     batch_size=8, prob=0.5, num_workers=4):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    crop = transforms.RandomResizedCrop(
        img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(
        lambda x: crop(x) if random.random() < prob else x)

    transform = transforms.Compose([
        rand_crop,
        transforms.Resize([img_size, img_size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    if which == 'source':
        dataset = ImageFolder(root, transform)
    elif which == 'reference':
        dataset = ReferenceDataset(root, transform)
    else:
        raise NotImplementedError

    sampler = _make_balanced_sampler(dataset.targets)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


def get_eval_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = DefaultDataset(root, transform=transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def get_test_loader(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4, num_ref=1):
    print('Preparing DataLoader for the generation phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    if num_ref == 1:
        dataset = ImageFolder(root, transform)
    elif num_ref > 1:
        dataset = ReferenceDataset(root, transform, num_ref=num_ref)
    else:
        dataset = ReferenceDataset(root, transform, num_ref=1)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode='', norm_type='adain', hidden_nc='256'):
        self.loader = loader
        self.loader_ref = loader_ref
        if norm_type == 'adain':
            self.latent_dim = latent_dim
        elif norm_type == 'sean':
            self.latent_dim = hidden_nc
        else:
            raise NotImplementedError('norm type [%s] is not found' % norm_type)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode
        self.iter = None

    def _fetch_inputs(self):
        if self.iter is None:
            self.iter = iter(self.loader)
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def _fetch_mult_refs(self):
        try:
            xs_y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            xs_y = next(self.iter_ref)
        return xs_y

    def __next__(self):
        x, y = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, x_ref2, y_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2,
                           z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'pretrain':
            z_trg = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, z_trg=z_trg)
        elif self.mode == 'val':
            xs_y = self._fetch_mult_refs()
            inputs = Munch(x_src=x, y_src=y, x_ref=xs_y[0], y_ref=xs_y[-1])
            for ref_idx, x_ref in enumerate(xs_y[1:-1], start=2):
                ref_key = f'x_ref{ref_idx}'
                inputs[ref_key] = x_ref
        elif self.mode == 'test':
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})


def get_style_loader(root, num_embeds=5, batch_size=8, num_workers=4):
    print('Preparing Style DataLoader for the training phase...')
    img_size = 224
    prob = 0.5
    crop = transforms.RandomResizedCrop(
        img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(
        lambda x: crop(x) if random.random() < prob else x)
    transform = transforms.Compose([
        rand_crop,
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = RandomReferenceDataset(root, transform, num_embeds)
    # infinite_random_sampler = InfiniteRandomBatchSampler(dataset, min_batch_size=1, max_batch_size=num_embeds)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


def get_test_style_loader(root, domain, num_embeds=5, batch_size=8, num_workers=4):
    print('Preparing Style DataLoader for the evaluation phase...')
    img_size = 224
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = SingleRandomReferenceDataset(root, domain, transform, num_embeds)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


class RandomReferenceDataset(Dataset):
    def __init__(self, root, transform=None, num_embeds=5):
        self.samples, self.targets, self.label_fnames = self._make_dataset(root)
        self.transform = transform
        self.num_embeds = num_embeds

    def _make_dataset(self, root):
        domains = os.listdir(root)
        domains.sort()
        fnames, fnames2, labels = [], [], []
        label_fnames = {}
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
            label_fnames[idx] = cls_fnames
        return list(zip(fnames, fnames2)), labels, label_fnames

    def __getitem__(self, index):
        fnames, fnames2 = self.samples[index]
        fnames, fnames2 = [fnames], [fnames2]
        label = self.targets[index]
        if self.num_embeds > 1:
            fnames += random.choices(self.label_fnames[label], k=(self.num_embeds - 1))
            fnames2 += random.choices(self.label_fnames[label], k=(self.num_embeds - 1))
        imgs = []
        imgs2 = []
        for fname, fname2 in zip(fnames, fnames2):
            img = Image.open(fname).convert('RGB')
            img2 = Image.open(fname2).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
                img2 = self.transform(img2)
            imgs.append(img)
            imgs2.append(img2)
        return torch.stack(imgs), torch.stack(imgs2), label

    def __len__(self):
        return len(self.targets)


class SingleRandomReferenceDataset(Dataset):
    def __init__(self, root, domain, transform=None, num_embeds=5):
        self.samples = listdir(os.path.join(root, domain))
        self.transform = transform
        self.num_embeds = num_embeds

    def __getitem__(self, index):
        fnames = [self.samples[index]]
        if self.num_embeds > 1:
            fnames += random.choices(self.samples, k=(self.num_embeds - 1))
        imgs = []
        for fname in fnames:
            img = Image.open(fname).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        return torch.stack(imgs)

    def __len__(self):
        return len(self.samples)
