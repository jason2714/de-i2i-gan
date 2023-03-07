import random
import numpy as np
import torch
import logging
from pathlib import Path
import torchvision
from torch import nn

def use_gpu(gpu_ids=0):
    use_cuda = torch.cuda.is_available()
    # print(use_cuda)
    device = torch.device(f'cuda:{gpu_ids}' if use_cuda else 'cpu')
    print('Device used:', device)
    if use_cuda:
        torch.cuda.set_device(device)
    return device


def fix_rand_seed(seed=123):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def init_logging(name, level=logging.INFO):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


def generate_mask(image_size, patch_size, mask_ratio):
    """
        input image_size: (b, c, h, w)
        output size: (b, 1, h, w)
    """
    up_scale = nn.Upsample(scale_factor=patch_size, mode='nearest')
    height_size = image_size[2] // patch_size
    width_size = image_size[3] // patch_size
    mask = torch.bernoulli(torch.full((image_size[0], 1, height_size, width_size), (1 - mask_ratio)))

    return up_scale(mask)
