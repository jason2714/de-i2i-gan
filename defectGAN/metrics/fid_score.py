"""Calculates the Frechet Inception Distance (FID) to evalulate GANs
The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.
See --help to see further details.
Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy
import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x
try:
    from metrics.inception import InceptionV3
except ModuleNotFoundError:
    from inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=128,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(4, num_cpus)`'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('path', type=str, nargs=2,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--num-imgs', type=int, nargs=2, default=[None, None], help='use # images to calculate FID score')
parser.add_argument('--save_npz', action='store_true', help='whether save the first statistics to npz file')
parser.add_argument('calc_mfid', action='store_true', help='whether calculate the mFID score')

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(files, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1, num_img=None):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)
    image_size = 128
    fid_transforms = transforms.Compose([
        # transforms.Resize(image_size),
        transforms.RandomCrop((image_size, image_size), pad_if_needed=True),
        transforms.ToTensor()
    ])
    dataset = ImagePathDataset(files, transforms=fid_transforms)
    num_img = len(dataset) if num_img is None else num_img
    sampler = RandomSampler(dataset, replacement=False, num_samples=num_img)
    dataloader = DataLoader(dataset,
                            sampler=sampler,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=num_workers,
                            pin_memory=True)
    pred_arr = None

    start_idx = 0
    # progress = tqdm(total=file_len)
    # while start_idx < file_len:
    #     batch = next(dataloader)
    for batch in tqdm(dataloader):
        batch = batch.to(device, non_blocking=True)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        if pred_arr is None:
            pred_arr = pred
        else:
            pred_arr = np.concatenate((pred_arr, pred), axis=0)

        start_idx = start_idx + pred.shape[0]
        # progress.update(pred.shape[0])
    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1, num_img=None):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, device, num_workers, num_img)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, dims, device,
                               num_workers=1, num_img=None, save_npz=False):
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                        for file in path.glob('*.{}'.format(ext))])
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, device, num_workers, num_img)
        if save_npz:
            save_id = 0
            save_name = f'{path}{save_id:02d}.npz'
            while os.path.exists(save_name):
                save_id += 1
                save_name = f'{path}{save_id:02d}.npz'
            np.savez(save_name, mu=m, sigma=s)

    return m, s


def calculate_fid_from_model(opt, model, inception_model, data_loader, label_loader,
                             input_stat, description=None):
    pred_arr = None
    if isinstance(label_loader, tuple):
        num_batches = opt.num_imgs // opt.batch_size
        labels = torch.FloatTensor(label_loader).view(1, -1).repeat(opt.batch_size, 1)
        label_loader = [(_, labels[:], _) for _ in range(num_batches)]
    # pbar = tqdm(label_loader, colour='MAGENTA')
    # # BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
    # for _, df_labels, _ in pbar:
    #     pbar.set_description(description)
    for _, df_labels, _ in label_loader:
        bg_data, bg_labels, _ = next(data_loader)
        bg_data, bg_labels = bg_data[:df_labels.size(0)], bg_labels[:df_labels.size(0)]
        fake_imgs, _ = model('inference', bg_data, df_labels)
        # save_generated_images(opt, file_paths, fake_imgs)
        fake_imgs = (fake_imgs + 1) / 2
        out = inception_model(fake_imgs)
        pred = out[0]
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        if pred_arr is None:
            pred_arr = pred
        else:
            pred_arr = np.concatenate((pred_arr, pred), axis=0)
    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    fid_value = calculate_fid((input_stat, [mu, sigma]), opt.batch_size,
                              opt.device, opt.dims,
                              num_workers=4, model=inception_model)
    return fid_value


def calculate_fid(inputs, batch_size, device, dims, num_workers=1, num_imgs=(None, None), save_npz=False, model=None):
    if model is None:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(device, non_blocking=True)
    m_s = []
    for input_data, num_img in zip(inputs, num_imgs):
        if isinstance(input_data, str):
            if not os.path.exists(input_data):
                raise RuntimeError('Invalid path: %s' % input_data)
            m, s = compute_statistics_of_path(input_data, model, batch_size,
                                              dims, device, num_workers, num_img, save_npz)
            m_s += [m, s]
        elif isinstance(input_data, (tuple, list)):
            for data in input_data:
                assert isinstance(data, np.ndarray), 'input type of mu and sigma must be np.ndarray'
            m_s = [*m_s, *input_data]
        else:
            raise ValueError(f'Invalid input type {type(input_data)} expected tuple or str')
    fid_value = calculate_frechet_distance(*m_s)

    return fid_value


def main():
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        num_avail_cpus = len(dir(os))
        num_workers = min(num_avail_cpus, 4)
    else:
        num_workers = args.num_workers
    if args.calc_mfid:
        fid_values = []
        first_class_stats = np.load(args.path[0], allow_pickle=True).item()
        second_class_stats = np.load(args.path[1], allow_pickle=True).item()
        for label in first_class_stats.keys():
            fid_value = calculate_fid((first_class_stats[label], second_class_stats[label]),
                                      args.batch_size,
                                      device,
                                      args.dims,
                                      num_workers,
                                      args.num_imgs,
                                      args.save_npz)
            fid_values.append(fid_value)
            print(f'FID for class {label}: {fid_value:.4f}')
        print(f'mFID: {sum(fid_values) / len(fid_values)}')
    else:
        fid_value = calculate_fid(args.path,
                                  args.batch_size,
                                  device,
                                  args.dims,
                                  num_workers,
                                  args.num_imgs,
                                  args.save_npz)
        print('FID: ', fid_value)


if __name__ == '__main__':
    main()
