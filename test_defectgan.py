from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import find_dataset_using_name
from options.defectgan_options import TestOptions
from utils.util import worker_init_fn
from torchvision import transforms
from utils.util import fix_rand_seed
from trainers import find_trainer_using_model_name
import math
from tqdm import tqdm
from models import create_model
import numpy as np
from metrics.fid_score import calculate_fid
from metrics.inception import InceptionV3
import torch
from torch.nn.functional import adaptive_avg_pool2d

DATATYPE = ["defects", "background"]

from pathlib import Path
import imageio
import shutil
def save_generated_images(opt, file_paths, fake_imgs):
    base_dir = opt.results_dir / opt.name / 'images'
    base_dir.mkdir(parents=True, exist_ok=True)
    for file_path, image_out in zip(file_paths, fake_imgs):
        file_path = Path(file_path)
        image_out = image_out.permute(1, 2, 0).cpu().numpy()
        image_out = np.uint8((image_out + 1) * 127.5)
        imageio.imsave(base_dir / f'{file_path.stem}_new_0.png', image_out)
        shutil.copyfile(file_path, base_dir / file_path.name)


@torch.no_grad()
def main():
    fix_rand_seed()
    # defectgan_options
    opt = TestOptions().parse()

    dataset_cls = find_dataset_using_name(opt.dataset_name)
    test_transform = transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.RandomCrop((opt.image_size, opt.image_size), pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_datasets = {
        data_type: dataset_cls(opt, phase='test', data_type=data_type, transform=test_transform)
        for data_type in DATATYPE
    }
    NUM_SAMPLES = {"defects": opt.num_imgs,
                   "background": int(1e10)}
    test_samplers = {
        data_type: RandomSampler(test_datasets[data_type], replacement=False, num_samples=NUM_SAMPLES[data_type])
        for data_type in DATATYPE
    }
    test_loaders = {
        data_type: DataLoader(test_dataset, sampler=test_samplers[data_type],
                              batch_size=opt.batch_size, shuffle=False,
                              num_workers=4, worker_init_fn=worker_init_fn, pin_memory=True)
        for data_type, test_dataset in test_datasets.items()
    }
    for data_type in DATATYPE:
        print(f'{len(test_loaders[data_type].dataset)} images in {opt.phase} {data_type} set')

    # change loader to iterator
    test_loaders['background'] = iter(test_loaders['background'])

    # view_data_after_transform(opt, train_loaders)
    model = create_model(opt)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[opt.dims]
    inception_model = InceptionV3([block_idx]).to(opt.device, non_blocking=True)
    # must use model.eval() to ignore dropout and batchNorm, otherwise the value will break
    inception_model.eval()

    # if opt.which_epoch == 'all':
    #     opt.ckpt_dir / opt.name
    model.load(opt.which_epoch)
    for i in range(10, 400, 10):
        model.load(i)
        pred_arr = None
        for df_data, df_labels, _ in tqdm(test_loaders['defects'], colour='MAGENTA'):
            bg_data, bg_labels, file_paths = next(test_loaders['background'])
            bg_data, bg_labels = bg_data[:df_data.size(0)], bg_labels[:df_data.size(0)]
            fake_imgs = model('inference', bg_data, df_labels)

            # save_generated_images(opt, file_paths, fake_imgs)

            fake_imgs = (fake_imgs + 1) / 2
            pred = inception_model(fake_imgs)[0]
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
        fid_value = calculate_fid((opt.npz_path, [mu, sigma]), opt.batch_size, opt.device, opt.dims,
                                  num_workers=4, model=inception_model)

        print('FID: ', fid_value)

if __name__ == '__main__':
    main()
    # python test_defectgan.py --data_dir A:\research\data --batch_size 128 --name original_lw_guess --npz_path A:\research\data\codebrim\train\defects00.npz
