from torch.utils.data import DataLoader

from datasets import find_dataset_using_name
from loaders.infinite_loader import InfiniteDataLoader
from datasets.concat_dataset import ConcatDataset
from options.defectgan_options import TrainOptions
from utils.util import worker_init_fn
from torchvision import transforms
from utils.util import fix_rand_seed
from trainers import find_trainer_using_model_name
import math

DATATYPE = ["defects", "background"]
LOADER_CLS = {"defects": DataLoader,
              "background": InfiniteDataLoader}


def view_data_after_transform(opt, loaders):
    import numpy as np
    import imageio
    from pathlib import Path
    for data_type in DATATYPE:
        if data_type == 'defects':
            iterator = iter(loaders[data_type])
        else:
            iterator = loaders[data_type]
        images, labels, file_paths = next(iterator)
        for i, (image, file_path) in enumerate(zip(images, file_paths)):
            file_path = Path(file_path)
            image_out = image.permute(1, 2, 0).numpy()
            image_out = np.uint8((image_out + 1) * 127.5)
            imageio.imsave(opt.ckpt_dir / opt.name / f'{file_path.stem}_new_0.png', image_out)
            import shutil
            shutil.copyfile(file_path, opt.ckpt_dir / opt.name / file_path.name)
        images, labels, file_paths = next(iterator)
        for i, (image, file_path) in enumerate(zip(images, file_paths)):
            file_path = Path(file_path)
            image_out = image.permute(1, 2, 0).numpy()
            image_out = np.uint8((image_out + 1) * 127.5)
            imageio.imsave(opt.ckpt_dir / opt.name / f'{file_path.stem}_new_1.png', image_out)
            import shutil
            shutil.copyfile(file_path, opt.ckpt_dir / opt.name / file_path.name)
        images, labels, file_paths = next(iterator)
        for i, (image, file_path) in enumerate(zip(images, file_paths)):
            file_path = Path(file_path)
            image_out = image.permute(1, 2, 0).numpy()
            image_out = np.uint8((image_out + 1) * 127.5)
            imageio.imsave(opt.ckpt_dir / opt.name / f'{file_path.stem}_new_2.png', image_out)
            import shutil
            shutil.copyfile(file_path, opt.ckpt_dir / opt.name / file_path.name)


def main():
    fix_rand_seed()
    # defectgan_options
    opt = TrainOptions().parse()

    dataset_cls = find_dataset_using_name(opt.dataset_name)
    # TODO calculate dataset's mean and std
    train_transform = transforms.Compose([
        transforms.Resize(int(opt.image_size * 1.5)),
        transforms.RandomResizedCrop((opt.image_size, opt.image_size), scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_datasets = {
        data_type: dataset_cls(opt, phase='train', data_type=data_type, transform=train_transform)
        for data_type in DATATYPE
    }
    train_loaders = {
        data_type: LOADER_CLS[data_type](train_dataset, batch_size=opt.batch_size, shuffle=True,
                                         num_workers=1, worker_init_fn=worker_init_fn)
        for data_type, train_dataset in train_datasets.items()
    }
    for data_type in DATATYPE:
        print(f'{len(train_loaders[data_type].dataset)} images in train {data_type} set')
    val_loaders = None
    if opt.phase == 'val':
        val_transform = transforms.Compose([
            transforms.Resize(opt.image_sideze),
            transforms.CenterCrop((opt.image_size, opt.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        val_datasets = {
            data_type: dataset_cls(opt, phase='val', data_type=data_type, transform=val_transform)
            for data_type in DATATYPE
        }
        val_loaders = {
            data_type: DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False,
                                  num_workers=4, worker_init_fn=worker_init_fn)
            for data_type, val_dataset in val_datasets.items()
        }
        for data_type in DATATYPE:
            print(f'{len(val_loaders[data_type].dataset)} images in train {data_type} set')
    # view_data_after_transform(opt, train_loaders)
    # trainer = find_trainer_using_model_name(opt.model)(opt,
    #                                                    max(len(train_loaders[data_type]) for data_type in DATATYPE))
    # trainer.train(train_loaders, val_loaders)


if __name__ == '__main__':
    main()
