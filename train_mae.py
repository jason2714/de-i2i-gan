from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import find_dataset_using_name
from options.defectgan_options import PreTrainOptions
from utils.util import worker_init_fn
from torchvision import transforms
from utils.util import fix_rand_seed
from trainers import find_trainer_using_model_name
import math

TRAIN_DATATYPE = ["defects", "background"]
TRAIN_NUM_SAMPLES = {"defects": None,
                     "background": None}
VAL_DATATYPE = ["background", "background_inf"]
VAL_NUM_SAMPLES = {"background": None,
                   "background_inf": int(1e10)}


def train():
    opt = PreTrainOptions().parse()
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
        for data_type in TRAIN_DATATYPE
    }
    train_samplers = {
        data_type: RandomSampler(train_datasets[data_type], replacement=False, num_samples=TRAIN_NUM_SAMPLES[data_type])
        for data_type in TRAIN_DATATYPE
    }
    train_loaders = {
        data_type: DataLoader(train_dataset, sampler=train_samplers[data_type],
                              batch_size=opt.batch_size, shuffle=False,
                              num_workers=4, worker_init_fn=worker_init_fn, pin_memory=True)
        for data_type, train_dataset in train_datasets.items()
    }
    for data_type in TRAIN_DATATYPE:
        print(f'{len(train_loaders[data_type].dataset)} images in train {data_type} set')

    # for val
    val_transform = transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.RandomCrop((opt.image_size, opt.image_size), pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    val_datasets = {
        data_type: dataset_cls(opt, phase='val', data_type='background', transform=val_transform)
        for data_type in VAL_DATATYPE
    }
    val_samplers = {
        data_type: RandomSampler(val_datasets[data_type], replacement=False, num_samples=VAL_NUM_SAMPLES[data_type])
        for data_type in VAL_DATATYPE
    }
    val_loaders = {
        data_type: DataLoader(val_dataset, sampler=val_samplers[data_type],
                              batch_size=opt.num_display_images, shuffle=False,
                              num_workers=4, worker_init_fn=worker_init_fn, pin_memory=True)
        for data_type, val_dataset in val_datasets.items()
    }
    for data_type in VAL_DATATYPE:
        print(f'{len(val_loaders[data_type].dataset)} images in val {data_type} set')

    # change loader to iterator
    val_loaders['background_inf'] = iter(val_loaders['background'])

    trainer = find_trainer_using_model_name('mae')(opt, len(train_loaders['background']))
    trainer.train(train_loaders, val_loaders)


if __name__ == '__main__':
    fix_rand_seed()
    train()
    '''
    python train_mae.py --data_dir A:/research/data --name mae --phase val --add_noise --use_spectral
    python train_mae.py --name mae --continue_training --load_from_opt_file
    tensorboard --logdir log/test_df --samples_per_plugin "images=100"
    '''
