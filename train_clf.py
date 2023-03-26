from datasets.codebrim_dataset import CodeBrimDataset
from PIL import Image
from torchvision import transforms
from datasets import find_dataset_using_name
from trainers import find_trainer_using_model_name
from utils.util import fix_rand_seed, worker_init_fn
from options.vit_options import TrainOptions
from torch.utils.data import DataLoader
from tqdm import tqdm


def arg_parse():
    opt = TrainOptions().parse()
    return opt


def train():
    opt = arg_parse()
    dataset_cls = find_dataset_using_name(opt.dataset_name)
    opt.clf_loss_type = dataset_cls.clf_loss_type

    train_transform = transforms.Compose([
        transforms.Resize(int(opt.image_size * 1.5)),
        transforms.RandomResizedCrop((opt.image_size, opt.image_size), scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = dataset_cls(opt, phase='train', data_type='fusion', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                              num_workers=4, worker_init_fn=worker_init_fn, pin_memory=True)
    print(f'{len(train_loader.dataset)} images in train fusion set')

    val_transform = transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.RandomCrop((opt.image_size, opt.image_size), pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_dataset = dataset_cls(opt, phase='val', data_type='fusion', transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False,
                            num_workers=4, worker_init_fn=worker_init_fn, pin_memory=True)
    print(f'{len(val_loader.dataset)} images in train fusion set')

    opt.iters_per_epoch = len(train_loader)
    trainer = find_trainer_using_model_name(opt.model)(opt)
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    fix_rand_seed()
    train()

"""
python train_clf.py --name vit --phase val --data_dir A:/research/data
"""