from torch.utils.data import DataLoader

from datasets import find_dataset_using_name
from options.defectgan_options import TrainOptions
from utils.util import worker_init_fn
from torchvision import transforms
from utils.util import fix_rand_seed
from trainers import find_trainer_using_model_name
import math


def main():
    fix_rand_seed()
    # defectgan_options
    opt = TrainOptions().parse()

    dataset_cls = find_dataset_using_name(opt.dataset_name)
    val_loader = None
    # TODO calculate dataset's mean and std
    train_transform = transforms.Compose([
        transforms.Resize(opt.image_size * 1.5),
        transforms.RandomResizedCrop((opt.image_size, opt.image_size), scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        # transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    train_dataset = dataset_cls(opt, phase='train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                              num_workers=4, worker_init_fn=worker_init_fn)
    print(f'{len(train_loader.dataset)} images in train set')  # Should print 36000
    if opt.phase == 'val':
        val_transform = transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.CenterCrop((opt.image_size, opt.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        val_dataset = dataset_cls(opt, phase='val', transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False,
                                num_workers=4, worker_init_fn=worker_init_fn)
        print(f'{len(val_loader.dataset)} images in val set')  # Should print 4000

    trainer = find_trainer_using_model_name(opt.model)(opt, len(train_loader))
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
