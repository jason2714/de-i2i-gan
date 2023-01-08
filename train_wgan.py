from torch.utils.data import DataLoader

from datasets import find_dataset_using_name
from options.wgan_options import TrainOptions
from utils.util import worker_init_fn
from torchvision import transforms
from utils.util import fix_rand_seed
from trainers import find_trainer_using_model_name
import math


def main():
    fix_rand_seed()
    # wgan_options
    opt = TrainOptions().parse()

    dataset_cls = find_dataset_using_name(opt.dataset_name)
    val_loader = None
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_dataset = dataset_cls(opt, phase='train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                              num_workers=4, worker_init_fn=worker_init_fn, pin_memory=True)
    print(f'{len(train_loader.dataset)} images in train set')  # Should print 36000
    if opt.phase == 'val':
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        val_dataset = dataset_cls(opt, phase='val', transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False,
                                num_workers=4, worker_init_fn=worker_init_fn, pin_memory=True)
        print(f'{len(val_loader.dataset)} images in val set')  # Should print 4000

    trainer = find_trainer_using_model_name(opt.model)(opt, len(train_loader))
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
