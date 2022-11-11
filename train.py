from torch.utils.data import DataLoader
from options.train_options import TrainOptions
from trainers.wgan_trainer import WGanTrainer
from datasets.face_dataset import FaceDataset
from utils.util import worker_init_fn
from torchvision import transforms
from utils.util import fix_rand_seed


def main():
    fix_rand_seed()
    opt = TrainOptions().parse()

    trainer = WGanTrainer(opt)
    val_loader = None
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_dataset = FaceDataset(opt, phase='train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                              num_workers=4, worker_init_fn=worker_init_fn)
    print(f'{len(train_loader)} images in train set')  # Should print 36000
    if opt.phase == 'val':
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        val_dataset = FaceDataset(opt, phase='val', transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False,
                                num_workers=4, worker_init_fn=worker_init_fn)
        print(f'{len(val_loader)} images in val set')  # Should print 4000

    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
