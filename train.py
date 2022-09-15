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

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    face_dataset = FaceDataset(opt, transform=transform)
    print(f'{len(face_dataset)} images in {opt.phase} set')  # Should print 40000
    data_loader = DataLoader(face_dataset, batch_size=opt.batch_size, shuffle=True,
                             num_workers=4, worker_init_fn=worker_init_fn)
    trainer = WGanTrainer(opt)
    trainer.train(data_loader)


if __name__ == '__main__':
    main()
