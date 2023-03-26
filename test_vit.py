import torch

from datasets.codebrim_dataset import CodeBrimDataset
from PIL import Image
from torchvision import transforms
from datasets import find_dataset_using_name
from models import create_model
from trainers import find_trainer_using_model_name
from utils.util import fix_rand_seed, worker_init_fn
from options.vit_options import TestOptions
from torch.utils.data import DataLoader
from tqdm import tqdm


def arg_parse():
    opt = TestOptions().parse()
    return opt


@torch.no_grad()
def test():
    opt = arg_parse()
    dataset_cls = find_dataset_using_name(opt.dataset_name)
    opt.clf_loss_type = dataset_cls.clf_loss_type

    test_transform = transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.RandomCrop((opt.image_size, opt.image_size), pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_dataset = dataset_cls(opt, phase='test', data_type='fusion', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False,
                             num_workers=4, worker_init_fn=worker_init_fn, pin_memory=True)
    print(f'{len(test_loader.dataset)} images in test fusion set')

    model = create_model(opt)
    model.load(opt.which_epoch)

    acc = 0
    loss = 0
    pbar = tqdm(test_loader, colour='MAGENTA')
    # BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
    for data, labels, _ in pbar:
        pbar.set_description(f'Validating model ... ')
        logits, clf_loss = model(data, labels, inference=True)
        predictions = (torch.sigmoid(logits) >= 0.5).int().cpu()
        acc += (predictions == labels).all(dim=1).sum()
        loss += clf_loss.item()
    print(f'Acc: {acc / len(test_loader.dataset):.3f} ({acc}/{len(test_loader.dataset)}), '
          f'Loss: {loss / len(test_loader):.3f}')


if __name__ == '__main__':
    fix_rand_seed()
    test()

"""
python test_vit.py --name vit --data_dir A:/research/data
"""
