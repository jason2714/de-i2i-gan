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
from collections import defaultdict


def arg_parse():
    opt = TestOptions().parse()
    return opt


def test_classifier(data_loader, model):
    acc = 0
    loss = 0
    pbar = tqdm(data_loader, colour='MAGENTA')
    # BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
    for data, labels, _ in pbar:
        pbar.set_description(f'Validating model ... ')
        logits, clf_loss = model('inference', data, labels)
        predictions = (torch.sigmoid(logits) >= 0.5).int().cpu()
        acc += (predictions == labels).all(dim=1).sum()
        loss += clf_loss.item()
    print(f'Acc: {acc / len(data_loader.dataset):.3f} ({acc}/{len(data_loader.dataset)}), '
          f'Loss: {loss / len(data_loader):.3f}')
    return acc / len(data_loader.dataset), loss / len(data_loader)


def get_embeddings(data_loader, model):
    label_embeddings = defaultdict(list)
    pbar = tqdm(data_loader, colour='MAGENTA')
    # BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
    for data, labels, _ in pbar:
        pbar.set_description(f'Aggregating Embeddings... ')
        embeddings = model('get_embedding', data, labels).cpu()
        for label, embedding in zip(labels, embeddings):
            label_embeddings[tuple(label.tolist())].append(embedding)
    return label_embeddings


def save_embeddings(label_embeddings, opt):
    out_dir = opt.results_dir / opt.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{opt.which_epoch}_embeddings.pth'
    torch.save(label_embeddings, out_path)
    print(f'Embeddings saved to {out_path}')

    embed_test = torch.load(out_path)
    print(embed_test.keys())
    for label in label_embeddings.keys():
        if sum(label) == 1:
            print(label, len(label_embeddings[label]))


@torch.no_grad()
def test():
    opt = arg_parse()
    # dataset_cls = find_dataset_using_name(opt.dataset_name)
    dataset_cls = CodeBrimDataset
    opt.clf_loss_type = dataset_cls.clf_loss_type

    test_transform = transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.RandomCrop((opt.image_size, opt.image_size), pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_dataset = dataset_cls(opt, phase=opt.phase, data_type='fusion', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False,
                             num_workers=4, worker_init_fn=worker_init_fn, pin_memory=True)
    print(f'{len(test_loader.dataset)} images in test fusion set')

    model = create_model(opt)
    model.load(opt.which_epoch)

    test_classifier(test_loader, model)
    if opt.save_embeddings:
        save_embeddings(get_embeddings(test_loader, model), opt)


if __name__ == '__main__':
    fix_rand_seed()
    test()

"""
python test_vit.py --name vit --data_dir A:/research/data
"""
