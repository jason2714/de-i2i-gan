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
from metrics.inception import InceptionV3
import torch
from metrics.fid_score import calculate_fid_from_model
from torchvision.utils import save_image

DATATYPE = ["defects", "background", "fusion"]


@torch.no_grad()
def main():
    fix_rand_seed()
    # defectgan_options
    opt = TestOptions().parse()
    dataset_cls = find_dataset_using_name(opt.dataset_name)
    opt.clf_loss_type = dataset_cls.clf_loss_type

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
                   "background": int(1e10),
                   "fusion": None}
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
    model.load(opt.which_epoch)

    if opt.cal_fid:
        assert opt.npz_path is not None, 'npz_path should not be None if cal_fid is True'
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[opt.dims]
        inception_model = InceptionV3([block_idx]).to(opt.device, non_blocking=True)
        # must use model.eval() to ignore dropout and batchNorm, otherwise the value will break
        inception_model.eval()

        fid_value = calculate_fid_from_model(opt, model, inception_model, test_loaders, 'Testing... ')
        print(f'FID: {fid_value} at epoch {opt.which_epoch}')

    if opt.save_img_grid:
        import json
        anno_dir = opt.data_dir / opt.dataset_name / 'metadata'
        label2idx_path = anno_dir / 'label2idx.json'
        label2idx = json.loads(label2idx_path.read_text())
        idx2label = {idx: label for label, idx in label2idx.items()}

        output_dir = opt.results_dir / opt.name / 'images'
        output_dir.mkdir(parents=True, exist_ok=True)
        bg_data, _, _ = next(test_loaders['background'])
        # _, df_labels, _ = next(iter(test_loaders['defects']))
        num_grids = 2
        for label_idx in range(1, opt.label_nc):
            grid_labels = torch.zeros(num_grids * num_grids, opt.label_nc, num_grids, num_grids)
            for grid_row in range(num_grids):
                for grid_col in range(num_grids):
                    grid_labels[grid_row * num_grids + grid_col, label_idx, grid_row, grid_col] = 1
            org_label = torch.zeros(1, opt.label_nc, num_grids, num_grids)
            org_label[0, label_idx] = 1
            grid_labels = torch.cat((org_label, grid_labels))
            df_grid = model('generate_grid', bg_data, grid_labels)
            output_path = output_dir / f'{idx2label[label_idx]}.png'
            save_image(df_grid, output_path)

    if opt.save_img:
        output_dir = opt.results_dir / opt.name / 'images'
        output_dir.mkdir(parents=True, exist_ok=True)
        bg_data, _, _ = next(test_loaders['background'])

        # generate single defect image grid
        labels = torch.eye(opt.label_nc)[1:]
        df_grid = model('generate_grid', bg_data, labels, img_only=True)
        single_output_path = output_dir / f'Single.png'
        save_image(df_grid, single_output_path)

        # generate multiple defect image grid
        _, df_labels, _ = next(iter(test_loaders['defects']))
        multi_df_grid = model('generate_grid', bg_data, df_labels, img_only=True)
        multi_output_path = output_dir / f'Multiple.png'
        save_image(multi_df_grid, multi_output_path)

    if opt.cal_clf:
        data_loader = test_loaders['fusion']
        acc = 0
        loss = 0
        pbar = tqdm(data_loader, colour='MAGENTA')
        # BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
        for data, labels, _ in pbar:
            pbar.set_description(f'Validating model ... ')
            logits, clf_loss = model('inference_classifier', data, labels, img_only=True)
            predictions = (torch.sigmoid(logits) >= 0.5).int().cpu()
            acc += (predictions == labels).all(dim=1).sum()
            loss += clf_loss.item()
        print(f'Acc: {acc / len(data_loader.dataset):.3f} ({acc}/{len(data_loader.dataset)}), '
              f'Loss: {loss / len(data_loader):.3f}')


if __name__ == '__main__':
    main()
    '''
    python test_defectgan.py --data_dir A:/research/data --batch_size 4 --name org --save_img_grid
    python test_defectgan.py --name mae_shrink_token_2 --data_dir A:/research/data --batch_size 32 --add_noise --use_spectral --npz_path A:\research\data\codebrim\test\defects00.npz
    '''
