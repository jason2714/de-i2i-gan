from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

from datasets import find_dataset_using_name
from metrics.defectgan_metrics import calculate_metrics_from_model
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
from metrics.fid_score import calculate_fid_from_model, calculate_activation_statistics
from torchvision.utils import save_image
from models.networks.normalization import SEAN, SPADE
from collections import defaultdict
from utils.util import visualize_embeddings
from models.networks.architecture import NormResBlock
import os
from datasets.codebrim_dataset import CodeBrimDataset
from torchsummary import summary

DATATYPE = ["defects", "background", "fusion"]

layer_embeddings = defaultdict(list)
layer_res_ratios = defaultdict(list)


def check_residual_ratio(model, data_loader):
    for attr_name, attr_value in model.networks['G'].named_modules():
        if isinstance(attr_value, NormResBlock):
            attr_value.register_forward_hook(create_hook(attr_name, hook_type='res'))
    pbar = tqdm(data_loader, colour='MAGENTA')
    # BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
    for data, labels, _ in pbar:
        pbar.set_description(f'Validating model ... ')
        model('inference', data, labels, img_only=True)
    for attr_name in layer_res_ratios.keys():
        layer_res_ratios[attr_name] = torch.cat(layer_res_ratios[attr_name], dim=0).mean().item()
        print(attr_name, layer_res_ratios[attr_name])


def create_hook(attr_name, hook_type='style'):
    def style_block_hook(model, input, output):
        if output.dim() == 3:
            output = output.mean(dim=1)
        elif isinstance(model, torch.nn.modules.conv.Conv2d) and output.dim() == 4:
            output = output.mean(dim=[2, 3])
        layer_embeddings[attr_name].append(output)

    def res_block_hook(model, input, output):
        res_output = output - input[0]
        res_norm = res_output.view(res_output.size(0), -1).norm(dim=1)
        input_norm = input[0].view(input[0].size(0), -1).norm(dim=1)
        layer_res_ratios[attr_name].append(res_norm / (res_norm + input_norm))

    if hook_type == 'style':
        return style_block_hook
    elif hook_type == 'res':
        return res_block_hook
    else:
        raise ValueError(f'Unknown hook type {hook_type}')


def get_style_embeddings(model, data_loader, embed_type):
    for attr_name, attr_value in model.networks['G'].named_modules():
        if embed_type == 'hidden' and ('mlp_shared' in attr_name or 'mlp_latent' in attr_name) and \
                isinstance(attr_value, torch.nn.modules.container.Sequential):
            attr_value.register_forward_hook(create_hook(attr_name, hook_type='style'))
        elif embed_type == 'mean' and 'mlp_beta' in attr_name:
            attr_value.register_forward_hook(create_hook(attr_name, hook_type='style'))
        elif embed_type == 'std' and 'mlp_gamma' in attr_name:
            attr_value.register_forward_hook(create_hook(attr_name, hook_type='style'))

    pbar = tqdm(data_loader, colour='MAGENTA')
    # BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
    full_labels = []
    for data, labels, _ in pbar:
        full_labels.append(labels)
        pbar.set_description(f"getting {embed_type} embeddings in style blocks... ")
        model('inference', data, labels, img_only=True)
    full_labels = torch.cat(full_labels, dim=0)
    new_layer_embeddings = dict()
    for attr_name in layer_embeddings.keys():
        layer_embeddings[attr_name] = torch.cat(layer_embeddings[attr_name], dim=0).cpu()
        embeddings = defaultdict(list)
        for label, embedding in zip(full_labels, layer_embeddings[attr_name]):
            embeddings[tuple(label.int().tolist())].append(embedding)
        new_layer_embeddings[attr_name] = embeddings
    layer_embeddings.clear()
    return new_layer_embeddings


def visualize_style_embeddings(opt, embeddings, reduction_type):
    for attr_name in embeddings.keys():
        plt_dir = opt.results_dir / opt.name / reduction_type
        # plt_name = f'{opt.which_epoch}_{opt.phase}_{opt.data_type}_tsne_test.png'
        plt_name = f'{attr_name}.png'
        print(f'saving embeddings for {attr_name} to {plt_dir / plt_name}')
        visualize_embeddings(embeddings[attr_name], plt_dir, plt_name, reduction_type=reduction_type)


def check_embeddings_std(mean_layer_embeddings, std_layer_embeddings):
    for attr_name in mean_layer_embeddings.keys():
        mean_embeddings = mean_layer_embeddings[attr_name]
        print(attr_name)
        print('=' * 20)
        for label in mean_embeddings.keys():
            correction = int(not (len(mean_embeddings[label]) == 1))
            real_std = torch.stack(mean_embeddings[label]).std(dim=0, correction=correction).mean().item()
            fake_std = torch.stack(std_layer_embeddings[attr_name][label]).mean(dim=0).mean().item()
            print(label, f'{real_std:.4f}', f'{fake_std:.4f}')


def save_stats(opt, dataset):
    # load inception model
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[opt.dims]
    inception_model = InceptionV3([block_idx]).to(opt.device, non_blocking=True)
    # must use model.eval() to ignore dropout and batchNorm, otherwise the value will break
    inception_model.eval()

    class_stats = dict()
    for class_idx in range(1, opt.label_nc):
        label = [0] * opt.label_nc
        label[class_idx] = 1
        label = tuple(label)
        files = [data[0] for data in dataset.data if tuple(data[1]) == label]
        print(f'calculating stats for class {label} with class size {len(files)}...')
        stat = calculate_activation_statistics(files, inception_model, opt.batch_size, opt.dims, opt.device,
                                               num_workers=4, num_img=opt.num_imgs)
        class_stats[label] = stat

        # print(stat[0].shape, stat[1].shape)
    save_id = 0
    save_path = dataset.data[0][0].parent.parent / f'stats_{save_id:02d}'
    while os.path.exists(str(save_path) + '.npy'):
        save_id += 1
        save_path = dataset.data[0][0].parent.parent / f'stats_{save_id:02d}'
    np.save(save_path, class_stats)


@torch.no_grad()
def main():
    fix_rand_seed(123)
    # defectgan_options
    opt = TestOptions().parse()
    # dataset_cls = find_dataset_using_name(opt.dataset_name)
    dataset_cls = CodeBrimDataset
    opt.clf_loss_type = dataset_cls.clf_loss_type

    test_transform = transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.RandomCrop((opt.image_size, opt.image_size), pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_datasets = {
        data_type: dataset_cls(opt, phase=opt.phase, data_type=data_type, transform=test_transform)
        for data_type in DATATYPE
    }
    NUM_SAMPLES = {"defects": opt.num_imgs if opt.num_imgs > 0 else None,
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
    if not opt.save_stats:
        model.load(opt.which_epoch)

    if opt.metrics is not None:
        metrics = {metric: None for metric in opt.metrics}
        metric_models = dict()
        if 'fid' in opt.metrics or 'is' in opt.metrics or 'mfid' in opt.metrics:
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[opt.dims]
            metric_models['inception'] = InceptionV3([block_idx]).to(opt.device, non_blocking=True)
            # must use model.eval() to ignore dropout and batchNorm, otherwise the value will break
            metric_models['inception'].eval()
        if 'lpips' in opt.metrics:
            metric_models['lpips'] = LearnedPerceptualImagePatchSimilarity(weights='DEFAULT').to(device=opt.device)
            metric_models['lpips'].eval()

        model.networks['G'].inference_running_stats = opt.use_running_stats
        metrics = calculate_metrics_from_model(opt, model,
                                               test_loaders['background'], test_loaders['defects'],
                                               metric_models, metrics, is_score_type='std')
        for metric_names in opt.metrics:
            print(f'{metric_names}: {metrics[metric_names]} at epoch {opt.which_epoch}')

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

    if opt.save_stats:
        save_stats(opt, test_datasets['defects'])
    if opt.vis_style_embeds is not None:
        embeddings = get_style_embeddings(model, test_loaders['defects'], embed_type=opt.vis_style_embeds)
        visualize_style_embeddings(opt, embeddings, reduction_type='pca')

    if opt.save_diverse_images:
        output_dir = opt.results_dir / opt.name / 'images'
        output_dir.mkdir(parents=True, exist_ok=True)
        bg_data, _, _ = next(test_loaders['background'])
        # # generate multiple defect image grid
        _, df_labels, _ = next(iter(test_loaders['defects']))
        row_sums = torch.sum(df_labels, dim=1)
        # Select rows where the sum is greater than one
        selected_labels = df_labels[row_sums > 1]
        # Get unique label sets using numpy
        selected_labels_np = selected_labels.numpy()
        unique_labels_np = np.unique(selected_labels_np, axis=0)

        model.networks['G'].inference_running_stats = opt.use_running_stats
        # Convert back to PyTorch tensor
        unique_labels = torch.from_numpy(unique_labels_np)
        for label in unique_labels:
            label = label.unsqueeze(0).repeat(10, 1)
            multi_df_grid = model('generate_grid', bg_data, label, img_only=True)
            multi_output_path = output_dir / f'Multiple_{tuple(label[0].tolist())}.png'
            save_image(multi_df_grid, multi_output_path)

        for label_index in range(1, opt.label_nc):
            # generate single defect image grid
            labels = torch.zeros(10, opt.label_nc)
            labels[:, label_index] = 1
            df_grid = model('generate_grid', bg_data, labels, img_only=True)
            single_output_path = output_dir / f'Single_{label_index}.png'
            save_image(df_grid, single_output_path)



    # model.netG.train()
    # summary(model.netG, [(3, 128, 128), (6, 1, 1)], depth=4)
    # model.netG.print_network()
    # model.netD.train()
    # summary(model.netD, (3, 128, 128))
    # model.netD.print_network()

    # mean_embeddings = get_style_embeddings(model, test_loaders['defects'], embed_type='mean')
    # std_embeddings = get_style_embeddings(model, test_loaders['defects'], embed_type='std')
    # check_embeddings_std(mean_embeddings, std_embeddings)
    # check_residual_ratio(model, test_loaders['defects'])


if __name__ == '__main__':
    main()
    '''
    python test_defectgan.py --data_dir A:/research/data --batch_size 4 --name org --save_img_grid
    python test_defectgan.py --name mae_shrink_token_2 --data_dir A:/research/data --batch_size 32 --add_noise --use_spectral --npz_path A:\research\data\codebrim\test\defects00.npz
    python test_defectgan.py --data_dir A:/research/data --name org_sean_embed1 --use_spectral --add_noise --batch_size 32 --npz_path A:\research\data\codebrim\test\defects00.npz --num_imgs -1
    --style_norm_block_type sean --sean_alpha 1 --num_embeds 1
    '''
