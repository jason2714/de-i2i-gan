from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import find_dataset_using_name
from options.defectgan_options import PreTrainOptions, TrainOptions
from utils.util import worker_init_fn
from torchvision import transforms
from utils.util import fix_rand_seed
from trainers import find_trainer_using_model_name
from datasets.mvtec_dataset import MTVecDataset

DATATYPE = ["defects", "background"]
NUM_SAMPLES = {"defects": None,
               "background": int(1e10)}
def train():
    opt = TrainOptions().parse()
    dataset_cls = MTVecDataset
    # initial args from dataset
    opt.clf_loss_type = dataset_cls.clf_loss_type

    # TODO calculate dataset's mean and std
    train_transforms = {'background': transforms.Compose([
        transforms.Resize(int(opt.image_size)),
        # transforms.RandomResizedCrop((opt.image_size, opt.image_size), scale=(0.8, 1.0)),
        # transforms.RandomHorizontalFlip(0.2),
        # transforms.RandomVerticalFlip(0.2),
        transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]), 'defects': transforms.Compose([
        transforms.Resize(int(opt.image_size)),
        # transforms.RandomResizedCrop((opt.image_size, opt.image_size), scale=(0.8, 1.0)),
        # transforms.RandomHorizontalFlip(0.2),
        # transforms.RandomVerticalFlip(0.2),
        transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])}
    train_datasets = {
        data_type: dataset_cls(opt, phase='train', data_type=data_type, transform=train_transforms[data_type])
        for data_type in DATATYPE
    }
    train_samplers = {
        data_type: RandomSampler(train_datasets[data_type], replacement=False, num_samples=NUM_SAMPLES[data_type])
        for data_type in DATATYPE
    }
    train_loaders = {
        data_type: DataLoader(train_dataset, sampler=train_samplers[data_type],
                              batch_size=opt.batch_size, shuffle=False,
                              num_workers=4, worker_init_fn=worker_init_fn, pin_memory=True)
        for data_type, train_dataset in train_datasets.items()
    }
    for data_type in DATATYPE:
        print(f'{len(train_loaders[data_type].dataset)} images in train {data_type} set')

    # for val
    val_transform = transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    val_datasets = {
        data_type: dataset_cls(opt, phase='test', data_type=data_type, transform=val_transform)
        for data_type in DATATYPE
    }
    valid_num_samples = {"defects": opt.num_imgs,
                         "background": int(1e10)}
    val_samplers = {
        data_type: RandomSampler(val_datasets[data_type], replacement=False, num_samples=valid_num_samples[data_type])
        for data_type in DATATYPE
    }
    val_loaders = {
        data_type: DataLoader(val_dataset, sampler=val_samplers[data_type],
                              batch_size=opt.num_display_images, shuffle=False,
                              num_workers=4, worker_init_fn=worker_init_fn, pin_memory=True)
        for data_type, val_dataset in val_datasets.items()
    }
    for data_type in DATATYPE:
        print(f'{len(val_loaders[data_type].dataset)} images in val {data_type} set')

    # view_data_after_transform(opt, train_loaders)

    # change loader to iterator
    train_loaders['background'] = iter(train_loaders['background'])
    val_loaders['background'] = iter(val_loaders['background'])

    opt.iters_per_epoch = len(train_loaders['defects'])
    trainer = find_trainer_using_model_name(opt.model)(opt)
    trainer.train(train_loaders, val_loaders)


if __name__ == '__main__':
    fix_rand_seed()
    train()
    '''
    python train_mtvec.py --dataset_name mtvec --dataset_data_type pill --name mtvec --loss_weight 2 5 10 1 3 --phase train --add_noise --use_spectral --scheduler cos --num_iters 5_000 --save_latest_freq 200 --save_img_freq 1 --num_critics 3 --diff_aug color,translation,cutout
    python train_defectgan.py --name org_lw_wo_resize --continue_training --load_from_opt_file
    tensorboard --logdir log\test_df --samples_per_plugin "images=100"
    python train_defectgan.py --data_dir A:/research/data --name org_shrink --loss_weight 2 5 10 1 3 --npz_path A:/research/data/codebrim/val/defects00.npz --phase val --add_noise --use_spectral --scheduler cos --num_iters 20_000 --save_latest_freq 200
    --style_norm_block_type sean --embed_path results/vit_shrink/latest_train_fusion_embeddings.pth
    python train_defectgan.py --name org_mae_shrink_token_zero --data_dir A:/research/data --load_model_name mae_shrink_token_zero --loss_weight 2 5 10 1 3 --npz_path A:/research/data/codebrim/val/defects00.npz --phase val --add_noise --use_spectral --scheduler cos --num_iters 20_000 --save_latest_freq 200 --dataset_name codebrim_shrink
    '''
