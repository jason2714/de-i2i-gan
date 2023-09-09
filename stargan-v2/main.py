"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import argparse

from munch import Munch
from torch.backends import cudnn
import torch

from core.data_loader import get_train_loader, get_style_loader
from core.data_loader import get_test_loader
from core.solver import Solver
from pathlib import Path


def str2bool(v):
    return v.lower() in ('true')


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]


def main(args):
    print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    solver = Solver(args)
    if args.mode == 'train':
        assert len(subdirs(args.train_img_dir)) == args.num_domains
        assert len(subdirs(args.val_img_dir)) == args.num_domains
        if args.norm_type == 'adain':
            ref_loader = get_train_loader(root=args.train_img_dir,
                                          which='reference',
                                          img_size=args.img_size,
                                          batch_size=args.batch_size,
                                          prob=args.randcrop_prob,
                                          num_workers=args.num_workers)

        elif args.norm_type == 'sean':
            ref_loader = get_style_loader(root=args.train_img_dir,
                                          num_embeds=args.num_embeds,
                                          batch_size=args.batch_size,
                                          num_workers=args.num_workers)
        else:
            raise NotImplementedError
        loaders = Munch(src=get_train_loader(root=args.train_img_dir,
                                             which='source',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers),
                        ref=ref_loader,
                        val=get_test_loader(root=args.val_img_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers),
                        val2=get_test_loader(root=args.val_img_dir,
                                             img_size=args.img_size,
                                             batch_size=args.val_batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers,
                                             num_ref=args.num_val_refs))
        solver.train(loaders)
    elif args.mode == 'pretrain':
        assert len(subdirs(args.train_img_dir)) == args.num_domains
        assert len(subdirs(args.val_img_dir)) == args.num_domains
        # if args.norm_type == 'adain':
        #     ref_loader = get_train_loader(root=args.train_img_dir,
        #                                   which='reference',
        #                                   img_size=args.img_size,
        #                                   batch_size=args.batch_size,
        #                                   prob=args.randcrop_prob,
        #                                   num_workers=args.num_workers)
        #
        # elif args.norm_type == 'sean':
        #     ref_loader = get_style_loader(root=args.train_img_dir,
        #                                   num_embeds=args.num_embeds,
        #                                   batch_size=args.batch_size,
        #                                   num_workers=args.num_workers)
        # else:
        #     raise NotImplementedError
        ref_loader = get_train_loader(root=args.train_img_dir,
                                      which='reference',
                                      img_size=args.img_size,
                                      batch_size=args.batch_size,
                                      prob=args.randcrop_prob,
                                      num_workers=args.num_workers)
        loaders = Munch(train=get_train_loader(root=args.train_img_dir,
                                               which='source',
                                               img_size=args.img_size,
                                               batch_size=args.batch_size,
                                               prob=args.randcrop_prob,
                                               num_workers=args.num_workers),
                        ref=ref_loader,
                        val=get_test_loader(root=args.val_img_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers))
        solver.pretrain(loaders)
    elif args.mode == 'sample':
        assert len(subdirs(args.src_dir)) == args.num_domains
        assert len(subdirs(args.ref_dir)) == args.num_domains
        loaders = Munch(src=get_test_loader(root=args.src_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers),
                        ref=get_test_loader(root=args.ref_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers,
                                            num_ref=0))
        solver.sample(loaders)
    elif args.mode == 'eval':
        solver.evaluate()
    elif args.mode == 'update_stats':
        assert args.norm_type == 'sean', 'Only SEAN needs to update stats'
        loaders = Munch(src=get_train_loader(root=args.val_img_dir,
                                             which='source',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers),
                        ref=get_style_loader(root=args.val_img_dir,
                                             num_embeds=args.num_embeds,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers))
        solver.update_sean_stats(loaders)
    elif args.mode == 'align':
        from core.wing import align_faces
        align_faces(args, args.inp_dir, args.out_dir)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')
    parser.add_argument('--norm_type', type=str, default='adain', help='[adain | sean]')
    # for sean blocks
    parser.add_argument('--num_embeds', type=int, default=5, help='Number of embeddings for SEAN')
    parser.add_argument('--hidden_nc', type=int, default=256, help='Number of hidden channels for SEAN')
    parser.add_argument('--embed_nc', type=int, default=768, help='Number of embedding channels for SEAN')
    parser.add_argument('--num_val_refs', type=int, default=4, help='Number of reference images for validation')
    # for mae
    parser.add_argument('--patch_size', type=int, default=32, help='Patch size for masked region')
    parser.add_argument('--mask_ratio', type=float, default=0.65, help='Mask ratio for masked region')
    parser.add_argument('--mask_token_type', type=str, default='position',
                        help='type of mask token [zero|mean|scalar|vector|position|full]')
    parser.add_argument('--lambda_rec', type=float, default=10,
                        help='Weight for reconstruction loss')
    parser.add_argument('--pretrain_iter', type=int, default=None,
                        help='Number of loading pretrain iterations for training')
    parser.add_argument('--pretrain_dir', type=Path, default=None,
                        help='Directory for pretrain network checkpoints')

    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')
    parser.add_argument('--lambda_cyc', type=float, default=1,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sty', type=float, default=1,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=1,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--ds_iter', type=int, default=100000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--w_hpf', type=float, default=1,
                        help='weight for high-pass filtering')

    # training arguments
    parser.add_argument('--randcrop_prob', type=float, default=0.5,
                        help='Probabilty of using random-resized cropping')
    parser.add_argument('--total_iters', type=int, default=40_000,
                        help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=0,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for F')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=10,
                        help='Number of generated images per domain during sampling')
    parser.add_argument('--DiffAugment', help='Comma-separated list of DiffAugment policy', default='')

    # misc
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'sample', 'eval', 'align', 'update_stats', 'pretrain'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # directory for training
    parser.add_argument('--train_img_dir', type=str, default='data/celeba_hq/train',
                        help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default='data/celeba_hq/val',
                        help='Directory containing validation images')
    parser.add_argument('--sample_dir', type=str, default='expr/samples',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=Path, default='expr/checkpoints',
                        help='Directory for saving network checkpoints')

    # directory for calculating metrics
    parser.add_argument('--eval_dir', type=str, default='expr/eval',
                        help='Directory for saving metrics, i.e., FID and LPIPS')

    # directory for testing
    parser.add_argument('--result_dir', type=Path, default='expr/results',
                        help='Directory for saving generated images and videos')
    parser.add_argument('--src_dir', type=str, default='assets/representative/celeba_hq/src',
                        help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='assets/representative/celeba_hq/ref',
                        help='Directory containing input reference images')
    parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female',
                        help='input directory when aligning faces')
    parser.add_argument('--out_dir', type=str, default='assets/representative/celeba_hq/src/female',
                        help='output directory when aligning faces')

    # face alignment
    parser.add_argument('--wing_path', type=str, default='expr/checkpoints/wing.ckpt')
    parser.add_argument('--lm_path', type=str, default='expr/checkpoints/celeba_lm_mean.npz')

    # step size
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=5000)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=50000)
    parser.add_argument('--update_sean_every', type=int, default=10, help='update SEAN every n iterations')

    args = parser.parse_args()
    main(args)

'''
python main.py --mode pretrain --num_domains 3 --w_hpf 0 --lambda_reg 1 --lambda_sty 1 --lambda_ds 2 --lambda_cyc 0 --train_img_dir data/afhq_shrink/train --val_img_dir data/afhq_shrink/test --num_workers 0 --total_iters 40000 --norm_type sean --sample_every 500 --save_every 10000 --num_embeds 1
python main.py --mode train --num_domains 3 --w_hpf 0 --lambda_reg 1 --lambda_sty 1 --lambda_ds 2 --lambda_cyc 1 --train_img_dir data/afhq_shrink/train --val_img_dir data/afhq_shrink/test --num_workers 0 --total_iters 100000 --norm_type sean --sample_every 1000 --eval_every 10000 --save_every 10000 --num_embeds 5 --print_every 1000
--DiffAugment color,translation,cutout
python main.py --mode train --num_domains 3 --w_hpf 0 --lambda_reg 1 --lambda_sty 1 --lambda_ds 2 --lambda_cyc 1 --train_img_dir data/afhq_shrink/train --val_img_dir data/afhq_shrink/test --num_workers 0 --total_iters 100000 --norm_type sean --eval_every 10000 --save_every 10000 --num_embeds 2
python main.py --mode train --num_domains 3 --w_hpf 0 --lambda_reg 1 --lambda_sty 1 --lambda_ds 2 --lambda_cyc 1 --train_img_dir data/afhq_shrink/train --val_img_dir data/afhq_shrink/test --num_workers 0 --total_iters 40000
python main.py --mode eval --num_domains 3 --w_hpf 0 --resume_iter 20000 --train_img_dir data/afhq_shrink/train --val_img_dir data/afhq_shrink/test --checkpoint_dir expr/checkpoints --eval_dir expr/eval --num_workers 0
python main.py --mode sample --num_domains 3 --resume_iter 80000 --w_hpf 0 --checkpoint_dir expr/checkpoints --result_dir expr/results/afhq --src_dir assets/representative/afhq/src --ref_dir assets/representative/afhq/ref --num_workers 0

celeba
python main.py --mode train --num_domains 2 --w_hpf 1 --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 --train_img_dir data/celeba_hq_shrink/train  --val_img_dir data/celeba_hq_shrink/val --num_workers 0 --total_iters 100000 --norm_type sean --num_embeds 1 --hidden_nc 64 --sample_every 1000 --eval_every 10000 --save_every 10000 --print_every 1000
python main.py --mode pretrain --num_domains 2 --w_hpf 1 --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 0 --train_img_dir data/celeba_hq_shrink/train  --val_img_dir data/celeba_hq_shrink/val --num_workers 0 --total_iters 40000 --norm_type sean --num_embeds 1 --hidden_nc 64 --sample_every 1000 --save_every 5000 --print_every 1000
'''
