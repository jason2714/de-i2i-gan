from pathlib import Path
import argparse
import shutil
import random

DATA_SPLITS = ('train', 'val', 'test')
SHRINK_TYPES = ['train']
DATA_TYPES = ['background', 'defects']

def fix_rand_seed(seed=123):
    # Python built-in random module
    random.seed(seed)

def arg_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=Path, default='./data/codebrim')
    parser.add_argument('--shrink_ratio', type=float, default=0.1)
    args = parser.parse_args()
    return args


def shrink_data(args):
    new_dir = args.data_dir.parents[0] / f'{args.data_dir.stem}_shrink'

    for img_dir in args.data_dir.iterdir():
        if img_dir.name in DATA_SPLITS:
            for data_type in DATA_TYPES:
                img_type_dir = img_dir / data_type
                img_paths = [
                    img_path for img_path in img_type_dir.iterdir()
                    if img_path.suffix.lower() in ('.jpg', '.png')
                ]
                org_num_imgs = len(img_paths)
                if img_dir.name in SHRINK_TYPES:
                    num_imgs = int(org_num_imgs * args.shrink_ratio)
                    img_paths = random.choices(img_paths, k=num_imgs)
                new_base_dir = new_dir / img_dir.name / data_type
                new_base_dir.mkdir(parents=True, exist_ok=True)
                for img_path in img_paths:
                    shutil.copy(img_path, new_base_dir / img_path.name)
                print(f'img_dir: {img_dir.name}_{data_type}, org_img_cnt: {org_num_imgs}, img_cnt: {len(img_paths)}')


if __name__ == '__main__':
    fix_rand_seed(17)
    shrink_data(arg_parse())
    """
    python data/shrink_data.py --data_dir A:/research/data/codebrim
    """
