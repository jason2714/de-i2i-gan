import json

import torch
from torch.utils.data import Dataset
from PIL import Image
from data.codebrim.create_annos import create_annos
from functools import reduce


class CodeBrimDataset(Dataset):
    DATA_TYPE = ['defects', 'background']
    clf_loss_type = 'bce'

    def __init__(self, opt, phase, data_type, transform=None):
        """
            data_type must be defects or background
            phase must be train, val or test
        """
        assert data_type in CodeBrimDataset.DATA_TYPE or data_type == 'fusion', \
            'data_type must be defects, background or fusion'
        assert phase in ('train', 'val', 'test'), 'phase must be train, val or test'
        self.transform = transform

        # initialize filename to label map
        anno_dir = opt.data_dir / opt.dataset_name / 'metadata'

        data_types = CodeBrimDataset.DATA_TYPE if data_type == 'fusion' else [data_type]
        fn_label_map = {}
        for crt_data_type in data_types:
            anno_path = anno_dir / f'{crt_data_type}.json'
            if not anno_path.exists():
                create_annos(anno_dir)
            fn_label_map = {**fn_label_map, **(json.loads(anno_path.read_text()))}
        # initialize label to index map
        label2idx_path = anno_dir / 'label2idx.json'
        self.label2idx = json.loads(label2idx_path.read_text())

        # initialize data with format list of (filename, label)
        data_dirs = [opt.data_dir / opt.dataset_name / phase / crt_data_type for crt_data_type in data_types]
        # data_dirs = [opt.data_dir / opt.dataset_name / phase / crt_data_type for crt_data_type in data_types]
        self.data = [(filename, fn_label_map[filename.name])
                     for data_dir in data_dirs for filename in data_dir.iterdir() if filename.suffix == '.png']
        # if filter_label is not None:
        #     self.data = [data for data in self.data if tuple(data[1]) == tuple(filter_label)]
        self.data.sort()
        self.len = len(self.data)

    def __getitem__(self, index):
        image_fn, label = self.data[index]
        image = Image.open(image_fn)
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label), str(image_fn)

    def __len__(self):
        return self.len
