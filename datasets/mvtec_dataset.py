import torch
from torch.utils.data import Dataset
from PIL import Image


class MTVecDataset(Dataset):
    DATA_TYPE = ['defects', 'background']
    clf_loss_type = 'cce'

    def __init__(self, opt, phase, data_type, transform=None):
        """
            data_type must be defects or background
            phase must be train or test
        """
        assert data_type in MTVecDataset.DATA_TYPE or data_type == 'fusion', \
            'data_type must be defects, background or fusion'
        assert phase in ('train', 'val', 'test'), 'phase must be train, val or test'
        assert opt.dataset_data_type is not None, 'dataset_data_type must be specified, e.g. pill, capsule, etc.'
        self.transform = transform

        data_dir = opt.data_dir / opt.dataset_name / opt.dataset_data_type / phase
        labels = [label_path.name for label_path in data_dir.iterdir()]
        labels.sort(key=lambda x: (x != 'normal', x))
        one_hot_labels = torch.eye(len(labels)).tolist()
        self.label2idx = {label: one_hot_label for label, one_hot_label in zip(labels, one_hot_labels)}

        # initialize data with format list of (filename, label)
        data_dirs = []
        if data_type in ('background', 'fusion'):
            data_dirs += [data_dir / 'normal']
        if data_type in ('defects', 'fusion'):
            data_dirs += [data_dir / crt_data_type for crt_data_type in labels if crt_data_type != 'normal']
        self.data = [(filename, self.label2idx[data_dir.name])
                     for data_dir in data_dirs for filename in data_dir.iterdir() if filename.suffix == '.png']
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
