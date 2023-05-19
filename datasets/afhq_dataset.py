import torch
from torch.utils.data import Dataset
from PIL import Image


class AFHQDataset(Dataset):
    LABEL2IDX = {
        'cat': 0,
        'dog': 1,
        'wild': 2
    }
    clf_loss_type = 'cce'

    def __init__(self, opt, phase, transform=None):
        """
            data_type must be defects or background
            phase must be train or test
        """
        assert phase in ('train', 'test'), 'phase must be train or test'
        self.transform = transform

        # initialize data with format list of (filename, label)
        data_dirs = [opt.data_dir / opt.dataset_name / phase / label for label in self.LABEL2IDX.keys()]
        self.data = [(filename, self.LABEL2IDX[data_dir.name])
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
