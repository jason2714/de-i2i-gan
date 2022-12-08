import json

from torch.utils.data import Dataset
from PIL import Image


class CodeBrimDataset(Dataset):

    def __init__(self, opt, phase, data_type, transform=None):
        """
            data_type must be defects or background
            phase must be train, val or test
        """
        assert data_type in ('defects', 'background'), 'data_type must be defects or background'
        assert phase in ('train', 'val', 'test'), 'phase must be train, val or test'
        self.transform = transform

        # initialize filename to label map
        anno_dir = opt.data_dir / opt.dataset_name / 'metadata'
        anno_path = anno_dir / f'{data_type}.json'
        fn_label_map = json.loads(anno_path.read_text())

        # initialize label to index map
        label2idx_path = anno_dir / 'label2idx.json'
        self.label2idx = json.loads(label2idx_path.read_text())

        # initialize data with format list of (filename, label)
        data_dir = opt.data_dir / opt.dataset_name / phase / data_type
        self.data = [(filename, fn_label_map[filename])
                     for filename in data_dir.iterdir() if filename.suffix == '.png']
        self.data.sort()
        self.len = len(self.data)

    def __getitem__(self, index):
        image_fn, label = self.data[index]
        image = Image.open(image_fn)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return self.len
