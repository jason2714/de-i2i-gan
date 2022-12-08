from torch.utils.data import Dataset
from PIL import Image


class CodeBrimDataset(Dataset):

    def __init__(self, opt, phase, transform=None):
        self.transform = transform
        input_path = opt.data_dir / opt.dataset_name / phase
        self.filenames = [filename for filename in input_path.iterdir() if filename.suffix == '.png']
        self.filenames.sort()
        self.len = len(self.filenames)

    def __getitem__(self, index):
        image_fn = self.filenames[index]
        image = Image.open(image_fn)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return self.len
