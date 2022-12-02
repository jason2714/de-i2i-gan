import importlib
from torch.utils.data import Dataset


def find_dataset_using_name(dataset_name):
    # Given the option --dataset [datasetname],
    # the file "datasets/datasetname_dataset.py"
    # will be imported.
    dataset_filename = "datasets." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetnameDataset() will
    # be instantiated. It has to be a subclass of torch.utils.data.Dataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
                and issubclass(cls, Dataset):
            dataset = cls

    if dataset is None:
        print(
            "In %s.py, there should be a subclass of torch.utils.data.Dataset "
            "with class name that matches %s in lowercase." % (
                dataset_filename, target_dataset_name))
        exit(0)

    return dataset
