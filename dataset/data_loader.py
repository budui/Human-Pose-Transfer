import torch
from torch.utils.data import DataLoader


class RandomSampleDataLoader(object):
    def __init__(self, data_loader, random_dataset_size=4000, random_select=True):
        if not isinstance(data_loader, DataLoader):
            raise TypeError("need DataLoader")
        self.data_loader = data_loader
        self.real_dataset_size = len(data_loader.dataset)
        self.random_dataset_size = random_dataset_size
        self.shuffle_indices = torch.randperm(self.real_dataset_size)[:self.random_dataset_size]

    def reset(self):
        self.shuffle_indices = torch.randperm(self.real_dataset_size)[:self.random_dataset_size]
