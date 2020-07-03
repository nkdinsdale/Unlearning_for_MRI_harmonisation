# Nicola Dinsdale 2020
# Pytorch dataset for numpy arrays
########################################################################################################################
# Import dependencies
import torch
from torch.utils.data import Dataset
########################################################################################################################

class numpy_dataset(Dataset):  # Inherit from Dataset class
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).float()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)

class numpy_dataset_three(Dataset):  # Inherit from Dataset class
    def __init__(self, data, target, domain, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).float()
        self.domain = torch.from_numpy(domain).float()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        d = self.domain[index]

        if self.transform:
            x = self.transform(x)

        return x, y, d

    def __len__(self):
        return len(self.data)

class numpy_dataset_four(Dataset):  # Inherit from Dataset class
    def __init__(self, data, target, domain, sex, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).float()
        self.domain = torch.from_numpy(domain).float()
        self.sex = torch.from_numpy(sex).float()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        d = self.domain[index]
        s = self.sex[index]

        if self.transform:
            x = self.transform(x)

        return x, y, d, s

    def __len__(self):
        return len(self.data)