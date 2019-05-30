import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np


__all__ = ['Cifar100Dataset']

class Cifar100Dataset(Dataset):
    def __init__(self, root, download=False, train=True, transform=None):

        data = torchvision.datasets.CIFAR100(root=root,train=train, download=download)
        # n x 32 x 32 x 3 (uint8, np.array)
        self.images = data.data
        # n (list)
        self.labels = data.targets
        self.transform = transform
        self.train = train

    def __getitem__(self, index):
        if self.train:
            image = self.images[index]
            target = np.zeros((100),dtype=np.float32)
            target[self.labels[index]] = 1.0

            if self.transform is not None:
                image = self.transform(image)

        else:
            image = self.images[index]
            target = np.zeros((100), dtype=np.float32)
            target[self.labels[index]] = 1.0
            if self.transform is not None:
                image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.images)