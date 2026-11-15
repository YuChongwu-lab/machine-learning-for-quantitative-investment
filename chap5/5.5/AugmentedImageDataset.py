"""Dataset and DataLoader with data augmentation 

The book's Section 5.5 experiments go through ``data_utils.create_dataloader``
(no augmentation). This file provides an optional augmented version for
readers to experiment with: random horizontal flip, random rotation
(+/-10 deg), random translation (+/-10%).

Usage: replace ``from data_utils import create_dataloader`` with
``from AugmentedImageDataset import create_dataloader`` in an experiment,
passing ``augment=True``. Note that this changes the RNG stream, so
results will differ from the book.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

#Data augmentation
class AugmentedImageDataset(Dataset):
    """Data augmentation: random horizontal flip, random rotation (+/-10 deg), random translation (+/-10%)"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(torch.tensor(image))
        return image.astype(np.float32), label.astype(np.int64)

def create_dataloader(images, labels, batch_size=32, shuffle=True, augment=False):
    if augment:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
        ])
    else:
        transform = None
    dataset = AugmentedImageDataset(images, labels, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
