import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset

class PoseDataset(Dataset):
    def __init__(self):
        self._data = range(10)

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


samples_weight = [0.1] * 6
sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

train_dataset = PoseDataset()
train_loader = DataLoader(
    train_dataset, batch_size=10, num_workers=1, sampler=sampler)

for i, sample in enumerate(train_loader):
    print("Add at shell")
    print(sample)