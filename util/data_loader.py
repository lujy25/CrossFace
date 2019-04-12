import os
import numpy as np
import pandas as pd
from skimage import io
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from enum import Enum
import random

class Sample_Type(Enum):
    Frontal2Frontal = 1
    Profile2Profile = 2
    Middle2Middle = 3
    Frontal2Profile = 4
    Frontal2Middle = 5
    Profile2Middle = 6
    All = 7

class Pose_Type(Enum):
    Frontal = 1
    Profile = 2
    Middle = 3
    All = 4

class SampleDataset(Dataset):
    def __init__(self, root_dir, csv_name,  transform=None):
        self._root_dir = root_dir
        self._df = pd.read_csv(csv_name)
        self._transform = transform
        self._analyze_df(self._df)

    def _analyze_df(self, df):
        self._sample_weights = []
        self._sample_faces = []
        self._classes = []
        for id, single_df in self._df.groupby(by=['subject_id']):
            sample_weight = len(self._df) / len(single_df)
            sample_weights = [sample_weight] * len(single_df)
            self._sample_weights.extend(sample_weights)
            for index in single_df.index:
                face_path, face_yaw = single_df.ix[index, ['file', 'yaw']]
                self._sample_faces.append([id, face_path, face_yaw])
            self._classes.append(id)

    def get_class_num(self):
        return len(self._classes)

    def get_samle_weight(self):
        return self._sample_weights

    def __getitem__(self, idx):
        face_class, face_path, face_yaw = self._sample_faces[idx]
        img_path = os.path.join(self._root_dir, str(face_path))
        face_img = io.imread(img_path)
        sample = {'face_path': face_path, 'face_img': face_img, 'face_class': face_class, 'face_yaw': face_yaw}
        if self._transform:
            sample['face_img'] = self._transform(sample['face_img'])
        return sample

    def __len__(self):
        return len(self._sample_faces)


def get_train_face_extraction_dataloader(root_dir, csv_name, batch_size, num_workers):

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([128, 128]),
        transforms.RandomCrop([112, 112]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    )

    dataset = SampleDataset(
        root_dir=root_dir,
        csv_name=csv_name,
        transform=transform
    )
    sampler = WeightedRandomSampler(
        dataset.get_samle_weight(),
        len(dataset.get_samle_weight()),
        replacement=True
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
    return dataset, dataloader
