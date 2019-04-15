import os
import numpy as np
import pandas as pd
from skimage import io
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from enum import Enum
import random
import cv2
import dlib

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

    def analyze_df(self, pose_type):
        if pose_type == Pose_Type.Frontal:
            df = self._df[self._df['pose'] == 0]
        elif pose_type == Pose_Type.Middle:
            df = self._df[self._df['pose'] == 1]
        elif pose_type == Pose_Type.Profile:
            df = self._df[self._df['pose'] == 2]
        else:
            df = self._df
        self._sample_weights = []
        self._sample_faces = []
        self._classes = []
        for id, single_df in df.groupby(by=['class']):
            sample_weight = len(df) / len(single_df)
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


def get_train_face_extraction_dataloader(root_dir, csv_name, batch_size, num_workers, pose_type):

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
    dataset.analyze_df(pose_type=pose_type)
    sampler = WeightedRandomSampler(
        dataset.get_samle_weight(),
        len(dataset.get_samle_weight()),
        replacement=True
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
    return dataset, dataloader

class TripletDataset(Dataset):
    def __init__(self, root_dir, csv_name, load_img, num_triplet=None, transform=None):
        self._root_dir = root_dir
        self._df = pd.read_csv(csv_name)
        self._transform = transform
        self._num_triplet = num_triplet
        self._triplets = []
        self._load_img = load_img

    def cal_embed(self, file_path):
        img = cv2.imread(file_path)
        rec = dlib.rectangle(0, 0, img.shape[1], img.shape[0])
        print(rec)
        shape = self._shape_predict(img, rec)
        print(shape)
        print(self._facerec)
        face_descriptor = self._facerec.compute_face_descriptor(img, shape)
        print(face_descriptor)
        return np.array([elem for elem in face_descriptor])

    def _analyze_df(self, pose_type):
        if pose_type == Pose_Type.Frontal:
            df = self._df[self._df['pose'] == 0]
        elif pose_type == Pose_Type.Middle:
            df = self._df[self._df['pose'] == 1]
        elif pose_type == Pose_Type.Profile:
            df = self._df[self._df['pose'] == 2]
        else:
            df = self._df
        self._class_path = dict()
        self._class_yaw = dict()
        self._classes = []
        for face_class, single_df in df.groupby(by=['class']):
            if len(single_df) < 2:
                continue
            self._class_path[face_class] = np.array(single_df['file'].tolist())
            self._class_yaw[face_class] = np.array(single_df['yaw'].tolist())
            self._classes.append(face_class)

    def _sample_triplet(self, pos_class):
        neg_class = np.random.choice(self._classes, 1, replace=False)[0]
        while neg_class == pos_class:
            neg_class = np.random.choice(self._classes, 1, replace=False)[0]
        anc_pos_index = np.random.choice(range(len(self._class_path[pos_class])), 2, replace=False)
        anc_path, pos_path = self._class_path[pos_class][anc_pos_index]
        anc_yaw, pos_yaw = self._class_yaw[pos_class][anc_pos_index]
        neg_index = np.random.choice(range(len(self._class_path[neg_class])), 1, replace=False)[0]
        neg_path = self._class_path[neg_class][neg_index]
        neg_yaw = self._class_yaw[neg_class][neg_index]
        return pos_class, neg_class, anc_path, pos_path, neg_path, anc_yaw, pos_yaw, neg_yaw

    def sample_triplets(self, pose_type=Pose_Type.All):
        self._analyze_df(pose_type)
        self._triplets = []
        for pos_class in self._classes:
            pos_class, neg_class, anc_path, pos_path, neg_path, anc_yaw, pos_yaw, neg_yaw = self._sample_triplet(pos_class)
            self._triplets.append([pos_class, neg_class, anc_path, pos_path, neg_path, anc_yaw, pos_yaw, neg_yaw])
        if self._num_triplet:
            sample_class = np.random.choice(self._classes, self._num_triplet, replace=True)
            for pos_class in sample_class:
                pos_class, neg_class, anc_path, pos_path, neg_path, anc_yaw, pos_yaw, neg_yaw = self._sample_triplet(
                    pos_class)
                self._triplets.append([pos_class, neg_class, anc_path, pos_path, neg_path, anc_yaw, pos_yaw, neg_yaw])


    def __getitem__(self, idx):
        pos_class, neg_class, anc_path, pos_path, neg_path, anc_yaw, pos_yaw, neg_yaw = self._triplets[idx]
        anc_img_path = os.path.join(self._root_dir, str(anc_path))
        pos_img_path = os.path.join(self._root_dir, str(pos_path))
        neg_img_path = os.path.join(self._root_dir, str(neg_path))
        if self._load_img:
            anc_face_img = io.imread(anc_img_path)
            pos_face_img = io.imread(pos_img_path)
            neg_face_img = io.imread(neg_img_path)
            sample = {'anc_path': anc_path, 'pos_path': pos_path, 'neg_path': neg_path,
                      'anc_yaw': abs(anc_yaw), 'pos_yaw': abs(pos_yaw), 'neg_yaw': abs(neg_yaw),
                      'anc_img': anc_face_img, 'pos_img': pos_face_img, 'neg_img': neg_face_img,
                      'pos_class': pos_class, 'neg_class': neg_class}
            if self._transform:
                sample['anc_img'] = self._transform(sample['anc_img'])
                sample['pos_img'] = self._transform(sample['pos_img'])
                sample['neg_img'] = self._transform(sample['neg_img'])
        else:
            sample = {'anc_img_path': anc_img_path, 'pos_img_path': pos_img_path, 'neg_img_path': neg_img_path,
                      'anc_path': anc_path, 'pos_path': pos_path, 'neg_path': neg_path,
                      'pos_class': pos_class, 'neg_class': neg_class,
                      'anc_yaw': abs(anc_yaw), 'pos_yaw': abs(pos_yaw), 'neg_yaw': abs(neg_yaw)}
        return sample

    def __len__(self):
        return len(self._triplets)


def get_valid_face_extraction_dataloader(root_dir, csv_name, batch_size, num_workers, load_img=True, num_triplet=None):

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([128, 128]),
        transforms.CenterCrop([112, 112]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    )

    dataset = TripletDataset(
        root_dir=root_dir,
        csv_name=csv_name,
        transform=transform,
        num_triplet=num_triplet,
        load_img=load_img,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return dataset, dataloader