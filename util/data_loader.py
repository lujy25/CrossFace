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
        for id, single_df in self._df.groupby(by=['class']):
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

class TripletDataset(Dataset):
    def __init__(self, root_dir, csv_name, load_img=False, transform=None):
        self._root_dir = root_dir
        self._df = pd.read_csv(csv_name)
        self._transform = transform
        self._triplets = []
        self._analyze_df(self._df)

    def _analyze_df(self, df):
        self._class_path = dict()
        df['pose'] = df['pose'].astype(int)
        self._frontal_class_path = dict()
        self._profile_class_path = dict()
        self._middle_class_path = dict()
        self._total_class_path = dict()
        for face_class, face_paths in df.groupby(by=['class']):
            if len(face_paths[face_paths['pose'] == 0]) > 0:
                self._frontal_class_path[face_class] = face_paths[face_paths['pose'] == 0]['path'].tolist()
            if len(face_paths[face_paths['pose'] == 1]) > 0:
                self._profile_class_path[face_class] = face_paths[face_paths['pose'] == 1]['path'].tolist()
            if len(face_paths[face_paths['pose'] == 2]) > 0:
                self._middle_class_path[face_class] = face_paths[face_paths['pose'] == 2]['path'].tolist()
            self._total_class_path[face_class] = face_paths['path'].tolist()
        self._classes = df['class'].unique()

    def _sample_triplet_class(self, pos_classes, neg_classes):
        pos_classes = list(pos_classes)
        neg_classes = list(neg_classes)
        pos_class, neg_class = np.random.choice(neg_classes, 2, replace=False)
        while pos_class not in pos_classes:
            pos_class, neg_class = np.random.choice(neg_classes, 2, replace=False)
        return pos_class, neg_class

    def _sample_pair_class(self, anc_classes, trans_classes):
        anc_classes = list(anc_classes)
        trans_classes = list(trans_classes)
        trans_class = np.random.choice(trans_classes, 1, replace=False)[0]
        while trans_class not in anc_classes:
            trans_class = np.random.choice(trans_classes, 1, replace=False)[0]
        return trans_class

    def _sample_pair(self, sample_type):
        if sample_type == Sample_Type.Frontal2Profile:
            trans_class = self._sample_pair_class(anc_classes=self._frontal_class_path.keys(),
                                                  trans_classes=self._profile_class_path.keys())
            ianc = np.random.randint(0, len(self._frontal_class_path[trans_class]))
            itrans = np.random.randint(0, len(self._profile_class_path[trans_class]))
            anc_path = self._frontal_class_path[trans_class][ianc]
            trans_path = self._profile_class_path[trans_class][itrans]
        else:
            assert sample_type == Sample_Type.Frontal2Middle
            trans_class = self._sample_pair_class(anc_classes=self._frontal_class_path.keys(),
                                                  trans_classes=self._middle_class_path.keys())
            ianc = np.random.randint(0, len(self._frontal_class_path[trans_class]))
            itrans = np.random.randint(0, len(self._middle_class_path[trans_class]))
            anc_path = self._frontal_class_path[trans_class][ianc]
            trans_path = self._middle_class_path[trans_class][itrans]
        # else:
        #     assert sample_type == Sample_Type.Profile2Middle
        #     trans_class = self._sample_pair_class(anc_classes=self._profile_class_path.keys(),
        #                                           trans_classes=self._middle_class_path.keys())
        #     ianc = np.random.randint(0, len(self._profile_class_path[trans_class]))
        #     itrans = np.random.randint(0, len(self._middle_class_path[trans_class]))
        #     anc_path = self._profile_class_path[trans_class][ianc]
        #     trans_path = self._middle_class_path[trans_class][itrans]
        return anc_path, trans_path, trans_class

    def _sample_triplet(self, sample_type):
        if sample_type == Sample_Type.Frontal2Frontal:
            pos_class, neg_class = self._sample_triplet_class(pos_classes=self._frontal_class_path.keys(),
                                                      neg_classes=self._frontal_class_path.keys())
            ianc, ipos = np.random.choice(range(
                len(self._frontal_class_path[pos_class])), size=2, replace=False)
            ineg = np.random.randint(0, len(self._frontal_class_path[neg_class]))
            anc_path = self._frontal_class_path[pos_class][ianc]
            pos_path = self._frontal_class_path[pos_class][ipos]
            neg_path = self._frontal_class_path[neg_class][ineg]
            anc_pose = 0
            pos_pose = 0
            neg_pose = 0
        elif sample_type == Sample_Type.Profile2Profile:
            pos_class, neg_class = self._sample_triplet_class(pos_classes=self._profile_class_path.keys(),
                                                      neg_classes=self._profile_class_path.keys())
            ianc, ipos = np.random.choice(range(
                len(self._profile_class_path[pos_class])), size=2, replace=False)
            ineg = np.random.randint(0, len(self._profile_class_path[neg_class]))
            anc_path = self._profile_class_path[pos_class][ianc]
            pos_path = self._profile_class_path[pos_class][ipos]
            neg_path = self._profile_class_path[neg_class][ineg]
            anc_pose = 1
            pos_pose = 1
            neg_pose = 1
        elif sample_type == Sample_Type.Middle2Middle:
            pos_class, neg_class = self._sample_triplet_class(pos_classes=self._middle_class_path.keys(),
                                                      neg_classes=self._middle_class_path.keys())
            ianc, ipos = np.random.choice(range(
                len(self._middle_class_path[pos_class])), size=2, replace=False)
            ineg = np.random.randint(0, len(self._middle_class_path[neg_class]))
            anc_path = self._middle_class_path[pos_class][ianc]
            pos_path = self._middle_class_path[pos_class][ipos]
            neg_path = self._middle_class_path[neg_class][ineg]
            anc_pose = 2
            pos_pose = 2
            neg_pose = 2
        elif sample_type == Sample_Type.Frontal2Profile:
            pos_class, neg_class = self._sample_triplet_class(pos_classes=self._frontal_class_path.keys(),
                                                      neg_classes=self._profile_class_path.keys())
            ianc = np.random.randint(0, len(self._frontal_class_path[pos_class]))
            ipos = np.random.randint(0, len(self._profile_class_path[pos_class]))
            ineg = np.random.randint(0, len(self._profile_class_path[neg_class]))
            anc_path = self._frontal_class_path[pos_class][ianc]
            pos_path = self._profile_class_path[pos_class][ipos]
            neg_path = self._profile_class_path[neg_class][ineg]
            anc_pose = 0
            pos_pose = 1
            neg_pose = 1
        elif sample_type == Sample_Type.Frontal2Middle:
            pos_class, neg_class = self._sample_triplet_class(pos_classes=self._frontal_class_path.keys(),
                                                      neg_classes=self._middle_class_path.keys())
            ianc = np.random.randint(0, len(self._frontal_class_path[pos_class]))
            ipos = np.random.randint(0, len(self._middle_class_path[pos_class]))
            ineg = np.random.randint(0, len(self._middle_class_path[neg_class]))
            anc_path = self._frontal_class_path[pos_class][ianc]
            pos_path = self._middle_class_path[pos_class][ipos]
            neg_path = self._middle_class_path[neg_class][ineg]
            anc_pose = 0
            pos_pose = 2
            neg_pose = 2
        elif sample_type == Sample_Type.Profile2Middle:
            pos_class, neg_class = self._sample_triplet_class(pos_classes=self._profile_class_path.keys(),
                                                      neg_classes=self._middle_class_path.keys())
            ianc = np.random.randint(0, len(self._profile_class_path[pos_class]))
            ipos = np.random.randint(0, len(self._middle_class_path[pos_class]))
            ineg = np.random.randint(0, len(self._middle_class_path[neg_class]))
            anc_path = self._profile_class_path[pos_class][ianc]
            pos_path = self._middle_class_path[pos_class][ipos]
            neg_path = self._middle_class_path[neg_class][ineg]
            anc_pose = 1
            pos_pose = 2
            neg_pose = 2
        else:
            assert sample_type == Sample_Type.All
            pos_class, neg_class = self._sample_triplet_class(pos_classes=self._total_class_path.keys(),
                                                      neg_classes=self._total_class_path.keys())
            ianc, ipos = np.random.choice(range(len(self._total_class_path[pos_class])), size=2, replace=False)
            ineg = np.random.randint(0, len(self._total_class_path[neg_class]))
            anc_path = self._total_class_path[pos_class][ianc]
            pos_path = self._total_class_path[pos_class][ipos]
            neg_path = self._total_class_path[neg_class][ineg]
            anc_pose = 0
            pos_pose = 0
            neg_pose = 0
        return anc_path, pos_path, neg_path, pos_class, neg_class, anc_pose, pos_pose, neg_pose