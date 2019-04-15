# import numpy as np
# import argparse
# import torch
# from util.models import *
from torch.nn.modules.distance import PairwiseDistance
# from tqdm import tqdm
# # from util.data_loader import *
# import dlib
# import cv2
# import os
# import torch
# import pandas as pd
# from util.little_block import *
# from util.evaluate import *
# analyze_fold = './analyze_data'
# shape_predict = dlib.shape_predictor('model/dlib/shape_predictor_68_face_landmarks.dat')
# facerec = dlib.face_recognition_model_v1('model/dlib/dlib_face_recognition_resnet_model_v1.dat')

# parser = argparse.ArgumentParser(
#     description='Face Recognition using Triplet Loss')
# parser.add_argument('--num-triplets', default=10000,
#                     type=int, metavar='NTT',
#                     help='number of triplets for evaluating (default: 10000)')
# parser.add_argument('--batch-size', default=16, type=int, metavar='BS',
#                     help='batch size (default: 128)')
# parser.add_argument('--num-workers', default=0, type=int, metavar='NW',
#                     help='number of workers (default: 8)')
# parser.add_argument('--root-dir', default='./datasets', type=str,
#                     help='path to train root dir')
# parser.add_argument('--csv-name', default='./xls_csv/test_IJB.csv', type=str,
#                     help='list of training images')
#
# args = parser.parse_args()
#device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
l2_dist = PairwiseDistance(2)

import cv2
import dlib
import numpy as np
shape_predict = dlib.shape_predictor('model/dlib/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('model/dlib/dlib_face_recognition_resnet_model_v1.dat')

def cal_embed(file_path):
    img = cv2.imread(file_path)
    rec = dlib.rectangle(0, 0, img.shape[1], img.shape[0])
    print(rec)
    shape = shape_predict(img, rec)
    print(shape)
    print(facerec)
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    return np.array([elem for elem in face_descriptor])

if __name__ == "__main__":
    img_path = './datasets/19/img_1373.jpg'
    print(cal_embed(img_path))