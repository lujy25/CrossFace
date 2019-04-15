import numpy as np
import argparse
import torch
from util.models import *
from torch.nn.modules.distance import PairwiseDistance
from tqdm import tqdm
from util.data_loader import *
import os
import torch
import pandas as pd
from util.little_block import *
from util.evaluate import *
analyze_fold = './analyze_data'


parser = argparse.ArgumentParser(
    description='Face Recognition using Triplet Loss')
parser.add_argument('--num-triplets', default=10000,
                    type=int, metavar='NTT',
                    help='number of triplets for evaluating (default: 10000)')
parser.add_argument('--batch-size', default=16, type=int, metavar='BS',
                    help='batch size (default: 128)')
parser.add_argument('--num-workers', default=0, type=int, metavar='NW',
                    help='number of workers (default: 8)')
parser.add_argument('--root-dir', default='./datasets', type=str,
                    help='path to train root dir')
parser.add_argument('--csv-name', default='./xls_csv/test_IJB.csv', type=str,
                    help='list of training images')

args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
l2_dist = PairwiseDistance(2)


def main(save_path):
    valid_dataset, valid_dataloader = get_valid_face_extraction_dataloader(root_dir=args.root_dir,
                                                                           csv_name=args.csv_name,
                                                                           batch_size=args.batch_size,
                                                                           num_workers=args.num_workers,
                                                                           num_triplet=args.num_triplets,
                                                                           load_img=False)
    print(80 * '=')
    for pose_type in [Pose_Type.Frontal, Pose_Type.Profile, Pose_Type.Middle, Pose_Type.All]:
        valid(pose_type=pose_type, valid_dataset=valid_dataset, valid_dataloader=valid_dataloader, save_path=save_path)
    print(80 * '=')

def valid(pose_type, valid_dataset, valid_dataloader, save_path):
    valid_dataset.sample_triplets(pose_type=pose_type)
    distances, labels = [], []
    yaw_cross = []
    anc_paths = []
    compair_paths = []

    for batch_sample in tqdm(valid_dataloader):
        anc_yaw = batch_sample['anc_yaw'].data.cpu().numpy()
        pos_yaw = batch_sample['pos_yaw'].data.cpu().numpy()
        neg_yaw = batch_sample['neg_yaw'].data.cpu().numpy()

        anc_embed = batch_sample['anc_embed'].to(device)
        pos_embed = batch_sample['pos_embed'].to(device)
        neg_embed = batch_sample['neg_embed'].to(device)

        anc_path = batch_sample['anc_path']
        pos_path = batch_sample['pos_path']
        neg_path = batch_sample['neg_path']

        dists = l2_dist.forward(anc_embed, pos_embed)
        distances.append(dists.data.cpu().numpy())
        labels.append(np.ones(dists.size(0)))
        yaw_cross.append(np.abs(anc_yaw - pos_yaw))
        anc_paths.append(np.array(anc_path))
        compair_paths.append(np.array(pos_path))

        dists = l2_dist.forward(anc_embed, neg_embed)
        distances.append(dists.data.cpu().numpy())
        labels.append(np.zeros(dists.size(0)))
        yaw_cross.append(np.abs(anc_yaw - neg_yaw))
        anc_paths.append(np.array(anc_path))
        compair_paths.append(np.array(neg_path))


    print("---Calculate Accuracy---")
    with open('%s.txt' % save_path, 'a') as f:
        f.write("---Calculate Accuracy---\n")
        f.close()
    print("Valid in", pose_type)
    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])
    yaw_cross = np.array([subdist for dist in yaw_cross for subdist in dist])
    anc_paths = np.array([subdist for dist in anc_paths for subdist in dist])
    compair_paths = np.array([subdist for dist in compair_paths for subdist in dist])

    accuracy, best_threshold, tp_cross, fp_cross, tn_cross, fn_cross, fn_anc_paths, fn_compair_paths, fp_anc_paths, fp_compair_paths = \
        cal_threshold_accuracy_with_yaw(distances, labels, yaw_cross,
                                        anc_paths, compair_paths
                                        )
    if not os.path.exists(os.path.join(analyze_fold, str(pose_type))):
        os.makedirs(os.path.join(analyze_fold, str(pose_type)))
    print(np.mean(accuracy),  np.mean(best_threshold))
    df = pd.DataFrame()
    df['anc_paths'] = fn_anc_paths
    df['compair_paths'] = fn_compair_paths
    df.to_csv(os.path.join(analyze_fold, os.path.join(str(pose_type), "fn_path.csv")), index=False)

    df = pd.DataFrame()
    df['anc_paths'] = fp_anc_paths
    df['compair_paths'] = fp_compair_paths
    df.to_csv(os.path.join(analyze_fold, os.path.join(str(pose_type), "fp_path.csv")), index=False)

    df = pd.DataFrame()
    df['tp_cross'] = tp_cross
    df.to_csv(os.path.join(analyze_fold, os.path.join(str(pose_type), "tp_cross.csv")), index=False)

    df = pd.DataFrame()
    df['fp_cross'] = fp_cross
    df.to_csv(os.path.join(analyze_fold, os.path.join(str(pose_type), "fp_cross.csv")), index=False)

    df = pd.DataFrame()
    df['tn_cross'] = tn_cross
    df.to_csv(os.path.join(analyze_fold, os.path.join(str(pose_type), "tn_cross.csv")), index=False)

    df = pd.DataFrame()
    df['fn_cross'] = fn_cross
    df.to_csv(os.path.join(analyze_fold, os.path.join(str(pose_type), "fn_cross.csv")), index=False)

    with open('%s.txt' % save_path, 'a') as f:
        f.write("%s %s\n" % (str(pose_type),  str(np.mean(accuracy))))
        f.close()
    try:
        df = pd.read_csv(save_path + ".csv", index_col=0)
    except:
        df = pd.DataFrame()
    epoch = len(df)
    df.loc[epoch, '%s_Accuracy' % str(pose_type)] = np.mean(accuracy)


if __name__ == '__main__':
    main('analyze_data/evaluate_origin')
