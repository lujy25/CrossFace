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
                                                                           )#num_triplet=args.num_triplets)
    ArcFace = Backbone().to(device)
    ArcFace.load_state_dict(torch.load('./log/ArcFace-Origin/ArcFace-Origin_BACKBONE_checkpoint_epoch120.pth')['state_dict'])
    ArcFace.eval()

    CosFace = Backbone().to(device)
    CosFace.load_state_dict(torch.load('./log/CosFace-Origin/CosFace-Origin_BACKBONE_checkpoint_epoch120.pth')['state_dict'])
    CosFace.eval()

    ArcFrontal = Backbone().to(device)
    ArcFrontal.load_state_dict(torch.load('./log/ArcFace-Frontal-Origin/ArcFace-Frontal-Origin_BACKBONE_checkpoint_epoch120.pth')['state_dict'])
    ArcFrontal.eval()

    faceExtractionModel = [
        {
            'name': 'ArcFace',
            'model': ArcFace
        },
        {
            'name': 'CosFace',
            'model': CosFace
        },
        {
            'name': 'ArcFrontal',
            'model': ArcFrontal
        }
    ]
    print(80 * '=')
    for pose_type in [Pose_Type.Frontal, Pose_Type.Profile, Pose_Type.Middle, Pose_Type.All]:
        valid(pose_type=pose_type, valid_dataset=valid_dataset, valid_dataloader=valid_dataloader, faceExtractionModel=faceExtractionModel, save_path=save_path)
    print(80 * '=')

def valid(pose_type, valid_dataset, valid_dataloader, faceExtractionModel, save_path):
    valid_dataset.sample_triplets(pose_type=pose_type)
    model_labels = {}
    model_distances = {}
    model_yaw_cross = {}
    model_anc_paths = {}
    model_compair_paths = {}
    for index in range(0, len(faceExtractionModel)):
        model_name = faceExtractionModel[index]['name']
        exec('%s_labels, %s_distances, %s_yaw_cross = [], [], []' % (model_name, model_name, model_name))
        exec('%s_anc_paths, %s_compair_paths = [], []' % (model_name, model_name))
        exec('model_labels["%s"]=%s_labels' % (model_name, model_name))
        exec('model_distances["%s"]=%s_distances' % (model_name, model_name))
        exec('model_yaw_cross["%s"]=%s_yaw_cross' % (model_name, model_name))
        exec('model_anc_paths["%s"]=%s_anc_paths' % (model_name, model_name))
        exec('model_compair_paths["%s"]=%s_compair_paths' % (model_name, model_name))


    for batch_sample in tqdm(valid_dataloader):
        anc_yaw = batch_sample['anc_yaw'].data.cpu().numpy()
        pos_yaw = batch_sample['pos_yaw'].data.cpu().numpy()
        neg_yaw = batch_sample['neg_yaw'].data.cpu().numpy()

        anc_path = batch_sample['anc_path']
        pos_path = batch_sample['pos_path']
        neg_path = batch_sample['neg_path']

        anc_img = batch_sample['anc_img'].to(device)
        pos_img = batch_sample['pos_img'].to(device)
        neg_img = batch_sample['neg_img'].to(device)

        for index in range(0, len(faceExtractionModel)):
            model_name = faceExtractionModel[index]['name']
            model_net = faceExtractionModel[index]['model']
            anc_embed = l2_norm(model_net(anc_img))
            pos_embed = l2_norm(model_net(pos_img))
            neg_embed = l2_norm(model_net(neg_img))

            dists = l2_dist.forward(anc_embed, pos_embed)
            model_distances[model_name].append(dists.data.cpu().numpy())
            model_labels[model_name].append(np.ones(dists.size(0)))
            model_yaw_cross[model_name].append(np.abs(anc_yaw - pos_yaw))
            model_anc_paths[model_name].append(np.array(anc_path))
            model_compair_paths[model_name].append(np.array(pos_path))

            dists = l2_dist.forward(anc_embed, neg_embed)
            model_distances[model_name].append(dists.data.cpu().numpy())
            model_labels[model_name].append(np.zeros(dists.size(0)))
            model_yaw_cross[model_name].append(np.abs(anc_yaw - neg_yaw))
            model_anc_paths[model_name].append(np.array(anc_path))
            model_compair_paths[model_name].append(np.array(neg_path))


    print("---Calculate Accuracy---")
    with open('%s.txt' % save_path, 'a') as f:
        f.write("---Calculate Accuracy---\n")
        f.close()
    for index in range(0, len(faceExtractionModel)):
        model_name = faceExtractionModel[index]['name']
        model_labels[model_name] = np.array([sublabel for label in model_labels[model_name] for sublabel in label])
        model_distances[model_name] = np.array([subdist for dist in model_distances[model_name] for subdist in dist])
        model_yaw_cross[model_name] = np.array([subdist for dist in model_yaw_cross[model_name] for subdist in dist])
        model_anc_paths[model_name] = np.array([subdist for dist in model_anc_paths[model_name] for subdist in dist])
        model_compair_paths[model_name] = np.array([subdist for dist in model_compair_paths[model_name] for subdist in dist])

        accuracy, best_threshold, tp_cross, fp_cross, tn_cross, fn_cross, fn_anc_paths, fn_compair_paths, fp_anc_paths, fp_compair_paths = \
            cal_threshold_accuracy_with_yaw(model_distances[model_name], model_labels[model_name], model_yaw_cross[model_name],
                                            model_anc_paths[model_name], model_compair_paths[model_name]
                                            )
        if not os.path.exists(os.path.join(analyze_fold, str(pose_type))):
            os.makedirs(os.path.join(analyze_fold, str(pose_type)))
        print("Valid in", pose_type)
        print(len(fn_anc_paths), len(fp_anc_paths))
        print(model_name, np.mean(accuracy),  np.mean(tp_cross), np.mean(fp_cross), np.mean(fn_cross))
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
            f.write("%s %s %s\n" % (str(pose_type), model_name, str(np.mean(accuracy))))
            f.close()
        try:
            df = pd.read_csv(save_path + ".csv", index_col=0)
        except:
            df = pd.DataFrame()
        epoch = len(df)
        df.loc[epoch, '%s_%s_Accuracy' % (str(pose_type), model_name)] = np.mean(accuracy)


if __name__ == '__main__':
    main('analyze_data/evaluate_origin')
