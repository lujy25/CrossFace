import argparse
import torch
from util.models import *
import numpy as np
from util.data_loader import get_train_face_extraction_dataloader, get_valid_face_extraction_dataloader
from util.evaluate import *
import os
from util.models import *
from tqdm import tqdm
from util.little_block import *


parser = argparse.ArgumentParser(
    description='Face Recognition using Arc Face')
parser.add_argument('--start-epoch', default=0, type=int, metavar='SE',
                    help='start epoch (default: 0)')
parser.add_argument('--end-epoch', default=125, type=int, metavar='NE',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--embedding-size', default=512, type=int, metavar='ES',
                    help='embedding size (default: 128)')
parser.add_argument('--valid-num-triplets', default=10000,
                    type=int, metavar='NTT',
                    help='number of triplets for valid (default: 10000)')
parser.add_argument('--train-batch-size', default=110, type=int, metavar='BS',
                    help='batch size (default: 128)')
parser.add_argument('--margin', default=0.5, type=float, metavar='MG',
                    help='margin (default: 0.5)')
parser.add_argument('--valid-batch-size', default=16, type=int, metavar='BS',
                    help='batch size (default: 128)')
parser.add_argument('--num-workers', default=4, type=int, metavar='NW',
                    help='number of workers (default: 8)')
parser.add_argument('--train-root-dir', default='./datasets', type=str,
                    help='path to train root dir')
parser.add_argument('--valid-root-dir', default='./datasets', type=str,
                    help='path to valid root dir')
parser.add_argument('--train-csv-name', default='./xls_csv/train_IJB.csv', type=str,
                    help='list of training images')
parser.add_argument('--valid-csv-name', default='./xls_csv/test_IJB.csv', type=str,
                    help='list of validtion images')

args = parser.parse_args()
LOSS = FocalLoss()
arc_losses = AverageMeter()
triplet_losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()
l2_dist = PairwiseDistance(2)

device_id = 0
device = torch.device('cuda:%d' % device_id if torch.cuda.is_available() else 'cpu')


def separate_irse_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'models' in str(layer.__class__) and not 'ArcFace' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn

def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 10.

def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params['lr'] = batch * init_lr / num_batch_warm_up

batch = 1
save_fold = 'not-fineturing-Origin'
if not os.path.exists(os.path.join('log', save_fold)):
    os.makedirs(os.path.join('log', save_fold))

def main():
    train_dataset, train_dataloader = get_train_face_extraction_dataloader(root_dir=args.train_root_dir,
                                                            csv_name=args.train_csv_name,
                                                            batch_size=args.train_batch_size,
                                                            num_workers=args.num_workers)
    valid_dataset, valid_dataloader = get_valid_face_extraction_dataloader(root_dir=args.valid_root_dir,
                                                                           csv_name=args.valid_csv_name,
                                                                           batch_size=args.valid_batch_size,
                                                                           num_workers=args.num_workers)
    faceExtraction = Backbone().to(device)
    # faceExtraction.load_state_dict(torch.load('./model/arcface_weight/backbone_ir50_ms1m_epoch120.pth'))
    arcOutput = ArcFace(in_features=args.embedding_size, out_features=train_dataset.get_class_num(), device_id=[device_id]).to(device)
    backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(faceExtraction)
    _, head_paras_wo_bn = separate_irse_bn_paras(arcOutput)
    optimizer = torch.optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': 5e-4},
                                 {'params': backbone_paras_only_bn}], lr=0.1, momentum=0.9)

    NUM_EPOCH_WARM_UP = args.end_epoch // 25
    NUM_BATCH_WARM_UP = len(train_dataloader) * NUM_EPOCH_WARM_UP
    for epoch in range(args.start_epoch, args.end_epoch):
        if epoch in [35, 65, 95]:
            schedule_lr(optimizer)
        print(80 * '=')
        print('Epoch [{}/{}]'.format(epoch, args.end_epoch))
        train_model(epoch, NUM_EPOCH_WARM_UP, NUM_BATCH_WARM_UP, faceExtraction, arcOutput, train_dataset, train_dataloader, optimizer)
        valid_model(epoch, faceExtraction, valid_dataset, valid_dataloader)
    print(80 * '=')


def valid_model(epoch, faceExtraction, valid_dataset, valid_dataloader):
    faceExtraction.eval()
    valid_dataset.sample_triplets()
    triplet_losses.reset()
    distances, labels = [], []
    for batch_idx, batch_sample in tqdm(enumerate(valid_dataloader)):
        anc_img = batch_sample['anc_img'].to(device)
        pos_img = batch_sample['pos_img'].to(device)
        neg_img = batch_sample['neg_img'].to(device)
        anc_embed = l2_norm(faceExtraction(anc_img))
        pos_embed = l2_norm(faceExtraction(pos_img))
        neg_embed = l2_norm(faceExtraction(neg_img))

        triplet_loss = TripletLoss(args.margin).forward(
            anc_embed, pos_embed, neg_embed).to(device)
        triplet_losses.update(triplet_loss.data.item(), anc_img.size(0))

        dists = l2_dist.forward(anc_embed, pos_embed)
        distances.append(dists.data.cpu().numpy())
        labels.append(np.ones(dists.size(0)))

        dists = l2_dist.forward(anc_embed, neg_embed)
        distances.append(dists.data.cpu().numpy())
        labels.append(np.zeros(dists.size(0)))
    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])
    accuracy, best_threshold = cal_10kfold_accuracy(distances, labels)
    print('Valid Loss         = {loss.val:.4f} ({loss.avg:.4f})\tAccuracy         = {accuracy:.4f}'.format(loss=triplet_losses, accuracy=np.mean(accuracy)))
    with open('./log/{}/{}_log.txt'.format(save_fold, str(save_fold)), 'a') as f:
        f.write('Valid Loss         = {loss.val:.4f} ({loss.avg:.4f})\tAccuracy         = {accuracy:.4}\n'.format(loss=triplet_losses, accuracy=np.mean(accuracy)))
        f.close()


def train_model(epoch, NUM_EPOCH_WARM_UP, NUM_BATCH_WARM_UP, faceExtraction, arcOutput, train_dataset, train_dataloader, optimizer):
    faceExtraction.train()
    arcOutput.train()
    arc_losses.reset()
    top1.reset()
    top5.reset()
    for batch_idx, batch_sample in tqdm(enumerate(train_dataloader)):
        global batch
        if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (
                batch + 1 <= NUM_BATCH_WARM_UP):  # adjust LR for each training batch during warm up
            warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, 0.1, optimizer)
        face_img = batch_sample['face_img'].to(device)
        face_class = batch_sample['face_class'].to(device).long()
        features = faceExtraction(face_img)
        outputs = arcOutput(features, face_class)
        loss = LOSS(outputs, face_class)
        # measure accuracy and record loss
        prec1, prec5 = cal_topk_accuracy(outputs.data, face_class, topk=(1, 5))
        arc_losses.update(loss.data.item(), face_img.size(0))
        top1.update(prec1.data.item(), face_img.size(0))
        top5.update(prec5.data.item(), face_img.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch += 1
    print('Epoch %d / %s\n' % (epoch, args.end_epoch))
    print('Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
          'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(loss=arc_losses, top1=top1, top5=top5))
    with open('./log/{}/{}_log.txt'.format(save_fold, str(save_fold)), 'a') as f:
        f.write('Epoch %d / %s\n' % (epoch, args.end_epoch))
        f.write('Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(loss=arc_losses, top1=top1, top5=top5))
        f.close()
    if epoch % 20 == 0:
        torch.save({'epoch': epoch,
                    'state_dict': faceExtraction.state_dict()},
                   './log/{}/{}_BACKBONE_checkpoint_epoch{}.pth'.format(str(save_fold), str(save_fold),
                                                                                        epoch))
        torch.save({'epoch': epoch,
                    'state_dict': arcOutput.state_dict()},
                   './log/{}/{}_HEAD_checkpoint_epoch{}.pth'.format(str(save_fold), str(save_fold),
                                                                                    epoch))


if __name__ == "__main__":
    main()