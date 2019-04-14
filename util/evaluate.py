import numpy as np
from sklearn.model_selection import KFold


def cal_topk_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def cal_10kfold_accuracy(distances, labels, nrof_folds=10):
    thresholds = np.arange(0, 30, 0.01)
    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    accuracy = np.zeros((nrof_folds))
    best_threshold = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, distances[train_set], labels[train_set])
        best_threshold_index = np.argmax(acc_train)
        best_threshold[fold_idx] = thresholds[best_threshold_index]
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index],
            distances[test_set], labels[test_set])
    return accuracy, best_threshold


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame),
                               np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

def cal_threshold_accuracy_with_yaw(distances, labels, yaw_cross, model_anc_paths, model_compair_paths, nrof_folds=10):
    thresholds = np.arange(0, 30, 0.01)
    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    accuracy = np.zeros((nrof_folds))
    best_threshold = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, distances[train_set], labels[train_set])
        best_threshold_index = np.argmax(acc_train)
        best_threshold[fold_idx] = thresholds[best_threshold_index]
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index],
            distances[test_set], labels[test_set])
    tp_cross, fp_cross, tn_cross, fn_cross, fn_anc_paths, fn_compair_paths, fp_anc_paths, fp_compair_paths =\
        calculate_false_yaw_cross(np.mean(best_threshold), distances, labels, yaw_cross, model_anc_paths, model_compair_paths)
    return accuracy, best_threshold, tp_cross, fp_cross, tn_cross, fn_cross, fn_anc_paths, fn_compair_paths, fp_anc_paths, fp_compair_paths

def calculate_false_yaw_cross(threshold, dist, actual_issame, yaw_cross, model_anc_paths, model_compair_paths):
    predict_issame = np.less(dist, threshold)
    tp_index = np.where(np.logical_and(predict_issame, actual_issame) == 1)
    fp_index = np.where(np.logical_and(predict_issame, np.logical_not(actual_issame)) == 1)
    tn_index = np.where(np.logical_and(np.logical_not(predict_issame),
                               np.logical_not(actual_issame)) == 1)
    fn_index = np.where(np.logical_and(np.logical_not(predict_issame), actual_issame) == 1)
    return yaw_cross[tp_index], yaw_cross[fp_index], yaw_cross[tn_index], yaw_cross[fn_index], \
           model_anc_paths[fn_index], model_compair_paths[fn_index], model_anc_paths[fp_index], model_compair_paths[fp_index]