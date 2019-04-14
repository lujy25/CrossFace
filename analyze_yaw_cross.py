import pandas as pd
import shutil
import os
import numpy as np
from sklearn.cluster import KMeans
analyze_fold = './analyze_data'


def cal_cross():
    tp_df = pd.read_csv(os.path.join(analyze_fold, 'tp_cross.csv'))
    tn_df = pd.read_csv(os.path.join(analyze_fold, 'tn_cross.csv'))
    fp_df = pd.read_csv(os.path.join(analyze_fold, 'fp_cross.csv'))
    fn_df = pd.read_csv(os.path.join(analyze_fold, 'fn_cross.csv'))
    tp_count = {}
    tn_count = {}
    fp_count = {}
    fn_count = {}
    for i in range(0, 90):
        tp_count[int(i)] = 0
        tn_count[int(i)] = 0
        fp_count[int(i)] = 0
        fn_count[int(i)] = 0
    for index in tp_df.index:
        tp_count[int(tp_df.ix[index, 'tp_cross'])] += 1
    for index in tn_df.index:
        tn_count[int(tn_df.ix[index, 'tn_cross'])] += 1
    for index in fp_df.index:
        fp_count[int(fp_df.ix[index, 'fp_cross'])] += 1
    for index in fn_df.index:
        fn_count[int(fn_df.ix[index, 'fn_cross'])] += 1
    df = pd.DataFrame()
    for i in range(0, 90):
        df.loc[i, 'tp'] = tp_count[int(i)]
        df.loc[i, 'tn'] = tn_count[int(i)]
        df.loc[i, 'fp'] = fp_count[int(i)]
        df.loc[i, 'fn'] = fn_count[int(i)]
    df.to_excel(os.path.join(analyze_fold, "cross_count.xls"))


def copy_false_images(file_name):
    root_dir = 'datasets'
    save_dir = '../Images/' + file_name.split("_")[0]
    df = pd.read_csv(os.path.join(analyze_fold, file_name))
    for index in df.index:
        anc_path, compair_path = df.ix[index, ['anc_paths', 'compair_paths']]
        print(anc_path, compair_path)
        if not os.path.exists(os.path.join(save_dir, str(index))):
            os.makedirs(os.path.join(save_dir, str(index)))
        shutil.copy(os.path.join(root_dir, anc_path),
                    os.path.join(save_dir, os.path.join(str(index), 'anc_image.' + anc_path.split(".")[-1])))
        shutil.copy(os.path.join(root_dir, compair_path),
                    os.path.join(save_dir, os.path.join(str(index), 'compair_image.') + compair_path.split(".")[-1]))

def analyze_false():
    cal_cross()
    copy_false_images('fp_path.csv')
    copy_false_images('fn_path.csv')

def analyze_pose_split():
    paths = ['./xls_csv/train_IJB.csv', './xls_csv/test_IJB.csv']
    for path in paths:
        count_frontal, count_profile, count_middle, count_all = 0, 0, 0, 0
        frontal_profile, frontal_middle, profile_middle = 0, 0, 0
        all_df = pd.read_csv(path)
        for id, df in all_df.groupby(by=['class']):
            frontal = df[(-15 < df['yaw']) & (df['yaw'] < 15)]
            middle = df[((-45 <= df['yaw']) & (df['yaw'] <= -15)) | ((15 <= df['yaw']) & (df['yaw'] <= 45))]
            profile = df[((-90 <= df['yaw']) & (df['yaw'] < -45)) | ((45 < df['yaw']) & (df['yaw'] <= 90))]
            satisfy_frontal = False
            satisfy_profile = False
            satisfy_middle = False
            if len(frontal) >= 10:
                satisfy_frontal = True
                count_frontal += 1
            if len(profile) >= 4:
                satisfy_profile = True
                count_profile += 1
            if len(middle) >= 10:
                satisfy_middle = True
                count_middle += 1
            if satisfy_frontal and satisfy_profile and satisfy_middle:
                count_all += 1
            if satisfy_frontal and satisfy_profile:
                frontal_profile += 1
            if satisfy_frontal and satisfy_middle:
                frontal_middle += 1
            if satisfy_profile and satisfy_middle:
                profile_middle += 1
        print(count_frontal, count_profile, count_middle, count_all)
        print(frontal_profile, frontal_middle, profile_middle)

def cal_yaw_center():
    df = pd.read_csv('./xls_csv/IJB_metadata.csv')
    yaws = np.array([[yaw] for yaw in df['yaw'].tolist()])
    estimator = KMeans(n_clusters=5)  # 构造聚类器
    estimator.fit(yaws)  # 聚类
    centroids = estimator.cluster_centers_  # 获取聚类中心
    print(centroids)

def add_pose(file_path):
    df = pd.read_csv(file_path)
    for index in df.index:
        yaw = df.ix[index, 'yaw']
        if -15 < yaw < 15:
            df.ix[index, 'pose'] = 0
        if (-45 <= yaw <= -15) | (15 <= yaw <= 45):
            df.ix[index, 'pose'] = 1
        if (-90 <= yaw < -45) | (45 < yaw <= 90):
            df.ix[index, 'pose'] = 2
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    add_pose('xls_csv/train_IJB.csv')
    add_pose('xls_csv/test_IJB.csv')
    add_pose(('xls_csv/IJB_metadata.csv'))