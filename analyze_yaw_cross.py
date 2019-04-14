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


def cal_yaw_center():
    df = pd.read_csv('./xls_csv/IJB_metadata.csv')
    yaws = np.array([[yaw] for yaw in df['yaw'].tolist()])
    estimator = KMeans(n_clusters=5)  # 构造聚类器
    estimator.fit(yaws)  # 聚类
    centroids = estimator.cluster_centers_  # 获取聚类中心
    print(centroids)
    frontal = [-20, 20]
    middle = [-40, -20, 20, 40]
    profile = [-90, 40, 40, 90]

if __name__ == "__main__":
    paths = ['./xls_csv/train_IJB.csv', './xls_csv/test_IJB.csv']
    for path in paths:
        df = pd.read_csv(path)
        frontal = df[(-20 < df['yaw']) & (df['yaw'] < 20)]
        middle = df[((-40 <= df['yaw']) & (df['yaw'] <= -20)) | ((20 <= df['yaw']) & (df['yaw'] <= 40))]
        profile = df[((-90 <= df['yaw']) & (df['yaw'] < -40)) | ((40 < df['yaw']) & (df['yaw'] <= 90))]
        print(len(frontal), len(middle), len(profile), len(df))