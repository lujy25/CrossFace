import pandas as pd
import numpy as np
import os
import shutil
from skimage import io
file_root_fold = 'xls_csv'

def analyze_ijbc_pose():
    df = pd.read_csv(os.path.join(file_root_fold, "test_IJB.csv"))
    pose_count_dict = {}
    for i in range(-90, 90):
        pose_count_dict[int(i)] = 0
    for index in df.index:
        yaw = df.ix[index, 'yaw']
        pose_count_dict[int(yaw)] += 1
    analyze_df = pd.DataFrame()
    for i in range(-90, 90):
        analyze_df.loc[i, 'num'] = pose_count_dict[int(i)]
    analyze_df.to_excel(os.path.join(file_root_fold, "analyze_test_pose.xls"))

def remove_no_id_image():
    df = pd.read_csv(os.path.join(file_root_fold, "ijbc_face_metadata.csv"))
    exist_files = df['file'].tolist()
    face_root_fold = 'datasets'
    img_frame = os.listdir(face_root_fold)
    exist_count = 0
    for parent_fold in img_frame:
        files = os.listdir(os.path.join(face_root_fold, parent_fold))
        for file in files:
            real_file = '%s/%s' % (parent_fold, file)
            if real_file not in exist_files:
                os.remove(os.path.join(face_root_fold, os.path.join(parent_fold, file)))
            else:
                exist_count += 1
    print(exist_count, len(df))

def select_content_image():
    face_root_dir = 'datasets'
    df = pd.read_csv(os.path.join(file_root_fold, "ijbc_face_metadata.csv"))
    not_exist_count = 0
    for index in df.index:
        file, yaw, subject_id = df.ix[index, ['file', 'yaw', 'subject_id']]
        if not os.path.exists(os.path.join(face_root_dir, file)):
            not_exist_count += 1
    print(not_exist_count)

def select_multi_files():
    df = pd.read_csv(os.path.join(file_root_fold, "ijbc_face_metadata.csv"))
    files = df['file'].tolist()
    multi_files = []
    for file in files:
        if files.count(file) > 1:
            multi_files.append(file)
    multi_files = list(np.unique(multi_files))
    df = pd.DataFrame()
    df['file'] = multi_files
    print(len(df))
    df.to_csv(("multi_file.csv"), index=False)

def crop_multi():
    save_df = pd.DataFrame()
    files = []
    yaws = []
    subject_ids = []
    import cv2
    image_fold = 'IJB-C/images'
    metadata_df = pd.read_csv( "ijbc_metadata.csv")
    multi_df = pd.read_csv("multi_file.csv")
    multi_files = multi_df['file'].tolist()
    output_dir = "multi-IJB-C"
    count = 0
    for index in metadata_df.index:
        subject_id, file, x, y, width, height, yaw = metadata_df.ix[
            index, ['SUBJECT_ID', 'FILENAME', 'FACE_X', 'FACE_Y', 'FACE_WIDTH', 'FACE_HEIGHT', 'YAW']]
        if file in multi_files:
            count += 1
            print(count, subject_id, file, yaw)
            image = cv2.imread(os.path.join(image_fold, file))
            face = image[int(y): int(y + height), int(x): int(x + width)]
            f_split = file.split('/')
            o_f_d = f_split[0]
            o_f_id = f_split[1]
            if not os.path.exists(os.path.join(output_dir, str(int(subject_id)))):
                os.makedirs(os.path.join(output_dir, str(int(subject_id))))
            cv2.imwrite(os.path.join(output_dir, os.path.join(str(int(subject_id)), '%s_%s' % (o_f_d, o_f_id))), face)
            files.append(os.path.join(str(int(subject_id)), '%s_%s' % (o_f_d, o_f_id)))
            yaws.append(yaw)
            subject_ids.append(subject_id)
    save_df['file'] = files
    save_df['yaw'] = yaws
    save_df['subject_id'] = subject_ids
    save_df.to_csv("multi_face_metadata.csv", index=False)

def classfy_face():
    out_dir = 'IJB'
    df = pd.read_csv(os.path.join(file_root_fold, "ijbc_face_metadata.csv"))
    for index in df.index:
        file, yaw, subject_id = df.ix[index, ['file', 'yaw', 'subject_id']]
        f_split = file.split('/')
        o_f_d = f_split[0]
        o_f_id = f_split[1]
        if not os.path.exists(os.path.join(out_dir, str(int(subject_id)))):
            os.makedirs(os.path.join(out_dir, str(int(subject_id))))
        print(index, file)
        shutil.copy(os.path.join('datasets', file),
                    os.path.join(out_dir, os.path.join(str(int(subject_id)), '%s_%s' % (o_f_d, o_f_id))))

def ready_df():
    df = pd.read_csv(os.path.join(file_root_fold, "ijbc_face_metadata.csv"))
    files = []
    yaws = []
    subject_ids = []
    for index in df.index:
        file, yaw, subject_id = df.ix[index, ['file', 'yaw', 'subject_id']]
        f_split = file.split('/')
        o_f_d = f_split[0]
        o_f_id = f_split[1]
        file = os.path.join(str(int(subject_id)), '%s_%s' % (o_f_d, o_f_id))
        files.append(file)
        yaws.append(yaw)
        subject_ids.append(subject_id)
    save_df = pd.DataFrame()
    save_df['subject_id'] = subject_ids
    save_df['file'] = files
    save_df['yaw'] = yaws
    save_df.to_csv("IJB_metadata.csv", index=False)

def test_exist():
    root_dir = 'datasets'
    df = pd.read_csv('xls_csv/IJB_metadata.csv')
    count = 0
    for index in df.index:
        file = df.ix[index, 'file']
        if not os.path.exists(os.path.join(root_dir, file)):
            print(file)
        else:
            print(file)
            io.imread(os.path.join(root_dir, file))
            count += 1
    print(count, len(df))

def analyze_img_num():
    df = pd.read_csv('exist_IJB_metadata.csv')
    min_num = 100
    max_num = 0
    id_count = 0
    num_count = 0
    less_count = 0
    large_count = 0
    for id, single_df in df.groupby(by='subject_id'):
        if len(single_df) < min_num:
            min_num = len(single_df)
        if len(single_df) > max_num:
            max_num = len(single_df)
        if len(single_df) < 10:
            less_count += 1
        if len(single_df) > 100:
            large_count += 1
        num_count += len(single_df)
        id_count += 1
    print(min_num, max_num, num_count / id_count)
    print(less_count, large_count, id_count - less_count, id_count - large_count)

def ready_exist_df():
    root_dir = 'datasets'
    df = pd.read_csv('xls_csv/IJB_metadata.csv')
    files = []
    yaws = []
    subject_ids = []
    for index in df.index:
        file, yaw, subject_id = df.ix[index, ['file', 'yaw', 'subject_id']]
        if os.path.exists(os.path.join(root_dir, file)):
            files.append(file)
            yaws.append(yaw)
            subject_ids.append(subject_id)
    save_df = pd.DataFrame()
    save_df['subject_id'] = subject_ids
    save_df['file'] = files
    save_df['yaw'] = yaws
    save_df.to_csv("exist_IJB_metadata.csv", index=False)

def adjust_class():
    df = pd.read_csv('xls_csv/test_IJB.csv')
    new_id = 0
    files = []
    yaws = []
    subject_ids = []
    new_subject_ids = []
    for id, single_df in df.groupby(by=['subject_id']):
        for index in single_df.index:
            file, yaw = single_df.ix[index, ['file', 'yaw']]
            files.append(file)
            yaws.append(yaw)
            subject_ids.append(id)
            new_subject_ids.append(new_id)
        new_id += 1
    save_df = pd.DataFrame()
    save_df['subject_id'] = subject_ids
    save_df['file'] = files
    save_df['yaw'] = yaws
    save_df['class'] = new_subject_ids
    save_df.to_csv("xls_csv/test_IJB.csv", index=False)


def split_train_test():
    df = pd.read_csv('xls_csv/IJB_metadata.csv')
    classes = list(np.unique(df['class'].tolist()))
    np.random.shuffle(classes)
    train_classes = classes[:int(len(classes) * 0.8)]
    train_df = df[df['class'].isin(train_classes)]
    test_df = df[~df['class'].isin(train_classes)]
    train_df.to_csv('xls_csv/train_IJB.csv')
    test_df.to_csv(("xls_csv/test_IJB.csv"))
if __name__ == "__main__":
    adjust_class()
