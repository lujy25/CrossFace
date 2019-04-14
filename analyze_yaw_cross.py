import pandas as pd
import os
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


if __name__ == "__main__":
    cal_cross()