import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import math
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import PNN.PNN as PNN
from ReliefF import  ReliefF

'''
PRNN
'''
def PRNN(data_x, data_y, best_acc, best_std, best_gene, best_W, DX, DY, p):
    if (p < 0.9):
        p = 0.5 + (math.tan((9 * math.pi / 40) * (1 - (data_x.shape[1] / DX.shape[1])))) / 2
    #print(f"p = {p}")
    search_space = {"lr": [0.00001, 0.0001, 0.001, 0.01, 0.1], "batch_size": [16, 32, 64, 128]}
    optimal_parameter = PNN.get_optimal_parameter(search_space, data_x=data_x, data_y=data_y)
    gene, Obj_max_DF = PNN.PNN_FS(data_x, data_y, optimal_parameter["batch_size"], optimal_parameter["lr"], p)
    #print(len(gene))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=111)
    PRNN_acc = np.array([])
    # k折交叉验证
    Fold = 0
    for train_index, test_index in skf.split(DX, DY):
        Fold = Fold + 1
        #print(f"第 {Fold} 次交叉验证")
        train_data_x = DX.iloc[train_index, :]  # 得到训练数据
        train_data_y = DY.iloc[train_index]
        test_data_x = DX.iloc[test_index, :]  # 得到测试数据
        test_data_y = DY.iloc[test_index]
        # 获取R_NNP数据
        PRNN_train_data_x = train_data_x.loc[:, gene]
        PRNN_train_data_y = train_data_y
        PRNN_test_data_x = test_data_x.loc[:, gene]
        PRNN_test_data_y = test_data_y
        # 对NN_SF+GWO选择的数据使用SVM分类
        PRNN_model = SVC()
        PRNN_model.fit(PRNN_train_data_x, PRNN_train_data_y)
        predictions = PRNN_model.predict(PRNN_test_data_x)
        # 计算acc
        acc = accuracy_score(PRNN_test_data_y, predictions)
        PRNN_acc = np.append(PRNN_acc, values=acc)
        #print(f"PRNN ：acc = {acc}")
    #print("PRNN 的平均acc：")
    #print(f"acc = {np.mean(PRNN_acc)}, std = {np.std(PRNN_acc)}")
    if(np.mean(PRNN_acc) > best_acc):
        if(len(gene)<100):
            best_acc = np.mean(PRNN_acc)
            best_std = np.std(PRNN_acc)
            best_gene = gene
            best_W = Obj_max_DF
    if(len(gene)<=10):
        return best_gene, best_W
    data_x = data_x.loc[:, gene]
    return PRNN(data_x, data_y, best_acc, best_std, best_gene, best_W, DX, DY, p)

def main():
    ######################### 数据读取及处理 ##########################
    data = pd.read_csv(r'DATA\original\Multi_omics-TCGA-BIC-Zscore-merged-PAM50.csv', header=0, index_col=0)
    data = data.T  # 原始数据形状：（286，50829）
    # 读入标签
    target = pd.read_csv(r'DATA\original\target_PAM50.csv', header=None,index_col=0)
    target = target.iloc[:, 0]
    y_map = {'Normal': 0, 'LumA': 1, 'LumB': 2, 'Her2': 3, 'Basal': 4}
    target = target.map(y_map)
    # 合并数据和标签
    data['target'] = list(target)
    print(f"原始数据：{data.shape}")
    # 将数据中NaN值替换为均值
    for column in list(data.columns[data.isnull().sum() > 0]):
        mean_val = data[column].mean()
        data.fillna({column: mean_val}, inplace=True)
    DX = data.iloc[:, 0:(data.shape[1] - 1)]
    DY = data.iloc[:, data.shape[1] - 1]
    data_x = data.iloc[:, 0:(data.shape[1] - 1)]
    data_y = data.iloc[:, data.shape[1] - 1]
    # 使用ReliefF进行数据预处理
    f = ReliefF.ReliefF(100, data_x.shape[1]*0.1, 10)
    ReliefF_data_x = f.fit_transform(np.array(data_x), np.array(data_y))
    important_feature_index_list = list(f._important_weight.keys())
    ReliefF_important_feature_name = data_x.iloc[:,important_feature_index_list].columns
    print(f"ReliefF后的训练数据形状：{ReliefF_data_x.shape}")
    data_x = pd.DataFrame(ReliefF_data_x)
    data_x.columns = ReliefF_important_feature_name

    ######################### PRNN ##########################
    GeneName, best_W = PRNN(data_x, data_y, best_acc=0, best_std=0, best_gene=[], best_W=[], DX=DX, DY=DY, p=0.5)

    print(f"PRNN选择的基因数：{len(GeneName)}")
    print(f"PRNN选择的基因名称：{GeneName}")

if __name__ == '__main__':
    main()