import numpy as np
from sklearn.model_selection import StratifiedKFold
import math
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import NN.NN as NN

def RNN(data_x, data_y, best_acc, best_std, best_gene, best_W, DX, DY, p):
    if (p < 0.9):
        p = 0.5 + (math.tan((9 * math.pi / 40) * (1 - (data_x.shape[1] / DX.shape[1])))) / 2
    print(f"p = {p}")
    search_space = {"lr": [0.00001, 0.0001, 0.001, 0.01, 0.1], "batch_size": [16, 32, 64, 128]}
    optimal_parameter = NN.get_optimal_parameter(search_space, data_x=data_x, data_y=data_y)
    gene, Obj_max_DF = NN.NN_FS(data_x, data_y, optimal_parameter["batch_size"], optimal_parameter["lr"], p)
    print(len(gene))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=111)
    R_NNP_acc = np.array([])
    # k折交叉验证
    Fold = 0
    for train_index, test_index in skf.split(DX, DY):
        Fold = Fold + 1
        print(f"第 {Fold} 次交叉验证")
        train_data_x = DX.iloc[train_index, :]  # 得到训练数据
        train_data_y = DY.iloc[train_index]
        test_data_x = DX.iloc[test_index, :]  # 得到测试数据
        test_data_y = DY.iloc[test_index]
        # 获取R_NNP数据
        R_NNP_train_data_x = train_data_x.loc[:, gene]
        R_NNP_train_data_y = train_data_y
        R_NNP_test_data_x = test_data_x.loc[:, gene]
        R_NNP_test_data_y = test_data_y
        # 对NN_SF+GWO选择的数据使用SVM分类
        R_NNP_model = SVC()
        R_NNP_model.fit(R_NNP_train_data_x, R_NNP_train_data_y)
        predictions = R_NNP_model.predict(R_NNP_test_data_x)
        # 计算acc
        acc = accuracy_score(R_NNP_test_data_y, predictions)
        R_NNP_acc = np.append(R_NNP_acc, values=acc)
        print(f"R_NN ：acc = {acc}")
    print("R_NN 的平均acc：")
    print(f"acc = {np.mean(R_NNP_acc)}, std = {np.std(R_NNP_acc)}")
    if(np.mean(R_NNP_acc) > best_acc):
        if(len(gene)<100):
            best_acc = np.mean(R_NNP_acc)
            best_std = np.std(R_NNP_acc)
            best_gene = gene
            best_W = Obj_max_DF
    if(len(gene)<=10):
        return best_gene, best_W
    data_x = data_x.loc[:, gene]
    return RNN(data_x, data_y, best_acc, best_std, best_gene, best_W, DX, DY, p)