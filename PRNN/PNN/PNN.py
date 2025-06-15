import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader,Dataset,TensorDataset
from sklearn.model_selection import StratifiedKFold
import math
from PNN.EarlyStopping.early_stopping import EarlyStopping
from torch.autograd import Variable as V
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

def load_data(data_x, data_y, batch_size):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
    # k折交叉验证
    Fold = 0
    for train_index, test_index in skf.split(data_x, data_y):
        Fold = Fold + 1
        train_data_x = data_x.iloc[train_index, :]  # 得到训练数据
        train_data_y = data_y.iloc[train_index]
        test_data_x = data_x.iloc[test_index, :]  # 得到测试数据
        test_data_y = data_y.iloc[test_index]
        if(Fold==1):
            break
    train_data_x_tensor = torch.from_numpy(train_data_x.values).type(torch.float32)
    train_data_y_tensor = torch.tensor(np.array(train_data_y).astype(float))
    train_data_tensor = TensorDataset(train_data_x_tensor, train_data_y_tensor)
    train_loader = DataLoader(train_data_tensor, batch_size=batch_size, shuffle=True, drop_last=True)
    test_data_x_tensor = torch.from_numpy(test_data_x.values).type(torch.float32)
    test_data_y_tensor = torch.tensor(np.array(test_data_y).astype(float))
    test_data_tensor = TensorDataset(test_data_x_tensor, test_data_y_tensor)
    test_loader = DataLoader(test_data_tensor, batch_size=32, shuffle=True)
    return train_loader, test_loader

def train_epoch(model, optimizer, train_loader, class_num):
    BP_early_stopping = EarlyStopping()
    if(class_num==2):
        loss_fun = torch.nn.BCELoss()
    else:
        loss_fun = torch.nn.CrossEntropyLoss()
    for epoch in range(7):
        total_loss = 0
        model.train()
        for i, (x, y) in enumerate(train_loader):
            y = y.squeeze()
            y = torch.eye(class_num)[y.long(), :]
            pred = model(V(x))
            loss = loss_fun(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
        BP_early_stopping(total_loss, model)
        if BP_early_stopping.early_stop:
            #print(f"epoch = {epoch},Early stopping")
            break  # 跳出迭代，结束训练
    return BP_early_stopping.model

def test(model, test_loader):
    model.eval()
    true_label_list = []
    pre_label_list = []
    with torch.no_grad():
        for Data in test_loader:
            x, y = Data
            out = model(V(x))
            pre = out.argmax(1)
            true_label_list.append(y)
            pre_label_list.append(pre)
    y_true = np.concatenate(true_label_list)
    y_pre = np.concatenate(pre_label_list)
    acc = accuracy_score(y_true, y_pre)
    return acc

def get_optimal_parameter(search_space, data_x, data_y):  # ①
    best_acc = 0
    class_num = data_y.nunique()
    #print(f"class_num = {class_num}")
    best_parameter = {"lr": None, "batch_size": None}
    start_dim = data_x.shape[1]
    end_dim = class_num
    mean_split_space = math.pow(2, (math.log2(start_dim / end_dim)) / 5)
    class MyBP(nn.Module):
        def __init__(self, start_dim=start_dim, end_dim=end_dim, mean_split_space=mean_split_space):
            super(MyBP, self).__init__()
            self.BP = nn.Sequential(
                nn.Linear(start_dim, int(start_dim // mean_split_space)),
                nn.LeakyReLU(),
                nn.Linear(int(start_dim // mean_split_space), int(start_dim // math.pow(mean_split_space, 2))),
                nn.LeakyReLU(),
                nn.Linear(int(start_dim // math.pow(mean_split_space, 2)), int(start_dim // math.pow(mean_split_space, 3))),
                nn.LeakyReLU(),
                nn.Linear(int(start_dim // math.pow(mean_split_space, 3)), int(start_dim // math.pow(mean_split_space, 4))),
                nn.LeakyReLU(),
                nn.Linear(int(start_dim // math.pow(mean_split_space, 4)), end_dim),
                nn.Sigmoid(),
            )
        def forward(self, x):
            pre = self.BP(x)
            return pre
    for lr in search_space["lr"]:
        for batch_size in search_space["batch_size"]:
            train_loader, test_loader = load_data(data_x, data_y, batch_size)  # Load some data
            model = MyBP()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            model = train_epoch(model, optimizer, train_loader, class_num)  # Train the model
            acc = test(model, test_loader)  # Compute test accuracy
            #print(f"lr: {lr}, batch_size: {batch_size} --- acc = {acc}")
            if(acc > best_acc):
                best_acc = acc
                best_parameter["lr"] = lr
                best_parameter["batch_size"] = batch_size
    #print(f"Best_acc: {best_acc}")
    #print(f"Best_parameter: {best_parameter}")
    return best_parameter

def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs>costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

def PNN_FS(data_x,data_y,batch_size,lr, p):
    class_num = data_y.nunique()
    start_dim = data_x.shape[1]
    end_dim = class_num
    mean_split_space = math.pow(2, (math.log2(start_dim / end_dim)) / 5)
    class MyBP(nn.Module):
        def __init__(self, start_dim=start_dim, end_dim=end_dim, mean_split_space=mean_split_space):
            super(MyBP, self).__init__()
            self.BP = nn.Sequential(
                nn.Linear(start_dim, int(start_dim // mean_split_space)),
                nn.LeakyReLU(),
                nn.Linear(int(start_dim // mean_split_space), int(start_dim // math.pow(mean_split_space, 2))),
                nn.LeakyReLU(),
                nn.Linear(int(start_dim // math.pow(mean_split_space, 2)),
                          int(start_dim // math.pow(mean_split_space, 3))),
                nn.LeakyReLU(),
                nn.Linear(int(start_dim // math.pow(mean_split_space, 3)),
                          int(start_dim // math.pow(mean_split_space, 4))),
                nn.LeakyReLU(),
                nn.Linear(int(start_dim // math.pow(mean_split_space, 4)), end_dim),
                nn.Sigmoid(),
            )
        def forward(self, x):
            pre = self.BP(x)
            return pre

    data_x_tensor = torch.from_numpy(data_x.values).type(torch.float32)
    data_y_tensor = torch.tensor(np.array(data_y).astype(float))
    data = TensorDataset(data_x_tensor, data_y_tensor)
    dataset_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    #创建BP
    model = MyBP()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    BP_early_stopping = EarlyStopping()
    if (class_num == 2):
        loss_fun = torch.nn.BCELoss()
    else:
        loss_fun = torch.nn.CrossEntropyLoss()
    for epoch in range(1000):
        total_loss = 0
        model.train()
        for i, (x, y) in enumerate(dataset_loader):
            y = y.squeeze()
            y = torch.eye(class_num)[y.long(), :]
            pred = model(V(x))
            loss = loss_fun(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
        BP_early_stopping(total_loss, model)
        if BP_early_stopping.early_stop:
            #print(f"epoch = {epoch},Early stopping")
            break  # 跳出迭代，结束训练

    model = BP_early_stopping.model
    ##将训练好的BP神经网络权重依次相乘，根据得到的矩阵删除冗余特征
    w0 = model.state_dict()['BP.0.weight']
    w2 = model.state_dict()['BP.2.weight']
    w4 = model.state_dict()['BP.4.weight']
    w6 = model.state_dict()['BP.6.weight']
    w8 = model.state_dict()['BP.8.weight']
    #联乘权重矩阵
    W = torch.mm(torch.mm(torch.mm(torch.mm(w8, w6), w4), w2), w0)
    W = W.t()
    W = W.numpy()
    P_W = W
    P_W[P_W < 0] = -P_W[P_W < 0]
    #获得最高分数 obj向量
    min_max_scaler = preprocessing.MinMaxScaler()
    P_W_minMax = min_max_scaler.fit_transform(P_W)
    P_W_minMax_rank = P_W_minMax
    Obj_max = np.amax(P_W_minMax, axis=1)
    Obj_max_DF = pd.DataFrame(Obj_max)
    Obj_class = []
    for i, x in enumerate(Obj_max):
        for j in range(class_num):
            if (P_W_minMax[i, j] == x):
                Obj_class.extend([j])
    #获得最高1/排名 obj向量
    for i in range(class_num):
        index = np.argsort(-P_W_minMax[:, i])
        rank = np.argsort(index)
        P_W_minMax_rank[:, i] = rank + 1
    print(P_W_minMax_rank)
    P_W_minMax_rank = 1 / P_W_minMax_rank
    Obj_rank_max = np.amax(P_W_minMax_rank, axis=1)
    #合并两个目标向量
    Obj = np.concatenate(([Obj_max], [Obj_rank_max]), axis=0)
    Obj = Obj.T

    geneName = data_x.columns.tolist()
    Obj_max_DF.index = data_x.columns.tolist()
    fronts = []
    while (Obj.shape[0] != 0):
        res = is_pareto_efficient(Obj, True)
        res_index = [i for i, x in enumerate(res) if x]
        front_geneName = [x for i, x in enumerate(geneName) if i in res_index]
        geneName = np.delete(geneName, res_index, 0)
        Obj = np.delete(Obj, res_index, 0)
        fronts.append(front_geneName)
    fronts_len = []
    for value in fronts:
        fronts_len.extend([len(value)])
    #print(fronts)
    e = int(len(fronts_len)*p)
    final_geneName = []
    for i in range(e):
        final_geneName.extend(fronts[i])
    #print(final_geneName)
    return final_geneName, Obj_max_DF
