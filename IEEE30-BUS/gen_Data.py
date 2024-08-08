import numpy as np
import torch
import random
import os
from torch_geometric.data import DataLoader, Data
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # ensure deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)  # set PYTHONHASHSEED environment variable for reproducibility


def gen_GNN_data():
    fix_seed(50)
    node_data = pd.read_csv("data/IEEE30_combined_data_LoadDef.csv").values
    label_data = pd.read_csv("data/all_a_array_2.csv", header=None).values
    #节点特征处理
    new_node_data = node_data.reshape((365, 288, 20)).transpose(0, 2, 1)#365*20*288
    insert_positions = [1, 2, 3, 6, 7, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 28, 29]
    node_feat_data = np.zeros((365, 30, 288))
    node_feat_data[:, insert_positions, :] = new_node_data
    x_all = torch.from_numpy(node_feat_data).float()
    #边矩阵
    edgeindex = [[0, 1], [0, 2], [1, 3], [2, 3], [1, 4], [1, 5], [3, 5], [4, 6], [5, 6], [5, 27], [5, 7],
    [7, 27], [27, 26], [26, 29], [29, 28], [26, 28], [26, 24], [24, 25], [5, 8], [8, 10], [8, 9], [5, 9], [9, 20],
    [20, 21], [9, 16], [15, 16], [3, 11], [11, 12], [11, 17], [11, 15], [17, 18], [18, 19], [9, 19], [9, 23], [11, 13],
    [13, 14], [14, 22], [22, 23], [23, 24], [1, 0], [2, 0], [3, 1], [3, 2], [4, 1], [5, 1], [5, 3], [6, 4], [6, 5], [27, 5],
    [7, 5], [27, 7], [26, 27], [29, 26],  [28, 29], [28, 26], [24, 26], [25, 24], [8, 5], [10, 8], [9, 8], [9, 5], [20, 9],
    [21, 20], [16, 9], [16, 15], [11, 3], [12, 11], [17, 11], [15, 11], [18, 17], [19, 18], [19, 9], [23, 9], [13, 11], [14, 13], [22, 14], [23, 22], [24, 23]]
    edge_index = torch.from_numpy(np.array(edgeindex).transpose()).long()
    #标签值
    new_label_data = label_data#.reshape((365, 288*6 ))
    y_all = torch.from_numpy(new_label_data).float()

    indices = np.arange(365)
    np.random.shuffle(indices)
    test_indices = indices[:30]
    train_indices = indices[30:]
    split = 12
    step = int(split / 2)

    # Prepare data for training set
    Gdata_list_train = []
    for i in train_indices:
        for j in range(288 - split + 1):
            if j % (split - step) == 0:
                x = x_all[i, :, j:j + split]
                y = y_all[i, j * 6:(j + split) * 6]
                data = Data(x=x, edge_index=edge_index, y=y)
                Gdata_list_train.append(data)

    # Prepare data for test set
    step = 0
    Gdata_list_test = []
    for i in test_indices:
        for j in range(288 - split + 1):
            if j % (split - step) == 0:
                x = x_all[i, :, j:j + split]
                y = y_all[i, j * 6:(j + split) * 6]
                data = Data(x=x, edge_index=edge_index, y=y)
                Gdata_list_test.append(data)

    return Gdata_list_train, Gdata_list_test, split,test_indices

Gdata_list_train, Gdata_list_test, split ,test_indices= gen_GNN_data()
print('Gdata_list_train_size:', len(Gdata_list_train))
print('Gdata_list_test_size:', len(Gdata_list_test))
print(Gdata_list_train[0])