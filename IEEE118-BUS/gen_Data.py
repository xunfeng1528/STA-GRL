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
    df = pd.read_csv("data/all_busloads_118.csv", index_col=None)
    data = df.values
    reshaped_data = data.ravel(order='F')
    reshaped_data = np.array(reshaped_data)
    newdata = reshaped_data.reshape((366, 99, 288)).transpose((0, 2, 1))
    zeros = np.zeros((newdata.shape[0], newdata.shape[1], 1))
    zero_point = [4, 8, 9, 24, 25, 29, 36, 37, 60, 62, 63, 64, 67, 68, 70, 80, 86, 88, 110]
    for i in zero_point:
        newdata = np.concatenate((newdata[:, :, :i], zeros, newdata[:, :, i:]), axis=2)
    newdata = newdata.transpose((0, 2, 1))
    x_all = torch.from_numpy(newdata).float()
    # 边矩阵
    edgeindex = [[0, 1], [0, 2], [3, 4], [2, 4], [4, 5], [5, 6], [7, 8], [8, 9],
                 [3, 10], [4, 10], [10, 11], [1, 11], [2, 11], [6, 11], [10, 12],
                 [11, 13], [12, 14], [13, 14], [11, 15], [14, 16], [15, 16], [16, 17],
                 [17, 18], [18, 19], [14, 18], [19, 20], [20, 21], [21, 22], [22, 23],
                 [22, 24], [24, 26], [26, 27], [27, 28], [7, 29], [25, 29], [16, 30],
                 [28, 30], [22, 31], [30, 31], [26, 31], [14, 32], [18, 33], [34, 35],
                 [34, 36], [32, 36], [33, 35], [33, 36], [36, 38], [36, 39], [29, 37],
                 [38, 39], [39, 40], [39, 41], [40, 41], [42, 43], [33, 42], [43, 44],
                 [44, 45], [45, 46], [45, 47], [46, 48], [41, 48], [41, 48], [44, 48],
                 [47, 48], [48, 49], [48, 50], [50, 51], [51, 52], [52, 53], [48, 53],
                 [48, 53], [53, 54], [53, 55], [54, 55], [55, 56], [49, 56], [55, 57],
                 [50, 57], [53, 58], [55, 58], [55, 58], [54, 58], [58, 59], [58, 60],
                 [59, 60], [59, 61], [60, 61], [62, 63], [37, 64], [63, 64], [48, 65],
                 [48, 65], [61, 65], [61, 66], [65, 66], [64, 67], [46, 68], [48, 68],
                 [68, 69], [23, 69], [69, 70], [23, 71], [70, 71], [70, 72], [69, 73],
                 [69, 74], [68, 74], [73, 74], [75, 76], [68, 76], [74, 76], [76, 77],
                 [77, 78], [76, 79], [76, 79], [78, 79], [67, 80], [76, 81], [81, 82],
                 [82, 83], [82, 84], [83, 84], [84, 85], [85, 86], [84, 87], [84, 88],
                 [87, 88], [88, 89], [88, 89], [89, 90], [88, 91], [88, 91], [90, 91],
                 [91, 92], [91, 93], [92, 93], [93, 94], [79, 95], [81, 95], [93, 95],
                 [79, 96], [79, 97], [79, 98], [91, 99], [93, 99], [94, 95], [95, 96],
                 [97, 99], [98, 99], [99, 100], [91, 101], [100, 101], [99, 102], [99, 103],
                 [102, 103], [102, 104], [99, 105], [103, 104], [104, 105], [104, 106],
                 [104, 107], [105, 106], [107, 108], [102, 109], [108, 109], [109, 110],
                 [109, 111], [16, 112], [31, 112], [31, 113], [26, 114], [113, 114], [67, 115],
                 [11, 116], [74, 117], [75, 117], [25, 24], [29, 16], [37, 36], [62, 58], [63, 60],
                 [64, 65], [67, 68], [80, 79], [7, 4], [1, 0], [2, 0], [4, 3], [4, 2], [5, 4], [6, 5],
                 [8, 7], [9, 8], [10, 3], [10, 4], [11, 10], [11, 1], [11, 2], [11, 6], [12, 10],
                 [13, 11], [14, 12], [14, 13], [15, 11], [16, 14], [16, 15], [17, 16], [18, 17],
                 [19, 18], [18, 14], [20, 19], [21, 20], [22, 21], [23, 22], [24, 22], [26, 24],
                 [27, 26], [28, 27], [29, 7], [29, 25], [30, 16], [30, 28], [31, 22], [31, 30],
                 [31, 26], [32, 14], [33, 18], [35, 34], [36, 34], [36, 32], [35, 33], [36, 33],
                 [38, 36], [39, 36], [37, 29], [39, 38], [40, 39], [41, 39], [41, 40], [43, 42],
                 [42, 33], [44, 43], [45, 44], [46, 45], [47, 45], [48, 46], [48, 41], [48, 41],
                 [48, 44], [48, 47], [49, 48], [50, 48], [51, 50], [52, 51], [53, 52], [53, 48],
                 [53, 48], [54, 53], [55, 53], [55, 54], [56, 55], [56, 49], [57, 55], [57, 50],
                 [58, 53], [58, 55], [58, 55], [58, 54], [59, 58], [60, 58], [60, 59], [61, 59],
                 [61, 60], [63, 62], [64, 37], [64, 63], [65, 48], [65, 48], [65, 61], [66, 61],
                 [66, 65], [67, 64], [68, 46], [68, 48], [69, 68], [69, 23], [70, 69], [71, 23],
                 [71, 70], [72, 70], [73, 69], [74, 69], [74, 68], [74, 73], [76, 75], [76, 68],
                 [76, 74], [77, 76], [78, 77], [79, 76], [79, 76], [79, 78], [80, 67], [81, 76],
                 [82, 81], [83, 82], [84, 82], [84, 83], [85, 84], [86, 85], [87, 84], [88, 84],
                 [88, 87], [89, 88], [89, 88], [90, 89], [91, 88], [91, 88], [91, 90], [92, 91],
                 [93, 91], [93, 92], [94, 93], [95, 79], [95, 81], [95, 93], [96, 79], [97, 79],
                 [98, 79], [99, 91], [99, 93], [95, 94], [96, 95], [99, 97], [99, 98], [100, 99],
                 [101, 91], [101, 100], [102, 99], [103, 99], [103, 102], [104, 102], [105, 99],
                 [104, 103], [105, 104], [106, 104], [107, 104], [106, 105], [108, 107], [109, 102],
                 [109, 108], [110, 109], [111, 109], [112, 16], [112, 31], [113, 31], [114, 26], [114, 113], [115, 67],
                 [116, 11],
                 [117, 74], [117, 75], [24, 25], [16, 29], [36, 37], [58, 62], [60, 63], [65, 64], [68, 67], [79, 80],
                 [4, 7]]

    edge_index = torch.from_numpy(np.array(edgeindex).transpose()).long()

    label_data_or = pd.read_csv("data/y_118.csv").values
    label_data = label_data_or.T

    # 标签值
    new_label_data = label_data  # .reshape((365, 288*54 ))
    y_all = torch.from_numpy(new_label_data).float()
    indices = np.arange(366)
    np.random.shuffle(indices)
    test_indices = indices[:30]
    train_indices = indices[30:]
    split = 6
    step = int(split / 2)
    # Prepare data for training set
    Gdata_list_train = []
    for i in train_indices:
        for j in range(288 - split + 1):
            if j % (split - step) == 0:
                x = x_all[i, :, j:j + split]
                y = y_all[i, j * 54:(j + split) * 54]
                data = Data(x=x, edge_index=edge_index, y=y)
                Gdata_list_train.append(data)
    # Prepare data for test set
    step = 0
    Gdata_list_test = []
    for i in test_indices:
        for j in range(288 - split + 1):
            if j % (split - step) == 0:
                x = x_all[i, :, j:j + split]
                y = y_all[i, j * 54:(j + split) * 54]
                data = Data(x=x, edge_index=edge_index, y=y)
                Gdata_list_test.append(data)

    return Gdata_list_train, Gdata_list_test, split,test_indices

Gdata_list_train, Gdata_list_test, split,test_indices = gen_GNN_data()
print('Gdata_list_train_size:', len(Gdata_list_train))
print('Gdata_list_test_size:', len(Gdata_list_test))
print(Gdata_list_train[0])