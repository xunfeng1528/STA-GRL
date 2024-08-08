import numpy as np
import torch
import torch.nn as nn
import random
import os
from torch_geometric.data import DataLoader, Data
import pandas as pd
from matplotlib import pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings("ignore")
from TGT import STConv
import gen_Data

current_path = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在的项目根目录
root_path = os.path.dirname(current_path)
print("项目根目录路径：", root_path)


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # ensure deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)  # set PYTHONHASHSEED environment variable for reproducibility


edgeindex = [[0, 1], [0, 2], [1, 3], [2, 3], [1, 4], [1, 5], [3, 5], [4, 6], [5, 6], [5, 27], [5, 7],
[7, 27], [27, 26], [26, 29], [29, 28], [26, 28], [26, 24], [24, 25], [5, 8], [8, 10], [8, 9], [5, 9], [9, 20],
[20, 21], [9, 16], [15, 16], [3, 11], [11, 12], [11, 17], [11, 15], [17, 18], [18, 19], [9, 19], [9, 23], [11, 13],
[13, 14], [14, 22], [22, 23], [23, 24], [1, 0], [2, 0], [3, 1], [3, 2], [4, 1], [5, 1], [5, 3], [6, 4], [6, 5], [27, 5],
[7, 5], [27, 7], [26, 27], [29, 26],  [28, 29], [28, 26], [24, 26], [25, 24], [8, 5], [10, 8], [9, 8], [9, 5], [20, 9],
[21, 20], [16, 9], [16, 15], [11, 3], [12, 11], [17, 11], [15, 11], [18, 17], [19, 18], [19, 9], [23, 9], [13, 11], [14, 13], [22, 14], [23, 22], [24, 23]]
edge_index = torch.from_numpy(np.array(edgeindex).transpose()).long()

fix_seed(50)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
Gdata_list_train, Gdata_list_test, split,test_indices = gen_Data.gen_GNN_data()

data_set = Gdata_list_train
test_dataset = Gdata_list_test
dataset_size = len(data_set)
train_ratio = 0.95  # 训练集占的比例
val_test_ratio = 1  # 验证集和测试集在剩余部分中的比例，验证集占一半

# 计算各部分的大小
train_size = int(dataset_size * train_ratio)
val_test_size = dataset_size - train_size
val_size = int(val_test_size * val_test_ratio)
test_size = val_test_size - val_size

# 顺序划分数据集
train_dataset = data_set[:train_size]
val_dataset = data_set[train_size:train_size + val_size]
# test_dataset = data_set[train_size + val_size:]
print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))
print("Test size:", len(test_dataset))

train_batch = 32
val_batch = 32
train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_batch, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=val_batch, shuffle=False)

in_channels = 288
Gout_put_channels = 128
Ghidden_channels = 128

hidden_channels = Gout_put_channels
out_channels = 1728

in_channels = split
out_channels = 54 * split

aggregate = 'cat'
lr = 0.001
ours_weight_decay = 5e-3
weight_decay = 5e-3
epochs = 600
val_min_num = 0

in_size = 30

STConv_net = STConv(118, 1, 32, 1, 3, 10)
model = STConv_net.to(device)

criterion_2 = nn.L1Loss()
criterion = torch.nn.SmoothL1Loss()
optimizer = optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=True)

def train():
    model.train()
    total_loss = 0
    for step, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        y = data.y.view(-1, out_channels)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        L1loss = criterion_2(out, y)
        total_loss += L1loss.item()
    return total_loss / len(train_loader)

def validate(model_xc):
    model_xc.eval()
    total_loss = 0
    all_predictions = []  # 用于保存验证过程中的所有预测值
    all_targets = []      # 用于保存验证过程中的所有真实值

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model_xc(data)
            y = data.y.view(-1, out_channels)
            loss = criterion(out, y)
            L1loss = criterion_2(out, y)
            total_loss += L1loss.item()
            all_predictions.extend(out.cpu().numpy().tolist()) # 保存预测值
            all_targets.extend(y.cpu().numpy().tolist())       # 保存真实值

    return total_loss / len(val_loader), all_predictions, all_targets

val_predictions = []
val_targets = []
val_loss_list = []
train_loss_list = []

best_val_loss = float('inf')
best_epoch = 0
best_model_state_dict = None

for epoch in range(epochs):  # number of epochs
    train_loss = train()
    val_loss, epoch_val_preds, epoch_val_targets = validate(model)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    if epoch > val_min_num:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state_dict = model.state_dict()
            val_predictions = epoch_val_preds
            val_targets = epoch_val_targets
        scheduler.step(train_loss)
    print(f'Epoch: {epoch}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

if best_model_state_dict is not None:
    torch.save(best_model_state_dict, 'best_model_GTrans_yuanshi.pt')

print(f'Best Validation Loss: {best_val_loss:.4f}')

val_df = pd.DataFrame({
    'Val_Predictions': [item for sublist in val_predictions for item in sublist],
    'Val_Targets': [item for sublist in val_targets for item in sublist]
})
val_df['Val_Predictions'] = val_df['Val_Predictions'].astype(float)
val_df['Val_Targets'] = val_df['Val_Targets'].astype(float)
val_df.to_csv('prediction_GTrans_yuanshi.csv', index=False, float_format='%.6f')
print("验证预测值和真实值已保存到CSV文件。")

train_loss_list = train_loss_list[val_min_num:]
val_loss_list = val_loss_list[val_min_num:]

episodes_train_list = list(range(len(train_loss_list)))
episodes_val_list = list(range(len(val_loss_list)))
plt.plot(episodes_train_list, train_loss_list, label='train_loss_change', color='green')
plt.plot(episodes_val_list, val_loss_list, label='val_loss_change', color='red')
plt.xlabel('Episodes')
plt.ylabel('loss')
plt.legend()
plt.show()


# 将训练和验证损失保存到CSV文件
loss_df = pd.DataFrame({
    'Epoch': list(range(epochs)),
    'Train_Loss': train_loss_list,
    'Val_Loss': val_loss_list
})
loss_df.to_csv('output/train_loss.csv', index=False, float_format='%.6f')
print("训练和验证损失已保存到CSV文件。")

# Define the function to compute metrics

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Calculate relative error while avoiding division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        re = np.abs((y_true - y_pred) / y_true)
        re = np.nan_to_num(re, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN and Inf with 0
        re = np.mean(re)

    return mae, mse, rmse, r2, re

# Load best model for testing
best_model = STConv(118, 1, 32, 1, 3, 10).to(device)
best_model.load_state_dict(torch.load('best_model_GTrans_yuanshi.pt'))
best_model.eval()

# Evaluate on test data
test_predictions = []
test_targets = []
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = best_model(data)
        y = data.y.view(-1, out_channels)
        test_predictions.extend(out.cpu().numpy().tolist())
        test_targets.extend(y.cpu().numpy().tolist())

# Convert to numpy arrays
test_predictions = np.array(test_predictions)
test_targets = np.array(test_targets)

# Compute metrics
mae, mse, rmse, r2, re = compute_metrics(test_targets, test_predictions)

# Save metrics to result285.csv
results_df = pd.DataFrame({
    'MAE': [mae],
    'MSE': [mse],
    'RMSE': [rmse],
    'R2': [r2],
    'RE': [re]
})
results_df.to_csv('result285.csv', index=False, float_format='%.6f')
print("测试指标已保存到result.csv文件。")

# Save test predictions and true values to test_predictions285.csv
test_df = pd.DataFrame({
    'Test_Predictions': [item for sublist in test_predictions for item in sublist],
    'Test_Targets': [item for sublist in test_targets for item in sublist]
})
test_df['Test_Predictions'] = test_df['Test_Predictions'].astype(float)
test_df['Test_Targets'] = test_df['Test_Targets'].astype(float)
test_df.to_csv('test_predictions285.csv', index=False, float_format='%.6f')
print("测试预测值和真实值已保存到CSV文件。")
