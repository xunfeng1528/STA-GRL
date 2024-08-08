import random
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from env_refine import SimEnv
# from TCN_net import TemporalConvNet
from TGT import STConv
import collections
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import  Batch
import time

# 记录开始时间
start_time = time.time()


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class ReplayBuffer:
   ''' 经验回放池 '''

   def __init__(self, capacity):
      self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

   def add(self, state, action, reward, next_state, done_flag, demand):  # 将数据加入buffer
      self.buffer.append((state, action, reward, next_state, done_flag, demand))

   def sample(self, sample_size):  # 从buffer中采样数据,数量为batch_size
      transitions = random.sample(self.buffer, sample_size)
      state, action, reward, next_state, done, demand = zip(*transitions)
      return state, action, reward, next_state, done, demand

   def size(self):  # 目前buffer中数据的数量
      return len(self.buffer)


class QValueNet(torch.nn.Module):
    def __init__(self, action_dim, hidden_dim, out_dim, factor):
        super(QValueNet, self).__init__()
        self.fc1 = nn.Linear(action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*factor)
        self.fc3 = nn.Linear(hidden_dim*factor, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, out_dim)
        self.fc5 = nn.Linear(12, 1)
        self.factor = factor

    def forward(self, action):

        x = F.relu(self.fc1(action))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.reshape(-1, 12)
        x = self.fc5(x)
        return x


class BCERPG:

    def __init__(self, state_dim, hidden_dim, action_dim,  actor_lr, critic_lr, device, tau = 0.005, gamma = 0.98,sigma = 0.01):
        self.actor = STConv(30, 1, 128, 1, 3, 10).to(device)
        self.actor.load_state_dict(torch.load('best_model_GTrans_yuanshi.pt'))
        self.critic = QValueNet(action_dim, hidden_dim, out_dim=1, factor=2).to(device)
        self.actor_optimizer = torch.optim.Adam([{'params': self.actor.params}], lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.device = device
        self.actor_lr = actor_lr
        self.critic_scheduler =ReduceLROnPlateau(self.critic_optimizer, mode='min', factor=0.9, patience=5, verbose=True)
        self.actor_scheduler =ReduceLROnPlateau(self.actor_optimizer, mode='min', factor=0.9, patience=5, verbose=True)

    def act(self, state):
        with torch.no_grad():
            #print(state)
            state = state.to(self.device)
            action = F.relu(self.actor(state)).cpu()  # 将动作移到CPU上
            action = action.reshape(-1, 12, 6)
            action = np.array(action[0])
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def train(self, model, train_loader):
        model.train()
        total_loss = 0
        for step, data in enumerate(train_loader):
            #data = data.to(device)
            a, r = data[0], data[1]
            self.critic_optimizer.zero_grad()
            out = model(a)
            out = torch.sum(out, dim=1)
            y = r.reshape(-1)
            loss = F.l1_loss(out, y)
            loss.backward()
            self.critic_optimizer.step()
            L1loss = F.l1_loss(out, y)
            total_loss += L1loss.item()
        return total_loss / len(train_loader)

    def update(self, transition_dict, net, batch_size, epochs, state_dict, pd_factor, pmin_factor, pmax_factor, cost_factor):
        if net == 'critic':
            actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
            rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
            dataset = TensorDataset(actions, rewards)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            train_loss_list = []
            val_min_num = 50
            for epoch in range(epochs):  # number of epochs
                train_loss = self.train(self.critic, dataloader)
                train_loss_list.append(train_loss)
                if epoch > val_min_num:
                    self.critic_scheduler.step(train_loss)
                print(f'Epoch: {epoch}, critic Loss: {train_loss:.4f}')
            critic_dict = self.critic.state_dict()
            torch.save(critic_dict, 'critic_dict.pt')
            return train_loss_list

        elif net == 'actor':
            self.critic.load_state_dict(torch.load('critic_dict.pt'))
            state_list = [data.to(self.device) for data in transition_dict['states']]
            next_state_list = [data.to(self.device) for data in transition_dict['next_states']]
            states = Batch.from_data_list(state_list).to(self.device)
            next_states = Batch.from_data_list(next_state_list).to(self.device)
            actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
            rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1).to(self.device)
            dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1).to(self.device)
            a = torch.tensor(state_dict['aCost'], dtype=torch.float).to(self.device)
            b = torch.tensor(state_dict['bCost'], dtype=torch.float).to(self.device)
            c = torch.tensor(state_dict['cCost'], dtype=torch.float).to(self.device)
            p_min = torch.tensor(state_dict['p_min'], dtype=torch.float).to(self.device)
            p_max = torch.tensor(state_dict['p_max'], dtype=torch.float).to(self.device)
            demand = torch.tensor(np.array(transition_dict['demand']), dtype=torch.float).to(self.device)
            #先计算了梯度成本
            action_new = F.relu(self.actor(states).reshape(-1, 12, 6))#对不同的状态生成了不同的调度计划，在这里生成批调度规划
            re_1_intermediate = a * action_new ** 2 + b * action_new + c#计算每个机组在每个时段里的成本函数
            re_1_summed_6 = re_1_intermediate.sum(dim=2)#求出每个时段里的成本
            re_new = re_1_summed_6.sum(dim=1)#得出最终每个episode的成本
            #再计算约束项
            #第一个最重要的是供需平衡
            prod_demand_re = torch.abs(action_new.sum(dim=2) - demand)
            prod_demand_penalty = prod_demand_re.sum(dim=1)  # 每个episode的供需平衡惩罚

            # 计算最小约束违反惩罚
            min_violation_re = F.relu(p_min - action_new)
            min_violation_penalty = min_violation_re.sum(dim=2).sum(dim=1)  # 每个episode的最小约束违反惩罚

            # 计算最大约束违反惩罚
            max_violation_re = F.relu(action_new - p_max)
            max_violation_penalty = max_violation_re.sum(dim=2).sum(dim=1)  # 每个episode的最大约束违反惩罚

            # 计算总惩罚
            total_penalty = pd_factor*prod_demand_penalty + pmin_factor * min_violation_penalty + pmax_factor * max_violation_penalty

            #re_new = self.critic(action_new)
            re_all = total_penalty.reshape(-1, 1)#+cost_factor*(re_new.reshape(-1, 1))
            actor_loss = torch.mean(re_all)
            print('actor_loss', actor_loss.item())
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # 手动更新参数
            # with torch.no_grad():
            #     for param in self.actor.params:
            #         param -= self.actor_lr * param.grad
            # self.actor_lr = self.lr_scheduler.step(actor_loss.item())
            self.actor_optimizer.step()
            self.actor_scheduler.step(actor_loss)
            return actor_loss.item()

fix_seed(50)

### training

num_episodes = 250
critic_epochs = 300


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
default_supply_df = pd.read_csv('./data/processed_supply_file.csv')
# device = torch.device("cpu")

env = SimEnv(default_supply_df, device=device)
state, state_dict = env.reset()
state_dim = 30
action_dim = env.n_units
#print(state_dim)
#参数定义
gamma = 0.98
actor_lr = 1e-5
critic_lr = 0.001
hidden_dim = 128
agent = BCERPG(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, device)# def train():

pd_factor, pmin_factor, pmax_factor, cost_factor = 1, 1, 1, 1e-3
minimal_size = 3000
replay_buffer = ReplayBuffer(2*minimal_size)
batch_size = 128
sample_size = minimal_size
return_list = []

while replay_buffer.size() < minimal_size:
    print('Replay_buffer_size:', replay_buffer.size())
    state, state_dict = env.reset()
    is_done = False
    while not is_done:
        action = agent.act(state)
        # print(action)
        next_state, reward, is_done, state, prod_cost, prod_demand_cost, state_dict, demand = env.step(action)
        replay_buffer.add(state, action, reward, next_state, is_done, demand)
actor_loss_list = []
return_list = []

for i in tqdm(range(num_episodes), desc='Training Progress'):  # 使用tqdm显示进度条
    if i % 100 == 0:
      print("Epoch {}".format(i))
    state, state_dict = env.reset()
    day_cost = 0
    print(state_dict['day'])
    is_done = False
    while not is_done:
        action = agent.act(state)
        # print(action)
        next_state, reward, is_done, state, prod_cost, prod_demand_cost, state_dict, demand = env.step(action)
        replay_buffer.add(state, action, reward, next_state, is_done, demand)
        day_cost += reward
    return_list.append(-day_cost)
    state, action, reward, next_state, done, demand = replay_buffer.sample(sample_size)
    transition_dict = {
        'states': state,
        'actions': action,
        'next_states': next_state,
        'rewards': reward,
        'dones': done,
        'demand': demand
    }
    actor_loss = agent.update(transition_dict, 'actor', batch_size, critic_epochs, state_dict, pd_factor, pmin_factor, pmax_factor,cost_factor)
    actor_loss_list.append(actor_loss)
end_time = time.time()

# 计算时间差并转换为分钟
execution_time_seconds = end_time - start_time
execution_time_minutes = execution_time_seconds / 60

print("Execution time:", execution_time_minutes, "minutes")
best_model_state_dict = agent.actor.state_dict()
if best_model_state_dict is not None:
    torch.save(best_model_state_dict, 'best_model_RL.pt')

actor_loss_df = pd.DataFrame(actor_loss_list, columns=['actor_loss'])
actor_loss_df.to_csv('output/actorloss.csv', index=False)

# 将 return_list 保存到 output/return.csv
return_df = pd.DataFrame(return_list, columns=['return'])
return_df.to_csv('output/return.csv', index=False)

episodes_train_list = list(range(len(actor_loss_list)))
# plt.plot(episodes_train_list, actor_loss_list, label='actor_loss_change', color='green')
pd.Series(actor_loss_list).rolling(2).mean().plot(label='actor_loss_change',color='blue')
plt.xlabel('Episodes')
plt.ylabel('actor_loss')
plt.legend()
plt.show()

episodes_train_list = list(range(len(return_list)))
plt.plot(episodes_train_list, return_list, label='cost_change', color='green')
pd.Series(return_list).rolling(100).mean().plot(label='cost_change_rolling',color='blue')
plt.xlabel('Episodes')
plt.ylabel('Cost_change')
plt.legend()
plt.show()
