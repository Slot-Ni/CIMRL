import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class QNetwork(nn.Module):
    """DQN网络结构"""
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()  # 调用父类 nn.Module 的构造函数，完成父类的初始化
        # 定义一个顺序容器，按顺序包含一系列的神经网络层
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),  # 批量归一化
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
            # nn.LayerNorm(action_dim)
        )

    def forward(self, x):
        # 定义前向传播过程，将输入 x 依次通过 self.fc 中的所有层，最终返回网络的输出
        return self.fc(x)


class Compete_DQN:
    def __init__(self, epsilon, alpha, gamma, state_dim, action_dim, node_hyperedge_avgsize, avg_HigerOrder_dis_seednodes, candidate_nodes_pos=None, device='cpu'):
        self.q_net = QNetwork(state_dim, action_dim).to(device)  # 创建一个 QNetwork 实例，并将其移动到指定设备（如 GPU）上
        self.target_net = QNetwork(state_dim, action_dim).to(device)  # 创建一个目标网络，用于稳定训练过程，同样移动到指定设备上
        self.target_net.load_state_dict(self.q_net.state_dict())  # 将目标网络的参数初始化为与 Q 网络相同
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=alpha)  # 定义优化器，使用 Adam 优化算法来更新 Q 网络的参数，学习率为 alpha
        self.gamma = gamma  # 折扣因子，用于计算未来奖励的折扣值
        self.epsilon = epsilon  # 探索率，用于 ε-贪心策略，控制探索和利用的平衡
        self.state_dim = state_dim  # 当前状态的维度，由编码器编码当前状态决定
        self.action_dim = len(candidate_nodes_pos)  # 动作的维度，即可能的动作数量
        self.device = device  # 指定使用的设备，如 'cpu' 或 'cuda'
        self.loss_fn = nn.MSELoss()  # 定义损失函数，使用均方误差损失（MSE Loss）来计算预测值和目标值之间的误差
        self.node_hyperedge_avgsize = node_hyperedge_avgsize  # 获取nodes的平均超边大小
        self.avg_HigerOrder_dis_seednodes = avg_HigerOrder_dis_seednodes  # 获取nodes与seednodes的高阶距离列表
        self.candidate_nodes_pos = candidate_nodes_pos  # 候选节点的索引列表
        self.candidate_pos_to_index = {node: idx for idx, node in enumerate(candidate_nodes_pos)} # 候选节点索引到动作索引的映射字典

    def _encode_state(self, state):
        """基于超图的节点特征编码"""
        if state == -1:  # 初始状态
            return np.zeros(self.state_dim)
        else:
            features = []
            for node in state:
                hyperedge_avgsize = self.node_hyperedge_avgsize[node]  # 节点平均超边大小
                higerorder = self.avg_HigerOrder_dis_seednodes[node]  # 获取节点与种子节点的高阶距离
                features.extend([hyperedge_avgsize, higerorder])  # 节点平均超边大小
            features = features[:self.state_dim] + [0] * (self.state_dim - len(features))  # 截断或填充至固定长度
            return np.array(features)

    def take_action(self, state, CandidateNodesPool):
        state_feature = self._encode_state(state)  # 对当前状态进行编码，得到状态特征
        if np.random.rand() < self.epsilon:
            selected_node = np.random.choice(CandidateNodesPool)
            selected_index = self.candidate_pos_to_index[selected_node]  # 获取该节点在候选节点池中的索引
            return selected_node, selected_index  # 返回节点位置和索引
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_feature).unsqueeze(0).to(self.device)  # 将状态特征转换为FloatTensor类型的张量，并增加一个维度
                q_values = self.q_net(state_tensor)
                candidate_indices = [self.candidate_pos_to_index[node] for node in CandidateNodesPool]  # 获取候选节点在动作空间中的索引
                valid_q_values = q_values[0, candidate_indices]  # 从所有动作的 Q 值中选取候选节点对应的 Q 值
                action_idx = torch.argmax(valid_q_values).item()  # 选择 Q 值最大的动作的索引
                selected_node = CandidateNodesPool[action_idx]  # 根据索引选择候选节点
                selected_index = self.candidate_pos_to_index[selected_node]  # 获取该节点在候选节点池中的索引
                return selected_node, selected_index  # 返回节点位置和索引

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch  # 从采样的批次数据中解包出状态、动作、奖励、下一个状态和是否结束的标志
        states = torch.FloatTensor(np.array(states)).to(self.device)  # 将状态数据转换为NumPy数组，再转换为FloatTensor类型的张量，并移动到指定设备（CPU或GPU）上
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)  # 将动作数据转换为LongTensor类型的张量，并增加一个维度，使其形状为 [batch_size, 1]，然后移动到指定设备上
        rewards = torch.FloatTensor(rewards).to(self.device)  # 将奖励数据转换为FloatTensor类型的张量，并移动到指定设备上
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)   # 将下一个状态数据转换为NumPy数组，再转换为FloatTensor类型的张量，并移动到指定设备上
        dones = torch.FloatTensor(dones).to(self.device)  # 将是否结束的标志数据转换为FloatTensor类型的张量，并移动到指定设备上

        # 计算当前Q值
        current_q = self.q_net(states).gather(1, actions)  # self.q_net(states) 会输出当前状态下所有动作的Q值, .gather(1, actions) 会根据 actions 中的索引，从所有动作的Q值中选取对应动作的Q值

        # 计算目标Q值
        # 关闭梯度计算，因为目标网络的参数在这一步不需要更新
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]   # 计算下一个状态下所有动作的Q值，并取最大值，得到每个样本的最大Q值

            # 根据贝尔曼方程计算目标Q值
            # 当 dones 为 1 时，表示该回合结束，此时目标Q值仅为当前奖励
            # 当 dones 为 0 时，目标Q值为当前奖励加上折扣因子乘以未来最大Q值
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q.unsqueeze(1))  # 计算损失并反向传播
        self.optimizer.zero_grad()  # 清空优化器中累积的梯度，防止梯度累积
        loss.backward()  # 反向传播，计算损失函数关于网络参数的梯度
        self.optimizer.step()  # 使用优化器根据计算得到的梯度更新Q网络的参数
        return loss.item()  # 添加返回loss值

    def sync_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())   # 将Q网络的参数复制到目标网络中，使目标网络的参数与Q网络的参数相同

    def best_action(self, state, CandidateNodesPool):
        state_feature = self._encode_state(state)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_feature).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor)
            candidate_indices = [self.candidate_pos_to_index[node] for node in CandidateNodesPool]  # 获取候选节点在动作空间中的索引
            valid_q_values = q_values[0, candidate_indices]  # 从所有动作的 Q 值中选取候选节点对应的 Q 值
            action_idx = torch.argmax(valid_q_values).item()  # 选择 Q 值最大的动作的索引
            return CandidateNodesPool[action_idx]




class ReplayBuffer:
    def __init__(self, capacity):  # capacity是传入的参数，代表缓冲区所能容纳的最大经验数量。
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):  # add方法的作用是把一个经验元组添加到缓冲区里。
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # sample方法的用途是从缓冲区里随机采样出一个批次的经验数据。batch_size是传入的参数，代表要采样的经验数量。
        return random.sample(self.buffer, batch_size)

    def __len__(self):  # 返回self.buffer中当前存储的经验数量。
        return len(self.buffer)

