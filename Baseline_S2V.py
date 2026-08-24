import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import torch.nn.functional as F


class Structure2Vec(nn.Module):
    def __init__(self, embed_dim, nodes_number, num_iterations):
        super().__init__()
        self.embed_dim = embed_dim
        self.theta1 = nn.Linear(2, embed_dim)  # 输入维度为2（己方/对手标签）
        self.theta2 = nn.Linear(embed_dim, embed_dim)
        self.theta3 = nn.Linear(embed_dim, embed_dim)
        self.theta4 = nn.Linear(nodes_number, embed_dim)  # 边权重转换
        self.num_iterations = num_iterations

        self.norm1 = nn.LayerNorm(embed_dim)  # 归一化层
        self.norm2 = nn.LayerNorm(embed_dim)  # 归一化层
        self.norm3 = nn.LayerNorm(embed_dim)  # 归一化层



    def forward(self, x, adj_matrix, edge_weights):
        # x: 节点标签 [batch_size, num_nodes, 2]
        # adj_matrix: 邻接矩阵 [num_nodes, num_nodes]
        # edge_weights: 边权重 [num_nodes, num_nodes]
        mu = torch.zeros(x.size(0), x.size(1), self.embed_dim).to(x.device)
        for _ in range(self.num_iterations):

            theta1_out = self.theta1(x)  # [batch_size, num_nodes, embed_dim]
            theta4_out = self.theta4(edge_weights)  # [num_nodes, embed_dim]
            adj_mul_mu = torch.matmul(adj_matrix, mu)  # [batch_size, num_nodes, embed_dim]
            adj_mul_theta4 = torch.matmul(adj_matrix, F.relu(theta4_out))  # [num_nodes, embed_dim]

            # 添加归一化操作
            theta1_out = self.norm1(theta1_out)  # 归一化操作
            adj_mul_mu = self.norm2(adj_mul_mu)  # 归一化操作
            adj_mul_theta4 = self.norm3(adj_mul_theta4)  # 归一化操作

            # 组合并激活
            mu_new = F.relu(theta1_out + self.theta2(adj_mul_mu) + self.theta3(adj_mul_theta4))


            mu = mu_new
        return mu



class QNetwork(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.theta5 = nn.Linear(2 * embed_dim, 1)  # 拼接全局和节点嵌入
        self.theta6 = nn.Linear(embed_dim, embed_dim)
        self.theta7 = nn.Linear(embed_dim, embed_dim)

        self.norm6 = nn.LayerNorm(embed_dim)  # 归一化层
        self.norm7 = nn.LayerNorm(embed_dim)  # 归一化层

    def forward(self, node_embs):
        # global_emb: 全局池化嵌入 [batch_size, embed_dim]
        # node_embs: 节点嵌入 [batch_size, num_nodes, embed_dim]
        global_emb = torch.mean(node_embs, dim=1)
        global_proj = self.theta6(global_emb.unsqueeze(1))  # [batch_size, 1, embed_dim]
        node_proj = self.theta7(node_embs)                  # [batch_size, num_nodes, embed_dim]

        # 添加归一化操作
        global_proj = self.norm6(global_proj)  # 归一化操作
        node_proj = self.norm7(node_proj)  # 归一化操作

        combined = torch.cat([global_proj.expand(-1, node_embs.size(1), -1), node_proj], dim=-1)
        q_values = self.theta5(F.relu(combined)).squeeze(-1)  # [batch_size, num_nodes]

        return q_values


def state_to_tensor(state, nodes_number, device):
    tensor = torch.zeros((1, nodes_number, 2), device=device, dtype=torch.float32)
    for idx in state:
        tensor[0, idx, 0] = 1.0
    return tensor


class Compete_S2V_DQN:
    def __init__(self, epsilon, alpha, gamma, adj_matrix, edge_weights, nodes_number, candidate_pos, device, embedding_dim):
        # Structure2Vec 图嵌入网络
        self.s2v = Structure2Vec(embedding_dim, nodes_number, num_iterations=32)
        # Q 网络和目标网络
        self.q_net = QNetwork(embedding_dim).to(device)
        self.target_q_net = QNetwork(embedding_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=alpha)
        self.epsilon = epsilon
        self.gamma = gamma
        self.adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
        self.edge_weights = torch.tensor(edge_weights, dtype=torch.float32).to(device)
        self.nodes_number = nodes_number
        self.candidate_pos = candidate_pos
        self.device = device

    def take_action(self, state, candidate_nodes):
        # ε-贪心策略
        if np.random.rand() < self.epsilon:
            return np.random.choice(candidate_nodes)
        else:
            x = state_to_tensor(state, self.nodes_number, self.device)
            with torch.no_grad():
                node_embs = self.s2v(x, self.adj_matrix, self.edge_weights)  # 权重都为1
                q_values = self.q_net(node_embs)
                q_values = q_values.squeeze(0)
            return candidate_nodes[torch.argmax(q_values[candidate_nodes]).item()]

    def best_action(self, state, candidate_nodes):
        x = state_to_tensor(state, self.nodes_number, self.device)
        # 贪婪策略（推理阶段使用）
        with torch.no_grad():
            node_embs = self.s2v(x, self.adj_matrix, self.edge_weights)
            q_values = self.q_net(node_embs)

        q_values = q_values[0]
        # 从候选节点中选择具有最大 q 值的节点
        candidate_q_values = q_values[candidate_nodes]
        best_index = torch.argmax(candidate_q_values).item()
        return candidate_nodes[best_index]

    def sync_target_net(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())


    def update(self, batch):

        # 初始化空列表来存储每个样本的状态、动作、奖励、下一个状态和是否结束标志
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        # 遍历批次中的每个样本
        for state, action, reward, next_state, done in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        states_tensor = torch.cat([state_to_tensor(state, self.nodes_number, self.device) for state in states], dim=0)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)  # 将奖励数据转换为FloatTensor类型的张量，并移动到指定设备上
        next_states_tensor = torch.cat([state_to_tensor(next_state, self.nodes_number, self.device) for next_state in next_states], dim=0)
        dones = torch.FloatTensor(dones).to(self.device)  # 将是否结束的标志数据转换为FloatTensor类型的张量，并移动到指定设备上

        next_states_embs = self.s2v(next_states_tensor, self.adj_matrix, self.edge_weights)
        states_embs = self.s2v(states_tensor, self.adj_matrix, self.edge_weights)

        # 计算目标 Q 值
        with torch.no_grad():
            target_q = rewards + self.gamma * self.target_q_net(next_states_embs).max(1)[0]

        q_values = self.q_net(states_embs)
        current_q = q_values.gather(1, actions).squeeze(1)  # 确保 current_q 的尺寸为 [batch_size]

        # 确保 target_q 和 current_q 的尺寸一致
        assert current_q.shape == target_q.shape, f"current_q shape: {current_q.shape}, target_q shape: {target_q.shape}"

        # 计算损失并更新
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()


class S2VReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
