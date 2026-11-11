"""Network definitions adapted verbatim from local Baseline_S2V.py; training is replaced."""
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
        mu = torch.zeros(x.size(0), x.size(1), self.embed_dim, device=x.device)
        def aggregate(values):
            if not adj_matrix.is_sparse:
                return torch.matmul(adj_matrix, values)
            if values.ndim == 2:
                return torch.sparse.mm(adj_matrix, values)
            batch, nodes, dim = values.shape
            flat = values.permute(1, 0, 2).reshape(nodes, batch * dim)
            return torch.sparse.mm(adj_matrix, flat).reshape(nodes, batch, dim).permute(1, 0, 2)
        # These terms depend on the state/parameters but not on message iteration.
        # Hoisting keeps the same computational function and gradient contributions.
        theta1_out = self.norm1(self.theta1(x))
        theta4_out = self.theta4(edge_weights)
        static_neighbors = self.norm3(aggregate(F.relu(theta4_out)))
        for _ in range(self.num_iterations):
            adj_mul_mu = self.norm2(aggregate(mu))
            mu = F.relu(theta1_out + self.theta2(adj_mul_mu) + self.theta3(static_neighbors))
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


