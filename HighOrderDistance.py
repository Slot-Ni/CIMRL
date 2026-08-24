# -*- coding: UTF-8 -*-
import sys

from tqdm import trange, tqdm
import itertools
import numpy as np
import dhg
import os


from collections import defaultdict, deque
import heapq
import time

# def common_nodes_matrix(im):
#     cnm = im.transpose().dot(im)
#     return cnm
#
# def find_shortest_walk(im, v, s):  # 超边v到其他超边之间的最短距离,s为阶数
#     cnm = common_nodes_matrix(im)
#     np.fill_diagonal(cnm, 0)
#     s_cnm = (cnm >= s).astype(int)
#     E = s_cnm.shape[0]
#
#     P = [v]
#     T = [i for i in range(E) if i not in P]
#
#     infinity = float('inf')
#     dis = [infinity, ] * E
#     dis[v] = 0
#
#     for i in np.where(s_cnm[v, :] != 0)[0]:
#         dis[i] = 1
#
#     while len(P) < E:
#         T_distance = [dis[i] for i in T]
#         shortest_dis = min(T_distance)
#         dic = dict(zip(T, T_distance))
#         p = [k for k, v in dic.items() if v == shortest_dis]
#         s = np.random.choice(p)
#
#         for i in np.where(s_cnm[s, :] != 0)[0]:
#             dis[i] = min(dis[i], dis[s] + 1)
#         P.append(s)
#         T.remove(s)
#     return dis    # 向量
#
#
#
#
# def main():
#     # data = 'high-school_2012'
#     # data = 'house-committees'
#     # data = "synthetic_WS_v1000_e1000_k=8"
#     data = "synthetic_WS_v1000_e1000_k=10_p=0.5"
#     # data = "synthetic_WS_v1000_e1000_k=20_p=0.5"
#     # data = "synthetic_WS_v1000_e1000_k=50_p=0.5"
#     order = 1
#
#     im = np.load(f'../HyperGraphMatrix/{data}.npy')
#     file_path = f'../HyperData_Structure/{data}_structure'
#     algebra_hy = dhg.load_structure(f'../HyperData_Structure/{data}_structure')
#     al_hyedge_num = algebra_hy.num_e
#     edge2edge = []
#
#     for i in trange(al_hyedge_num):
#         edge2edge.append(find_shortest_walk(im, i, order))
#     np.save(f'../HyperData_Structure/{data}_e2e_{order}.npy', edge2edge)
#
#
#
#     # algebra_hy = dhg.load_structure(r'./highschool2012_structure')
#     im = algebra_hy.H.to_dense()
#     N = algebra_hy.num_v
#
#     D = np.load(f'../HyperData_Structure/{data}_e2e_{order}.npy')  # 超边距离矩阵
#     np.fill_diagonal(D, 1)
#     d = np.zeros(shape=(N, N))  # 节点距离矩阵
#
#     # 节点的笛卡尔积
#     node_product = itertools.product(range(N), range(N))
#     for i, j in tqdm(list(node_product)):
#         dis = []
#         edge1 = np.where(im[i, :] == 1)[0]  # 节点i所在的超边a
#         edge2 = np.where(im[j, :] == 1)[0]  # 节点j所在的超边a
#
#         # 检查 edge1 和 edge2 是否为空
#         if len(edge1) == 0 or len(edge2) == 0:
#             d[i, j] = np.inf
#         else:
#             # 取所在超边的所有距离的最小值作为节点间的距离d
#             edge_product = itertools.product(edge1, edge2)
#             for e1, e2 in edge_product:
#                 dis.append(D[e1, e2])
#             d[i, j] = min(dis)
#
#     # n3p.fill_diagonal(d, 0)    # 增加此行代码，将节点距离矩阵对角线元素重置为0
#     np.save(f'../Hyper_Distance/{data}_distance1_{order}.npy', d)
#
# if __name__ == '__main__':
#     main()





def optimize_higher_order_distance(G_matrix, o):
    """
    优化版计算超图中节点间的高阶距离，减少运行时间

    参数:
    G_matrix: 超图的邻接矩阵，行为节点，列为超边，元素为0或1
    o: 高阶距离的阶数参数

    返回:
    dist_matrix: 节点间的高阶距离矩阵
    """
    start_time = time.time()
    num_nodes, num_edges = G_matrix.shape

    # 优化1：预处理节点-超边和超边-节点映射，使用列表而非defaultdict
    node_edges = [[] for _ in range(num_nodes)]
    edge_nodes = [[] for _ in range(num_edges)]

    for node in range(num_nodes):
        for edge in range(num_edges):
            if G_matrix[node, edge] == 1:
                node_edges[node].append(edge)
                edge_nodes[edge].append(node)

    # 优化2：预处理超边邻接表，避免重复计算
    edge_adj = [[] for _ in range(num_edges)]
    for e1 in range(num_edges):
        nodes_e1 = set(edge_nodes[e1])
        for e2 in range(e1 + 1, num_edges):
            nodes_e2 = set(edge_nodes[e2])
            if len(nodes_e1 & nodes_e2) >= o:
                edge_adj[e1].append(e2)
                edge_adj[e2].append(e1)

    # 优化3：使用numpy数组初始化距离矩阵
    dist_matrix = np.full((num_nodes, num_nodes), np.inf)

    # 优化4：向量化处理节点自身距离
    np.fill_diagonal(dist_matrix, 1)

    # 优化5：分块处理节点对，减少内存压力
    block_size = max(1, num_nodes // 100)  # 假设最多分100块
    for i in trange(0, num_nodes, block_size):
        end_i = min(i + block_size, num_nodes)
        for j in range(i, num_nodes):
            if i == j:
                continue

            # 优化6：提前检查是否在同一超边
            common_edges = set(node_edges[i]) & set(node_edges[j])
            if common_edges:
                dist_matrix[i, j] = 1
                dist_matrix[j, i] = 1
                continue

            # 优化7：缓存超边距离计算结果
            min_dist = np.inf
            edges_i = node_edges[i]
            edges_j = node_edges[j]

            # 优化8：剪枝超边组合，只计算可能的最短路径
            if edges_i and edges_j:
                # 优先检查较小的超边集合
                if len(edges_i) > len(edges_j):
                    edges_i, edges_j = edges_j, edges_i

                # 优化9：使用记忆化搜索缓存超边距离
                edge_dist_cache = {}
                for e1 in edges_i:
                    for e2 in edges_j:
                        if (e1, e2) not in edge_dist_cache:
                            edge_dist = _optimized_edge_distance(edge_adj, e1, e2)
                            edge_dist_cache[(e1, e2)] = edge_dist
                            edge_dist_cache[(e2, e1)] = edge_dist
                        current_dist = edge_dist_cache[(e1, e2)]
                        if current_dist < min_dist:
                            min_dist = current_dist

                if min_dist != np.inf:
                    dist_matrix[i, j] = min_dist + 1
                    dist_matrix[j, i] = min_dist + 1

    print(f"优化算法运行时间: {time.time() - start_time:.4f}秒")
    return dist_matrix


def _optimized_edge_distance(edge_adj, start_edge, end_edge):
    """
    优化版计算超边间距离，使用BFS替代Dijkstra（适用于无权图）
    """
    if start_edge == end_edge:
        return 0

    visited = set()
    queue = deque([(start_edge, 0)])
    visited.add(start_edge)

    while queue:
        current_edge, dist = queue.popleft()

        for neighbor in edge_adj[current_edge]:
            if neighbor == end_edge:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return np.inf

# data = "synthetic_WS_v1000_e1000_k=8"
# data = "synthetic_WS_v1000_e1000_k=10_p=0.5"
# data = "synthetic_WS_v1000_e1000_k=20_p=0.5"
# data = "synthetic_WS_v1000_e1000_k=50_p=0.5"

exp = 3.0
nsize = 3000
esize = 3000

hpe_maxdeg = 10


G_matrix = np.load(f'../HyperGraphMatrix/synthetic_HCL_v{nsize}_e{esize}_exp{exp}_maxdeg{hpe_maxdeg}.npy', allow_pickle=True)
print(G_matrix.shape)
# 计算2阶距离
for i in trange(1, 11):
    distance_matrix = optimize_higher_order_distance(G_matrix, i)
    np.save(f'../Hyper_Distance/synthetic_HCL_v{nsize}_e{esize}_exp{exp}_maxdeg{hpe_maxdeg}_distance_{i}.npy', distance_matrix)
    print(distance_matrix.shape)

