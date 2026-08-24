import numpy as np
import os

# data = "Algebra"
# data = "Restaurants-Rev"
# data = "Geometry"
data = "Bars-Rev"
# data = "Music-Rev"
# data = "iAF1260b"
# data = "iJO1366"
# data = "NDC-classes"  # 1
# data = "soc-hamsterster"
# data = 'house-committees'

# data = "synthetic_HCL_v500_e500_exp2.0_maxdeg50"
# data = "synthetic_HCL_v500_e1000_exp2.0_maxdeg50"
# data = "synthetic_HCL_v500_e1500_exp2.0_maxdeg50"
# data = "synthetic_HCL_v1000_e500_exp2.0_maxdeg100"
# data = "synthetic_HCL_v1000_e1000_exp2.0_maxdeg100"
# data = "synthetic_HCL_v1000_e1500_exp2.0_maxdeg100"


G_matrix = np.load('../HyperGraphMatrix/' + data + '.npy', allow_pickle=True)
print(data)
print('number of nodes:   ' + str(G_matrix.shape[0]))
print('number of hyperedges:  ' + str(G_matrix.shape[1]))
print('average hyperedge size:   ' + str(G_matrix.sum(axis=0).mean()))
# print('average degree of nodes:   ' + str(G_matrix.sum(axis=).mean()))
print('average hyperdegree of nodes:   ' + str(G_matrix.sum(axis=1).mean()))

node_degrees = []
# 遍历每个节点
for node in range(G_matrix.shape[0]):
    # 找到该节点所在的所有超边（列索引）
    hyperedges = np.where(G_matrix[node] == 1)[0]
    # 收集这些超边中包含的所有节点（行索引）
    neighbor_nodes = set()
    for he in hyperedges:
        # 找到当前超边中的所有节点
        nodes_in_he = np.where(G_matrix[:, he] == 1)[0]
        # 排除自身，添加其他节点到邻节点集合（去重）
        neighbor_nodes.update(nodes_in_he[nodes_in_he != node])
    # 该节点的度 = 邻节点总数
    node_degrees.append(len(neighbor_nodes))

# 计算平均度
average_degree = np.mean(node_degrees)
print('average degree of nodes:   ' + str(average_degree))
#
# num = 0
# for i in G_matrix:
#     a = 0
#     for j in i:
#         if j == 1:
#             a = 1
#     if a == 0:
#         num += 1
# print(num)
# NodeList = list(range(G_matrix.shape[0]))
# np.save('../HyperGraphMatrix/' + data + '_nodelist.npy', NodeList)

# order = 10
# G_matrix = np.load('../Hyper_Distance/' + data + '_distance_' + str(order) + '.npy', allow_pickle=True)
# for i in G_matrix:
#     for j in i:
#         print(j)
























