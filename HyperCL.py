import numpy as np
import pandas as pd
import random
import powerlaw
from decimal import Decimal
import matplotlib.pyplot as plt
import os


def getAdjs(df_hyper_matrix, N):
    deg_list = []
    nodes_arr = np.arange(N)
    for node in nodes_arr:
        node_list = []
        edge_set = np.where(df_hyper_matrix.loc[node] == 1)[0]
        for edge in edge_set:
            node_list.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
        node_set = np.unique(np.array(node_list))
        deg_list.append(len(list(node_set)) - 1)
    return np.array(deg_list)


def getDegDistribution(deg_list):
    deg_max = max(deg_list)
    deg_dict = {}
    for i in range(0, int(deg_max) + 1):
        deg_dict[i] = 0
    for deg in range(0, int(deg_max) + 1):
        total = len(list(np.where(np.array(deg_list) == deg)[0]))
        deg_dict[deg] = deg_dict[deg] + total
    return deg_dict


# ---------------------------------------------------------
# 优化说明：原先的 select_node 会因为大量重复计算 sum 而非常慢。
# 这里在 hyperCL 函数中直接使用 numpy.random.choice 替代，极大提升速度。
# ---------------------------------------------------------
def hyperCL(*, hpe_size, hnd_deg):
    edge_aff_dict = {}

    # 归一化节点超度概率
    arr = np.array(hnd_deg) / sum(hnd_deg)
    nodes = np.arange(len(hnd_deg))

    for edge_index in range(0, len(hpe_size)):
        # 确保要抽取的超边大小不超过总节点数，以防报错
        target_size = min(int(hpe_size[edge_index]), len(nodes))

        # 使用 np.random.choice 按照超度概率 arr 无放回地抽取节点
        # replace=False 保证同一条超边内不会出现重复节点
        temp = np.random.choice(nodes, size=target_size, replace=False, p=arr).tolist()

        edge_aff_dict[edge_index] = temp

    return edge_aff_dict


def get_incidence_matrix(*, exp, nsize, esize, hpe_maxdeg):
    hnd_deg = powerlaw.Power_Law(xmin=1, parameters=[exp]).generate_random(nsize)
    hpe_size = np.random.uniform(0, hpe_maxdeg, esize)
    edge_aff_dict = hyperCL(hpe_size=hpe_size, hnd_deg=hnd_deg)
    matrix = np.zeros((len(hnd_deg), len(hpe_size)))
    for edge in edge_aff_dict:
        for node in edge_aff_dict[edge]:
            matrix[node][edge] = 1
    df_hyper_matrix = pd.DataFrame(matrix)
    return matrix


if __name__ == "__main__":
    # 确保保存的文件夹存在，避免报错
    os.makedirs('../HyperGraphMatrix', exist_ok=True)

    # 注意：论文中 HCL 提到的最大超边尺寸是 10，这里为了保持跟你原来的代码一致暂时设为 10
    # 你可以根据实际情况改回 50
    hpe_maxdeg = 10

    # 定义你需要生成的网络规模组合
    network_sizes = [
        {"nsize": 1000, "esize": 1000},
        {"nsize": 2000, "esize": 2000},
        {"nsize": 3000, "esize": 3000}
    ]

    # 定义你需要遍历的 exp 参数
    exp_values = [2.0, 2.5, 3.0]

    # 开始批量生成
    for size_config in network_sizes:
        nsize = size_config["nsize"]
        esize = size_config["esize"]

        for exp in exp_values:
            print(f"==================================================")
            print(f"正在生成: 节点={nsize}, 超边={esize}, exp={exp}")

            # 生成矩阵
            hypergraph = get_incidence_matrix(exp=exp, nsize=nsize, esize=esize, hpe_maxdeg=hpe_maxdeg)

            # 统计没有超边的孤立节点
            # 优化：通过 sum(axis=1) 直接求每行的和，和为0表示该节点没有任何超边
            isolated_nodes = np.sum(np.sum(hypergraph, axis=1) == 0)
            print(f"统计信息: 共 {isolated_nodes} 个节点没有超边")

            # 保存关联矩阵
            matrix_filename = f'../HyperGraphMatrix/synthetic_HCL_v{nsize}_e{esize}_exp{exp}_maxdeg{hpe_maxdeg}.npy'
            np.save(matrix_filename, hypergraph)

            # 保存 NodeList
            nodelist_filename = f'../HyperGraphMatrix/synthetic_HCL_v{nsize}_e{esize}_exp{exp}_maxdeg{hpe_maxdeg}_nodelist.npy'
            NodeList = list(range(hypergraph.shape[0]))
            np.save(nodelist_filename, NodeList)

            print(f"保存成功: {matrix_filename}")

    print("==================================================")
    print("所有网络已全部生成并保存完毕！")