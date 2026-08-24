import numpy as np
import os

from tqdm import trange
from tqdm import tqdm

from Diffusion_In_HyperGraph import HG_Diffusion_SICP
from Diffusion_In_HyperGraph import HG_Diffusion_SICP_pos
from Diffusion_In_HyperGraph import Simulations
from multiprocessing.shared_memory import SharedMemory



# 用于根据超图的超度选择竞争节点，返回节点名称,返回list
def H_Degree(G_matrix, NodeList, SeedNodes, CompeteNodeNumber):
    # 计算每个节点的超度，即 G_matrix 每一行元素的和
    hyper_degree = np.sum(G_matrix, axis=1)
    # 创建一个布尔掩码，用于标记不在种子节点列表中的节点
    candidate_mask = ~np.isin(NodeList, SeedNodes)
    # 根据掩码筛选出候选节点
    candidates = np.array(NodeList)[candidate_mask]
    # 对候选节点的超度进行降序排序，并获取排序后的索引
    sorted_indices = np.argsort(-hyper_degree[candidate_mask])
    # 选取超度最大的 CompeteNodeNumber 个节点作为竞争节点，并转换为列表返回
    return candidates[sorted_indices[:CompeteNodeNumber]].tolist()



# 用于根据超图中节点的度选择竞争节点，返回节点名称,返回list
def Degree(G_matrix,NodeList,SeedNodes,CompeteNodeNumber):
    # 每个节点的度
    DegreeList = []

    for i in range(len(G_matrix)):
        HyperEdge_index = np.where(G_matrix[i] == 1)[0]
        #节点的邻居索引
        Neighbors_index = []
        for j in HyperEdge_index:
            Neighbors_index += list(np.where(G_matrix[:,j] == 1)[0])
        Neighbors_index = np.unique(Neighbors_index)
        DegreeList.append(len(Neighbors_index) - 1)

    # 将HyperEdgeList从大到小排序后的索引
    HyperEdgeList_sort_index = np.argsort(-np.array(DegreeList))
    number = 0
    CompeteNodes = []

    for i in HyperEdgeList_sort_index:
        if number < CompeteNodeNumber:
            if NodeList[i] not in SeedNodes:
                CompeteNodes.append(NodeList[i])
                number += 1
        else:
            break

    return CompeteNodes




#输入的是种子节点序号，候选节点索引，返回的是节点索引,返回list
def Indicator_selection_Greedy(
        G_matrix, NodeList, SeedNodes, CandidateNodes_pos: list, CompeteNodeNumber,
        greedy_times, beta, T, checkpoint_path=None):
    greedy_pos_list = []
    pool = CandidateNodes_pos.copy()

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        greedy_pos_list = np.load(checkpoint_path, allow_pickle=True).tolist()
        pool = [node_pos for node_pos in pool if node_pos not in set(greedy_pos_list)]
        if len(greedy_pos_list) >= CompeteNodeNumber:
            return greedy_pos_list[:CompeteNodeNumber]


    # for _ in trange(CompeteNodeNumber, desc="indicator_selection_greedy"):
    #     max_inf = 0
    #     max_node_pos = 0
    #     for node_pos in pool:
    #         greedy_pos_list_copy = greedy_pos_list.copy()
    #         result_inf = 0
    #         for i in range(greedy_times):
    #             a, b, c, inf = HG_Diffusion_SICP_pos(G_matrix, NodeList, SeedNodes, greedy_pos_list_copy + [node_pos], beta, T)
    #             result_inf += inf
    #         result_inf /= greedy_times
    #         if result_inf > max_inf:
    #             max_inf = result_inf
    #             max_node_pos = node_pos
    #     pool.remove(max_node_pos)
    #     greedy_pos_list.append(max_node_pos)

    # 开辟共享内存
    shm = SharedMemory(create=True, size=G_matrix.nbytes)
    shm_np = np.ndarray(G_matrix.shape, dtype=G_matrix.dtype, buffer=shm.buf)
    np.copyto(shm_np, G_matrix)

    for _ in trange(len(greedy_pos_list), CompeteNodeNumber, desc="indicator_selection_greedy"):
        max_inf = 0
        max_node_pos = 0

        for node_pos in pool:
            greedy_pos_list_copy = greedy_pos_list.copy()
            result_inf = Simulations(
                greedy_times,
                HG_Diffusion_SICP_pos,
                shm.name,  # 共享内存名称
                G_matrix.dtype,  # 数据类型
                G_matrix.shape,  # 矩阵形状
                NodeList,
                SeedNodes,
                greedy_pos_list_copy + [node_pos],
                beta,
                T
            )
            if result_inf > max_inf:
                max_inf = result_inf
                max_node_pos = node_pos

        pool.remove(max_node_pos)
        greedy_pos_list.append(max_node_pos)
        if checkpoint_path is not None:
            np.save(checkpoint_path, np.array(greedy_pos_list, dtype=int))

    # 关闭共享内存
    shm.close()
    shm.unlink()

    return greedy_pos_list



def HADP(G_matrix, NodeList, SeedNodes, CompeteNodeNumber):

    G_matrix_copy = G_matrix.copy()

    for i in SeedNodes:
        idx = NodeList.index(i)
        for j in range(len(G_matrix_copy[idx])):
            G_matrix_copy[idx][j] = 0

    HADP_list_idx = []

    for i in trange(CompeteNodeNumber):
        Degree_max = 0
        Degree_max_idx = -1
        for j in range(len(G_matrix_copy)):
            HyperEdge_index = np.where(G_matrix_copy[j] == 1)[0]
            Degree_list = []
            for k in HyperEdge_index:
                Degree_list += list(np.where(G_matrix_copy[:, k] == 1)[0])
            Degree_list = np.unique(Degree_list)
            if len(Degree_list) > Degree_max:
                Degree_max = len(Degree_list)
                Degree_max_idx = j
        HADP_list_idx.append(Degree_max_idx)
        for j in range(len(G_matrix_copy[Degree_max_idx])):
            G_matrix_copy[Degree_max_idx][j] = 0

    CompeteNodes = []
    for i in HADP_list_idx:
        CompeteNodes.append(NodeList[i])

    return CompeteNodes








def CI(G_matrix, NodeList, SeedNodes, CompeteNodeNumber):
    CI_list = []
    G_matrix_copy = G_matrix.copy()

    for i in trange(len(G_matrix)):
        HyperEdge_index = np.where(G_matrix_copy[i] == 1)[0]
        ci = len(HyperEdge_index) - 1
        ball_nodes_idx = []
        for j in HyperEdge_index:
            ball_nodes_idx += list(np.where(G_matrix_copy[:, j] == 1)[0])
        ball_nodes_idx = np.unique(ball_nodes_idx)
        res = 0
        for j in ball_nodes_idx:
            res += (len(np.where(G_matrix_copy[j] == 1)[0]) - 1)
        ci += res
        CI_list.append(ci)

    CI_sort_idx = np.argsort(- np.array(CI_list))

    CompeteNodes = []
    idx = 0
    while True:
        if len(CompeteNodes) >= CompeteNodeNumber:
            break

        if NodeList[CI_sort_idx[idx]] not in SeedNodes:
            CompeteNodes.append(NodeList[CI_sort_idx[idx]])

        idx += 1

    return CompeteNodes




