import numpy as np

def get_node_hyperedge_avgsize(G_matrix):
 # 获取所有超边的大小
    HyperEdge_size = []
    trans_matrix = G_matrix.T
    for hyperedge in trans_matrix:
        HyperEdge_size.append(np.sum(hyperedge))
# 节点邻居超边平均size
    node_hyperedge_avgsize = np.zeros(G_matrix.shape[0])
    for i in range(G_matrix.shape[0]):
        pos = np.where(G_matrix[i] == 1)[0]
        hyper_num = len(pos)
        for j in pos:
            node_hyperedge_avgsize[i] += HyperEdge_size[j]
        if hyper_num > 0:
            node_hyperedge_avgsize[i] /= hyper_num
        else:
            node_hyperedge_avgsize[i] = 0.0  # 孤立节点直接置为0

    return node_hyperedge_avgsize


def get_avg_HigerOrder_dis(seednodes_pos, G_matrix, data, max_order, alpha):

    avg_HigerOrder_dis_seednodes = np.zeros(G_matrix.shape[0])
    for node in range(G_matrix.shape[0]):
        if node in seednodes_pos:
            continue
        avg_HigerOrder_dis = 0
        for order in range(1, max_order + 1):
            order_matrix = np.load('../Hyper_Distance/' + data + '_distance_' + str(order) + '.npy', allow_pickle=True)

            # 两种alpha代入方式，选其一即可
            alpha_ratio = 1 / pow(alpha, order)  # new
            # alpha_ratio = pow(alpha, order)  # new_1for order in range(

            avg_Order_dis = 0
            for seednode in seednodes_pos:
                if np.isinf(order_matrix[node][seednode]):
                    # Order_dis = max(order_matrix[node]) + 1
                    Order_dis = 0
                else:
                    Order_dis = order_matrix[node][seednode]
                avg_Order_dis += Order_dis
            avg_Order_dis /= len(seednodes_pos)
            avg_HigerOrder_dis += (alpha_ratio * avg_Order_dis)
        avg_HigerOrder_dis_seednodes[node] = avg_HigerOrder_dis
    return avg_HigerOrder_dis_seednodes


def Competition_composite_indicator_new(seednodes_pos, G_matrix, data, select_num, max_order, ind_alpha):
    node_hyperedge_avgsize = get_node_hyperedge_avgsize(G_matrix)
    avg_HigerOrder_dis_seednodes_new = get_avg_HigerOrder_dis(seednodes_pos, G_matrix, data, max_order, ind_alpha)
    composite_indicator = np.array(node_hyperedge_avgsize) * np.array(avg_HigerOrder_dis_seednodes_new)
    ######### 返回的是候选节点的索引 #########
    Candidate_nodes = np.argsort(-composite_indicator)[:select_num]
    return Candidate_nodes






# import os
# from Data import get_Graph
# from SeedNodesGenerator import H_Degree_SG

# data = "Algebra"
# data = "Bars-Rev"
# data = "Restaurants-Rev"
# data = "Geometry"
# data = "Music-Rev"


# if not os.path.exists('../HyperGraphMatrix/' + data + '.npy'):
#     print("saving file...")
#     G_matrix, NodeList = get_Graph(data)
#     np.save('../HyperGraphMatrix/' + data + '.npy', G_matrix)
#     np.save('../HyperGraphMatrix/' + data + '_nodelist.npy', NodeList)
#     print("new data: " + data + " save successfully！")
# else:
#     print("loading file...")
#     G_matrix = np.load('../HyperGraphMatrix/' + data + '.npy', allow_pickle=True)
#     NodeList = np.load('../HyperGraphMatrix/' + data + '_nodelist.npy', allow_pickle=True)
#     NodeList = NodeList.tolist()
#     print("load data: " + data + " successfully！")

# NodeNumber = len(NodeList)
# SeedNodes_number = int(0.01 * NodeNumber)
#
# SNG_mode = 'H_Degree'
# SeedNodes = H_Degree_SG(G_matrix, NodeList, SeedNodes_number)
#
# SeedNodes_pos = []
# for i in SeedNodes:
#     SeedNodes_pos.append(NodeList.index(i))

# node_hyperedge_avgsize, node_neighbor_num, avg_dis_seednodes, avg_HigerOrder_dis_seednodes = node_indicator(SeedNodes_pos, G_matrix)
# print(node_hyperedge_avgsize)
# print(node_neighbor_num)
# print(avg_dis_seednodes)
# print(avg_HigerOrder_dis_seednodes)


