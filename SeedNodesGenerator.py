import numpy as np



#########返回的数组应为节点序号，不是节点索引#########


def H_Degree_SG(G_matrix,NodeList,SeedNodeNumber):
    #每个节点的超度
    HyperEdgeList = []

    for i in range(len(G_matrix)):
        HyperEdgeNumber = len(np.where(G_matrix[i] == 1)[0])
        HyperEdgeList.append(HyperEdgeNumber)

    #将HyperEdgeList从大到小排序后的索引
    HyperEdgeList_sort_index = np.argsort(-np.array(HyperEdgeList))
    SeedNodes = []

    for i in range(SeedNodeNumber):
        SeedNodes.append(NodeList[HyperEdgeList_sort_index[i]])

    return SeedNodes




def Degree_SG(G_matrix,NodeList,SeedNodeNumber):
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
    SeedNodes = []

    for i in range(SeedNodeNumber):
        SeedNodes.append(NodeList[HyperEdgeList_sort_index[i]])

    return SeedNodes