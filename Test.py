from Data import get_Graph

from Baseline_Algorithm import H_Degree
from Baseline_Algorithm import Degree

from SeedNodesGenerator import H_Degree_SG
from SeedNodesGenerator import Degree_SG


from Diffusion_In_HyperGraph import HG_Diffusion_SICP
from Diffusion_In_HyperGraph import HG_Diffusion_SICP_pos

from Diffusion_In_HyperGraph import Simulations

import Diffusion_In_HyperGraph
# import Diffusion_In_HyperGraph_Optimized



from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from multiprocessing.shared_memory import SharedMemory

def main():
    # data = "Algebra"
    data = "Bars-Rev"
    # data = "Restaurants-Rev"
    # data = "Geometry"
    # data = "Music-Rev"
    # data = "fb-pages-politician"
    # data = "fb-pages-company"
    # data = "fb-pages-tvshow"
    # data = "soc-anybeat"
    # data = "socfb-Oberlin44"
    # data = "socfb-Simmons81"
    # data = "socfb-Smith60"
    # data = "socfb-USFCA72"
    # data = "soc-hamsterster"

    if not os.path.exists('../HyperGraphMatrix/' + data + '.npy'):
        print("saving file...")
        G_matrix, NodeList = get_Graph(data)
        np.save('../HyperGraphMatrix/' + data + '.npy', G_matrix)
        np.save('../HyperGraphMatrix/' + data + '_nodelist.npy', NodeList)
        print("new data: " + data + " save successfully！")
    else:
        print("loading file...")
        G_matrix = np.load('../HyperGraphMatrix/' + data + '.npy', allow_pickle=True)
        NodeList = np.load('../HyperGraphMatrix/' + data + '_nodelist.npy', allow_pickle=True)
        NodeList = NodeList.tolist()
        print("load data: " + data + " successfully！")

    SeedNodes = H_Degree_SG(G_matrix, NodeList, 24)
    BlockNodes = H_Degree(G_matrix, NodeList, SeedNodes, 24)
    beta = 0.05
    T = 30

    result_times = 1000

    total_num_nodes = 0
    total_inactive_nodes = 0
    total_rumor_nodes = 0
    total_block_nodes = 0

    # begin = time.time()
    # for _ in tqdm(range(result_times)):
    #     # num_nodes, inactive_nodes, rumor_nodes, block_nodes = Diffusion_In_HyperGraph.HG_Diffusion_SICP(G_matrix, NodeList, SeedNodes, BlockNodes, beta, T)
    #     num_nodes, inactive_nodes, rumor_nodes, block_nodes = Diffusion_In_HyperGraph.HG_Diffusion_SICP(G_matrix, NodeList, SeedNodes, BlockNodes, beta, T)
    #     total_num_nodes += num_nodes
    #     total_inactive_nodes += inactive_nodes
    #     total_rumor_nodes += rumor_nodes
    #     total_block_nodes += block_nodes
    # end = time.time()
    # print("Time:", end - begin)
    # print('num_nodes:' + str(total_num_nodes / result_times))
    # print('inactive_nodes:' + str(total_inactive_nodes / result_times))
    # print('rumor_nodes:' + str(total_rumor_nodes / result_times))
    # print('block_nodes:' + str(total_block_nodes / result_times))

    # begin = time.time()
    # block_nodes = Simulations(10000, HG_Diffusion_SICP, G_matrix, NodeList, SeedNodes, BlockNodes, beta, T)
    # end = time.time()

    begin = time.time()
    shm = SharedMemory(create=True, size=G_matrix.nbytes)
    shm_np = np.ndarray(G_matrix.shape, dtype=G_matrix.dtype, buffer=shm.buf)
    np.copyto(shm_np, G_matrix)
    # 修改调用 Simulations 的方式，传递共享内存信息

    block_nodes = Simulations(
        10000,
        HG_Diffusion_SICP,
        shm.name,  # 共享内存名称
        G_matrix.dtype,  # 数据类型
        G_matrix.shape,  # 矩阵形状
        NodeList,
        SeedNodes,
        BlockNodes,
        beta,
        T
    )
    shm.close()
    shm.unlink()
    end = time.time()
    print("Time:", end - begin)
    print(f'block_nodes:{block_nodes}')


    # plt.figure(figsize=(10, 5))
    # plt.scatter(range(G_matrix.shape[1]),sorted(np.sum(G_matrix, axis=0)),color = '#83639f',s = 15,alpha=0.5)
    # plt.xlabel('hypergragh')
    # plt.ylabel('size')
    # plt.show()

if __name__ == '__main__':
    main()
