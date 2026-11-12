import random
import time
from collections import deque

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool, freeze_support
from multiprocessing.shared_memory import SharedMemory
from tqdm import trange, tqdm

from Data import get_Graph
from SeedNodesGenerator import H_Degree_SG
from Baseline_S2V import Compete_S2V_DQN, S2VReplayBuffer
from Indicator import Competition_composite_indicator_new
from Indicator import get_avg_HigerOrder_dis
from Indicator import get_node_hyperedge_avgsize
from Diffusion_In_HyperGraph import HG_Diffusion_SICP_pos
from Diffusion_In_HyperGraph import HG_Diffusion_SICP
from Diffusion_In_HyperGraph import Simulations


def hypergraph_to_graph(hypergraph_matrix):
    """将超图矩阵转换为普通图矩阵"""
    num_nodes = hypergraph_matrix.shape[0]
    OR_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    num_hyperedges = hypergraph_matrix.shape[1]
    for e in range(num_hyperedges):
        nodes_in_edge = np.where(hypergraph_matrix[:, e] > 0)[0]
        for i in nodes_in_edge:
            for j in nodes_in_edge:
                if i != j:
                    OR_matrix[i, j] = 1.0
    return OR_matrix


def process_data(data):
    """处理单个数据集的函数"""
    start = time.time()

    # 设置随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 加载数据
    if not os.path.exists('../HyperGraphMatrix/' + data + '.npy'):
        print(f"saving {data} file...")
        G_matrix, NodeList = get_Graph(data)
        np.save('../HyperGraphMatrix/' + data + '.npy', G_matrix)
        np.save('../HyperGraphMatrix/' + data + '_nodelist.npy', NodeList)
        print(f"new data: {data} save successfully！")
    else:
        print(f"loading {data} file...")
        G_matrix = np.load('../HyperGraphMatrix/' + data + '.npy', allow_pickle=True)
        NodeList = np.load('../HyperGraphMatrix/' + data + '_nodelist.npy', allow_pickle=True)
        NodeList = NodeList.tolist()
        print(f"load data: {data} successfully！")

    NodeNumber = len(NodeList)
    SeedNodes_number = int(0.01 * NodeNumber)
    CompeteNodes_number = int(0.01 * NodeNumber)

    # 超图转普通图
    OR_matrix = hypergraph_to_graph(G_matrix)

    # 传播参数
    beta = 0.025
    T = 30

    # DQN训练参数
    num_episodes = 600
    alpha = 0.03
    gamma = 0.9
    epsilon_start = 0.5
    epsilon_end = 0.01
    epsilon_decay = (epsilon_start - epsilon_end) / num_episodes
    simulation_num = 50
    batch_size = 64
    buffer_capacity = 30000
    target_update_freq = 100

    # 设备配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化种子节点与候选池
    SeedNodes = H_Degree_SG(G_matrix, NodeList, SeedNodes_number)
    SeedNodes_pos = [NodeList.index(node) for node in SeedNodes]
    CandidateNodes_pos = [node for node in range(NodeNumber) if node not in SeedNodes_pos]

    # 初始化智能体与环境
    agent = Compete_S2V_DQN(
        epsilon=epsilon_start,
        alpha=alpha,
        gamma=gamma,
        adj_matrix=OR_matrix,
        edge_weights=OR_matrix,
        nodes_number=NodeNumber,
        candidate_pos=CandidateNodes_pos,
        device=device,
        embedding_dim=64
    )

    replay_buffer = S2VReplayBuffer(buffer_capacity)

    # 训练循环
    return_list = []
    loss_list = []
    total_steps = 0

    shm = SharedMemory(create=True, size=G_matrix.nbytes)
    shm_np = np.ndarray(G_matrix.shape, dtype=G_matrix.dtype, buffer=shm.buf)
    np.copyto(shm_np, G_matrix)

    for episode in trange(num_episodes, desc=f"S2V-DQN Training on {data}"):
        current_epsilon = max(epsilon_end, epsilon_start - episode * epsilon_decay)
        agent.epsilon = current_epsilon

        CandidateNodesPool = set(CandidateNodes_pos.copy())
        state = []
        episode_return = 0
        last_CompeteInf = 0
        done = False
        episode_loss = []

        while not done and len(CandidateNodesPool) > 0:
            action = agent.take_action(state, list(CandidateNodesPool))
            CandidateNodesPool.discard(action)

            CompeteInf = Simulations(
                simulation_num,
                HG_Diffusion_SICP_pos,
                shm.name,
                G_matrix.dtype,
                G_matrix.shape,
                NodeList,
                SeedNodes,
                state + [action],
                beta,
                T
            )

            reward = CompeteInf - last_CompeteInf
            last_CompeteInf = CompeteInf
            episode_return += reward

            next_state = state + [action]
            done = len(next_state) >= CompeteNodes_number

            replay_buffer.add(state, action, reward, next_state, done)

            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = agent.update(batch)
                episode_loss.append(loss)

            if total_steps % target_update_freq == 0:
                agent.sync_target_net()
            total_steps += 1

            state = next_state

        avg_loss = np.mean(episode_loss) if episode_loss else 0
        loss_list.append(avg_loss)
        return_list.append(episode_return)
        print(f"Episode {episode + 1}/{num_episodes} | Return: {episode_return:.1f} | Loss: {avg_loss:.4f}")

    shm.close()
    shm.unlink()

    # 生成最终策略
    final_policy = []
    candidate_pool = CandidateNodes_pos.copy()
    while len(final_policy) < CompeteNodes_number and candidate_pool:
        best_node = agent.best_action(final_policy, candidate_pool)
        final_policy.append(best_node)
        candidate_pool.remove(best_node)

    print(f"{data} 最终策略节点索引:", final_policy)

    # 保存结果
    S2V_DQN = [NodeList[idx] for idx in final_policy]
    np.save(f'HCL_synthetic_comparison/New_list/S2V_DQN_RL_({data})_(T_{T})_(beta_{beta}).npy', S2V_DQN)

    end = time.time()
    print(f"算法在{data}上的运行时间：{end - start}")

    # 可视化
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(return_list)
    # plt.title(f'{data} Training Return (time: {end - start:.1f}s)')
    # plt.subplot(1, 2, 2)
    # plt.plot(loss_list)
    # plt.title(f"{data} Training Loss")
    # plt.savefig(f'./New_pic/s2v_dqn_training_Reward&Loss_({data}).png')
    # plt.close('all')


def main():
    # 所有需要运行的数据集列表
    datasets = [
        # "Algebra",
        # "Restaurants-Rev",
        # "Geometry",
        # "Bars-Rev",
        # "Music-Rev",
        # "iAF1260b",
        # "iJO1366",
        "NDC-classes",
        # "soc-hamsterster",
        # "house-committees",
        # "synthetic_HCL_v500_e1000_exp2.0_maxdeg50",
        # "synthetic_HCL_v500_e1500_exp2.0_maxdeg50",
        # "synthetic_HCL_v500_e500_exp2.0_maxdeg50",
        # "synthetic_HCL_v1000_e1000_exp2.0_maxdeg100",
        # "synthetic_HCL_v1000_e1500_exp2.0_maxdeg100",
        # "synthetic_HCL_v1000_e500_exp2.0_maxdeg100",
        # "synthetic_HCL_v1000_e2000_exp2.0_maxdeg100"
    ]

    # 逐个处理数据集
    for data in datasets:
        print(f"\n===== 开始处理数据集: {data} =====")
        process_data(data)
        print(f"===== 数据集 {data} 处理完成 =====\n")


if __name__ == '__main__':
    freeze_support()
    main()



# import random
# import time
# from collections import deque
#
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import os
# from multiprocessing import Pool, freeze_support
# from multiprocessing.shared_memory import SharedMemory
# from tqdm import trange, tqdm
#
# from Data import get_Graph
# from SeedNodesGenerator import H_Degree_SG
# from Baseline_S2V import Compete_S2V_DQN, S2VReplayBuffer
# from Indicator import Competition_composite_indicator_new
# from Indicator import get_avg_HigerOrder_dis
# from Indicator import get_node_hyperedge_avgsize
# from Diffusion_In_HyperGraph import HG_Diffusion_SICP_pos
# from Diffusion_In_HyperGraph import HG_Diffusion_SICP
# from Diffusion_In_HyperGraph import Simulations
#
#
# def hypergraph_to_graph(hypergraph_matrix):
#     """
#     将超图矩阵转换为普通图矩阵
#
#     参数:
#     hypergraph_matrix: numpy数组，超图的关联矩阵，shape为[节点数, 超边数]
#
#     返回:
#     numpy数组，普通图的邻接矩阵，shape为[节点数, 节点数]
#     """
#     num_nodes = hypergraph_matrix.shape[0]
#     # 初始化普通图邻接矩阵
#     OR_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
#
#     # 获取每个超边包含的节点
#     num_hyperedges = hypergraph_matrix.shape[1]
#     for e in range(num_hyperedges):
#         # 获取当前超边中的所有节点
#         nodes_in_edge = np.where(hypergraph_matrix[:, e] > 0)[0]
#
#         # 为这些节点构建完全子图（任意两节点间有边）
#         for i in nodes_in_edge:
#             for j in nodes_in_edge:
#                 if i != j:
#                     # 边权重可以是1，或基于超边的其他计算方式
#                     OR_matrix[i, j] = 1.0
#
#     return OR_matrix
#
#
#
# def main():
#     start = time.time()
#
#     # #################### 设置随机种子 ####################
#     seed = 42
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     os.environ['PYTHONHASHSEED'] = str(seed)
#
#     # #################### 超参数与数据加载 ####################
#     # data = "Algebra"
#     # data = "Restaurants-Rev"
#     # data = "Geometry"
#
#     # data = "Bars-Rev"
#     data = "Music-Rev"
#     # data = "iAF1260b"
#     # data = "iJO1366"
#     # data = "NDC-classes"
#     # data = "soc-hamsterster"
#     # data = 'house-committees'
#     # data = "email-enron"
#     # data = "synthetic_HCL_v500_e1000_exp2.0_maxdeg50"
#     # data = "synthetic_HCL_v500_e1500_exp2.0_maxdeg50"
#     # data = "synthetic_HCL_v500_e500_exp2.0_maxdeg50"
#     # data = "synthetic_HCL_v1000_e1000_exp2.0_maxdeg100"
#     # data = "synthetic_HCL_v1000_e1500_exp2.0_maxdeg100"
#     # data = "synthetic_HCL_v1000_e500_exp2.0_maxdeg100"
#     # data = "synthetic_HCL_v1000_e2000_exp2.0_maxdeg100"
#
#
#     if not os.path.exists('../HyperGraphMatrix/' + data + '.npy'):
#         print("saving file...")
#         G_matrix, NodeList = get_Graph(data)
#         np.save('../HyperGraphMatrix/' + data + '.npy', G_matrix)
#         np.save('../HyperGraphMatrix/' + data + '_nodelist.npy', NodeList)
#         print("new data: " + data + " save successfully！")
#     else:
#         print("loading file...")
#         G_matrix = np.load('../HyperGraphMatrix/' + data + '.npy', allow_pickle=True)
#         NodeList = np.load('../HyperGraphMatrix/' + data + '_nodelist.npy', allow_pickle=True)
#         NodeList = NodeList.tolist()
#         print("load data: " + data + " successfully！")
#
#     NodeNumber = len(NodeList)
#     SeedNodes_number = int(0.01 * NodeNumber)
#     CompeteNodes_number = int(0.01 * NodeNumber)
#
#     # 把超图矩阵转换为clique普通图矩阵
#     OR_matrix = hypergraph_to_graph(G_matrix)
#
#     # 传播参数
#     beta = 0.025
#     T = 30
#
#     # DQN训练参数
#     num_episodes = 400
#     # num_episodes = 600
#     alpha = 0.03
#     gamma = 0.9
#     epsilon_start = 0.5
#     epsilon_end = 0.01
#     epsilon_decay = (epsilon_start - epsilon_end) / num_episodes
#     simulation_num = 50
#     batch_size = 64
#     buffer_capacity = 30000
#     target_update_freq = 100
#
#     # 设备配置
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#
#
#
#     # #################### 初始化种子节点与候选池 ####################
#     # 生成种子节点（示例：基于超图度数）
#     SeedNodes = H_Degree_SG(G_matrix, NodeList, SeedNodes_number)
#     SeedNodes_pos = [NodeList.index(node) for node in SeedNodes]
#     # 生成候选节点池
#     CandidateNodes_pos = [node for node in range(NodeNumber) if node not in SeedNodes_pos]
#
#     # #################### 初始化智能体与环境 ####################
#     agent = Compete_S2V_DQN(
#         epsilon=epsilon_start,
#         alpha=alpha,
#         gamma=gamma,
#         adj_matrix=OR_matrix,
#         edge_weights=OR_matrix,
#         nodes_number=NodeNumber,
#         candidate_pos=CandidateNodes_pos,
#         device=device,
#         embedding_dim=64
#     )
#
#     replay_buffer = S2VReplayBuffer(buffer_capacity)
#
#     # #################### 训练循环 ####################
#     return_list = []
#     loss_list = []
#     total_steps = 0
#
#     shm = SharedMemory(create=True, size=G_matrix.nbytes)
#     shm_np = np.ndarray(G_matrix.shape, dtype=G_matrix.dtype, buffer=shm.buf)
#     np.copyto(shm_np, G_matrix)
#
#     for episode in trange(num_episodes, desc=f"S2V-DQN Training on {data}"):
#         current_epsilon = max(epsilon_end, epsilon_start - episode * epsilon_decay)
#         agent.epsilon = current_epsilon
#
#         CandidateNodesPool = set(CandidateNodes_pos.copy())
#         state = []
#         episode_return = 0
#         last_CompeteInf = 0
#         done = False
#         episode_loss = []
#
#
#         while not done and len(CandidateNodesPool) > 0:
#             action = agent.take_action(state, list(CandidateNodesPool))
#             CandidateNodesPool.discard(action)
#
#             CompeteInf = Simulations(
#                 simulation_num,
#                 HG_Diffusion_SICP_pos,
#                 shm.name,
#                 G_matrix.dtype,
#                 G_matrix.shape,
#                 NodeList,
#                 SeedNodes,
#                 state + [action],
#                 beta,
#                 T
#             )
#
#             reward = CompeteInf - last_CompeteInf
#             last_CompeteInf = CompeteInf
#             episode_return += reward
#
#             next_state = state + [action]
#             done = len(next_state) >= CompeteNodes_number
#
#             replay_buffer.add(state, action, reward, next_state, done)
#
#             # 关键修复：添加经验回放和网络更新
#             if len(replay_buffer) >= batch_size:
#                 batch = replay_buffer.sample(batch_size)
#                 loss = agent.update(batch)
#                 episode_loss.append(loss)  # 记录loss
#
#             if total_steps % target_update_freq == 0:
#                 agent.sync_target_net()
#             total_steps += 1
#
#             state = next_state
#
#         avg_loss = np.mean(episode_loss) if episode_loss else 0
#         loss_list.append(avg_loss)
#         return_list.append(episode_return)
#         print(f"Episode {episode + 1}/{num_episodes} | Return: {episode_return:.1f} | Loss: {avg_loss:.4f}")
#
#     shm.close()
#     shm.unlink()
#
#     # #################### 保存与可视化 ####################
#     # 保存最终策略
#     # 在训练完成后调用best_action生成最终策略
#     final_policy = []
#
#     candidate_pool = CandidateNodes_pos.copy()  # 假设这是候选节点列表
#
#     while len(final_policy) < CompeteNodes_number and candidate_pool:
#         best_node = agent.best_action(final_policy, candidate_pool)
#         final_policy.append(best_node)
#         candidate_pool.remove(best_node)
#
#
#     print("最终策略节点索引:", final_policy)
#
#
#     S2V_DQN = [NodeList[idx] for idx in final_policy]
#     np.save(f'./New_List/S2V_DQN_RL_({data})_(T_{T})_(beta_{beta}).npy', S2V_DQN)
#
#     end = time.time()
#     print(f"算法在{data}上的运行时间：{end - start}")
#
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(return_list)
#     plt.title(f'Training Return_time{end - start}')
#     plt.subplot(1, 2, 2)
#     plt.plot(loss_list)
#     plt.title("Training Loss")
#     # plt.savefig(f'./New_pic/s2v_dqn_training_Reward&Loss_({data}).png')
#     plt.show()
#
# if __name__ == '__main__':
#     freeze_support()
#     main()
