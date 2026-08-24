import random
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool, freeze_support
from multiprocessing.shared_memory import SharedMemory
from tqdm import trange, tqdm

from Data import get_Graph
from SeedNodesGenerator import H_Degree_SG
from CompeteAlgorithm import Compete_DQN
from CompeteAlgorithm import Compete_DQN
from CompeteAlgorithm import ReplayBuffer
# from CompeteAlgorithm import PrioritizedReplayBuffer
from Indicator import Competition_composite_indicator_new
from Indicator import get_avg_HigerOrder_dis
from Indicator import get_node_hyperedge_avgsize
from Diffusion_In_HyperGraph import HG_Diffusion_SICP_pos
from Diffusion_In_HyperGraph import HG_Diffusion_SICP
from Diffusion_In_HyperGraph import Simulations


def run_experiment(data):  # 改为接受data参数的函数
    start = time.time()

    # #################### 设置随机种子 ####################
    seed = 42  # 固定种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 启用确定性卷积算法
    torch.backends.cudnn.benchmark = False  # 关闭基准优化
    os.environ['PYTHONHASHSEED'] = str(seed)  # 固定哈希种子

    # #################### 超参数与数据加载 ####################
    # 所有数据通过参数传入，不再硬编码

    # 加载超图数据
    if not os.path.exists(f'../HyperGraphMatrix/{data}.npy'):
        G_matrix, NodeList = get_Graph(data)
        np.save(f'../HyperGraphMatrix/{data}.npy', G_matrix)
        np.save(f'../HyperGraphMatrix/{data}_nodelist.npy', NodeList)
    else:
        G_matrix = np.load(f'../HyperGraphMatrix/{data}.npy', allow_pickle=True)
        NodeList = np.load(f'../HyperGraphMatrix/{data}_nodelist.npy', allow_pickle=True).tolist()

    NodeNumber = len(NodeList)
    SeedNodes_number = int(0.01 * NodeNumber)
    CompeteNodes_number = int(0.01 * NodeNumber)
    CandidateNodes_Ratio = 0.2

    # 这个方法的代号
    # 2为DQN，11为selection_DQN
    serial_number = 11

    # 传播参数
    beta = 0.025  # 传播概率
    T = 30  # 传播轮次

    # 指标参数
    Indicator_Order = 10
    Indicator_Alpha = 0.5

    # DQN训练参数
    # num_episodes = 400  # 训练轮次
    num_episodes = 600
    # num_episodes = 800
    alpha = 0.03  # 学习率
    # gamma = 0.9  # 代号2的折扣因子
    gamma = 0.9 # 代号11的折扣因子
    epsilon_start = 0.5  # 初始探索率
    epsilon_end = 0.01  # 最小探索率
    epsilon_decay = (epsilon_start - epsilon_end) / num_episodes  # 探索率的线性衰减率
    simulation_num = 500  # 计算reward时模拟的次数
    batch_size = 64  # 经验回放批次大小
    buffer_capacity = 30000  # 经验池容量
    target_update_freq = 100  # 目标网络同步频率
    state_dim = 2 * CompeteNodes_number  # 状态的维度，为竞争节点数量的2倍

    # 设备配置,如果有可用的GPU，则使用GPU进行计算，否则使用CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # #################### 初始化种子节点与候选池 ####################
    # 生成种子节点（示例：基于超图度数）
    SeedNodes = H_Degree_SG(G_matrix, NodeList, SeedNodes_number)
    SeedNodes_pos = [NodeList.index(node) for node in SeedNodes]

    # 生成候选节点池
    CandidateNodes_pos = Competition_composite_indicator_new(SeedNodes_pos, G_matrix, data,
                                                             int(CandidateNodes_Ratio * NodeNumber), Indicator_Order,
                                                             Indicator_Alpha)
    print(f"{data}的候选节点池: {CandidateNodes_pos}")
    # #################### 环境与代理初始化 ####################
    agent = Compete_DQN(
        epsilon=epsilon_start,
        alpha=alpha,
        gamma=gamma,
        state_dim=state_dim,
        action_dim=len(CandidateNodes_pos),
        node_hyperedge_avgsize=get_node_hyperedge_avgsize(G_matrix),
        avg_HigerOrder_dis_seednodes=get_avg_HigerOrder_dis(SeedNodes_pos, G_matrix, data, Indicator_Order,
                                                            Indicator_Alpha),
        candidate_nodes_pos=CandidateNodes_pos,
        device=device
    )
    replay_buffer = ReplayBuffer(buffer_capacity)

    # #################### 开辟共享内存 ####################
    shm = SharedMemory(create=True, size=G_matrix.nbytes)
    shm_np = np.ndarray(G_matrix.shape, dtype=G_matrix.dtype, buffer=shm.buf)
    np.copyto(shm_np, G_matrix)

    # #################### 训练循环 ####################
    # 初始化一个空列表，用于存储每一轮训练（episode）的总回报
    return_list = []
    total_steps = 0
    loss_list = []  # 用于记录每轮平均loss
    # 外层循环，遍历预设的训练轮数（num_episodes）
    for episode in trange(num_episodes, desc=data + "__DQN Training"):

        current_epsilon = max(epsilon_end, epsilon_start - episode * epsilon_decay)  # 计算当前ε值，确保不低于最小值
        # simulation_num = int(simulation_num_start * (decay_rate ** episode))  # 指数递减simulation次数
        agent.epsilon = current_epsilon  # 更新Agent的探索率
        CandidateNodesPool = set(CandidateNodes_pos.copy())  # 复制候选节点索引列表，并转换为集合类型,集合类型便于进行元素的删除操作，确保每个动作选择的节点是唯一的
        state = []  # 初始化当前状态为空列表，后续会随着动作选择逐渐更新
        episode_return = 0  # 初始化当前训练轮次的总回报为 0
        last_CompeteInf = 0  # 初始化上一次信息传播后的感染节点数量为 0
        done = False  # 初始化一个布尔变量，标记当前训练轮次是否结束，初始为 False
        episode_loss = []  # 记录本轮所有batch的loss
        # 内层循环，只要当前训练轮次未结束，就持续执行循环体中的操作
        while not done:
            # 调用 Compete_DQN 类的 take_action 方法，根据当前状态和候选节点池选择一个动作（即选择一个节点），修改动作选择返回值和经验池存储的action
            action_node, action_index = agent.take_action(state, list(CandidateNodesPool))  # 接收节点位置和索引
            CandidateNodesPool.discard(action_node)  # 使用节点位置移除候选池

            CompeteInf = Simulations(
                simulation_num,
                HG_Diffusion_SICP_pos,
                shm.name,  # 共享内存名称
                G_matrix.dtype,  # 数据类型
                G_matrix.shape,  # 矩阵形状
                NodeList,
                SeedNodes,
                state + [action_node],
                beta,
                T
            )

            reward = CompeteInf - last_CompeteInf  # 计算奖励，奖励定义为当前感染节点数量与上一次感染节点数量的差值
            last_CompeteInf = CompeteInf  # 更新上一次感染节点数量
            next_state = state + [action_node]  # 更新下一个状态，将选择的动作（节点）添加到当前状态中
            done = (
                        len(next_state) == CompeteNodes_number)  # 判断当前训练轮次是否结束，如果已选择的节点数量达到 CompeteNodes_number，则将 done 设置为 True

            state_encoded = agent._encode_state(state)  # 调用 Compete_DQN 类的 _encode_state 方法，将当前状态编码为适合神经网络输入的格式
            next_state_encoded = agent._encode_state(
                next_state)  # 调用 Compete_DQN 类的 _encode_state 方法，将下一个状态编码为适合神经网络输入的格式

            replay_buffer.add(state_encoded, action_index, reward, next_state_encoded,
                              done)  # 将编码后的状态、动作、奖励、下一个状态和结束标志添加到经验回放缓冲区中

            if len(replay_buffer) >= batch_size:  # 检查经验回放缓冲区中的经验数量是否达到 batch_size
                batch = replay_buffer.sample(batch_size)  # 从经验回放缓冲区中随机采样 batch_size 个经验
                loss = agent.update(zip(*batch))  # 调用 Compete_DQN 类的 update 方法，根据采样的经验更新神经网络的参数
                episode_loss.append(loss)  # 收集loss

            if total_steps % target_update_freq == 0:  # 每经过target_update_freq步，将Q网络的参数同步到目标网络
                agent.sync_target_net()
            total_steps += 1

            state = next_state  # 将下一个状态更新为当前状态
            episode_return += reward  # 将当前步骤的奖励累加到当前训练轮次的总回报中

        # 计算本轮平均loss
        if episode_loss:
            avg_loss = np.mean(episode_loss)
        else:
            avg_loss = 0  # 无更新时记为0
        loss_list.append(avg_loss)

        return_list.append(episode_return)  # 将当前训练轮次的总回报添加到 return_list 中

        print(f'{data}的loss: {avg_loss}')

    # #################### 关闭共享内存 ####################
    shm.close()
    shm.unlink()

    # #################### 保存最优策略 ####################
    # 提取最佳动的索引
    Indicator_selection_RL_pos = []
    current_state = []
    CandidateNodesPool = set(CandidateNodes_pos.copy())  # 初始化候选池
    for _ in range(CompeteNodes_number):
        available_actions = list(CandidateNodesPool)
        if not available_actions:
            break  # 候选池已空
        action = agent.best_action(current_state, available_actions)
        CandidateNodesPool.discard(action)
        current_state.append(action)
        Indicator_selection_RL_pos.append(action)

    # 转换为节点序号并保存
    Indicator_selection_RL = [NodeList[idx] for idx in Indicator_selection_RL_pos]
    np.save(
        f'HCL_synthetic_comparison/New_list/indicator_selection_RL_{serial_number}_({data})_(T_{T})_(beta_{beta})_(ratio_{CandidateNodes_Ratio}).npy',
        Indicator_selection_RL)

    end = time.time()
    print(f"算法在{data}上的运行时间：{end - start}")

    # #################### 可视化奖励和损失曲线 ####################
    # plt.figure(figsize=(12, 5))
    #
    # # 奖励曲线
    # plt.subplot(1, 2, 1)
    # plt.plot(return_list)
    # plt.xlabel('Episode')
    # plt.ylabel('Total Reward')
    # plt.title(f'Training Reward_{data}_{end - start:.2f}s')
    #
    # # 损失曲线
    # plt.subplot(1, 2, 2)
    # plt.plot(loss_list)
    # plt.xlabel('Episode')
    # plt.ylabel('Q-Net Loss')
    # plt.title(f'Training Loss_{data}')
    #
    # plt.tight_layout()
    # plt.savefig(
    #     f'./New_pic/RL_{serial_number}_Training_Reward&Loss_({data})_(ratio_{CandidateNodes_Ratio}).png')  # 改为保存图片避免交互阻塞
    # plt.close('all')  # 关闭图像释放资源


def main():
    # 所有需要运行的数据集列表（包含原注释中的所有数据）
    datasets = [
        # "Algebra",
        # "Restaurants-Rev",
        # "Geometry",
        # "Bars-Rev",
        # "Music-Rev",
        # "iAF1260b",
        # "iJO1366",
        # "NDC-classes",
        # "soc-hamsterster",
        # "house-committees",
        # "synthetic_HCL_v500_e1000_exp2.0_maxdeg50",
        # "synthetic_HCL_v500_e1500_exp2.0_maxdeg50",
        # "synthetic_HCL_v500_e500_exp2.0_maxdeg50",
        # "synthetic_HCL_v1000_e1000_exp2.0_maxdeg100",
        # "synthetic_HCL_v1000_e1500_exp2.0_maxdeg100",
        "synthetic_HCL_v1000_e500_exp2.0_maxdeg100"
    ]

    # 遍历所有数据集执行实验
    for data in datasets:
        print(f"\n===== 开始处理数据集: {data} =====")
        run_experiment(data)
        print(f"===== 数据集 {data} 处理完成 =====\n")


if __name__ == '__main__':
    freeze_support()
    main()

# def main():
#     start = time.time()
#
#     # #################### 设置随机种子 ####################
#     seed = 42  # 固定种子
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True  # 启用确定性卷积算法
#     torch.backends.cudnn.benchmark = False  # 关闭基准优化
#     os.environ['PYTHONHASHSEED'] = str(seed)  # 固定哈希种子
#
#
#     # #################### 超参数与数据加载 ####################
#     # data = "Algebra"
#     # data = "Restaurants-Rev"
#     # data = "Geometry"
#     # data = "Bars-Rev"
#     data = "Music-Rev"
#     # data = "iAF1260b"
#     # data = "iJO1366"
#     # data = "NDC-classes"
#     # data = "soc-hamsterster"
#     # data = 'house-committees'
#     # data = "synthetic_HCL_v500_e1000_exp2.0_maxdeg50"
#     # data = "synthetic_HCL_v500_e1500_exp2.0_maxdeg50"
#     # data = "synthetic_HCL_v500_e500_exp2.0_maxdeg50"
#     # data = "synthetic_HCL_v1000_e1000_exp2.0_maxdeg100"
#     # data = "synthetic_HCL_v1000_e1500_exp2.0_maxdeg100"
#     # data = "synthetic_HCL_v1000_e500_exp2.0_maxdeg100"
#     # data = "synthetic_HCL_v1000_e2000_exp2.0_maxdeg100"
#
#
#     # 加载超图数据
#     if not os.path.exists(f'../HyperGraphMatrix/{data}.npy'):
#         G_matrix, NodeList = get_Graph(data)
#         np.save(f'../HyperGraphMatrix/{data}.npy', G_matrix)
#         np.save(f'../HyperGraphMatrix/{data}_nodelist.npy', NodeList)
#     else:
#         G_matrix = np.load(f'../HyperGraphMatrix/{data}.npy', allow_pickle=True)
#         NodeList = np.load(f'../HyperGraphMatrix/{data}_nodelist.npy', allow_pickle=True).tolist()
#
#     NodeNumber = len(NodeList)
#     SeedNodes_number = int(0.01 * NodeNumber)
#     CompeteNodes_number = int(0.01 * NodeNumber)
#     CandidateNodes_Ratio = 1
#
#     # 这个方法的代号
#     # 2为DQN，11为selection_DQN
#     serial_number = 2
#
#     # 传播参数
#     beta = 0.025  # 传播概率
#     T = 30  # 传播轮次
#
#     # 指标参数
#     Indicator_Order = 10
#     Indicator_Alpha = 0.5
#
#     # DQN训练参数
#     num_episodes = 400  # 训练轮次
#     # num_episodes = 600
#     # num_episodes = 800
#     alpha = 0.01  # 学习率
#     gamma = 0.9  # 代号2的折扣因子
#     # gamma = 0.6  # 代号11的折扣因子
#     epsilon_start = 0.5  # 初始探索率
#     epsilon_end = 0.01  # 最小探索率
#     epsilon_decay = (epsilon_start - epsilon_end) / num_episodes  # 探索率的线性衰减率
#     simulation_num = 50  # 计算reward时模拟的次数
#     batch_size = 64  # 经验回放批次大小
#     buffer_capacity = 30000  # 经验池容量
#     target_update_freq = 100  # 目标网络同步频率
#     state_dim = 2 * CompeteNodes_number  # 状态的维度，为竞争节点数量的2倍
#
#
#
#     # 设备配置,如果有可用的GPU，则使用GPU进行计算，否则使用CPU
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#
#
#     # #################### 初始化种子节点与候选池 ####################
#     # 生成种子节点（示例：基于超图度数）
#     SeedNodes = H_Degree_SG(G_matrix, NodeList, SeedNodes_number)
#     SeedNodes_pos = [NodeList.index(node) for node in SeedNodes]
#
#     # 生成候选节点池
#     CandidateNodes_pos = Competition_composite_indicator_new(SeedNodes_pos, G_matrix, data, int(CandidateNodes_Ratio * NodeNumber), Indicator_Order, Indicator_Alpha)
#     print(CandidateNodes_pos)
#     # #################### 环境与代理初始化 ####################
#     agent = Compete_DQN(
#         epsilon=epsilon_start,
#         alpha=alpha,
#         gamma=gamma,
#         state_dim=state_dim,
#         action_dim=len(CandidateNodes_pos),
#         node_hyperedge_avgsize=get_node_hyperedge_avgsize(G_matrix),
#         avg_HigerOrder_dis_seednodes=get_avg_HigerOrder_dis(SeedNodes_pos, G_matrix, data, Indicator_Order, Indicator_Alpha),
#         candidate_nodes_pos=CandidateNodes_pos,
#         device=device
#     )
#     replay_buffer = ReplayBuffer(buffer_capacity)
#
#     # #################### 开辟共享内存 ####################
#     shm = SharedMemory(create=True, size=G_matrix.nbytes)
#     shm_np = np.ndarray(G_matrix.shape, dtype=G_matrix.dtype, buffer=shm.buf)
#     np.copyto(shm_np, G_matrix)
#
#     # #################### 训练循环 ####################
#     # 初始化一个空列表，用于存储每一轮训练（episode）的总回报
#     return_list = []
#     total_steps = 0
#     loss_list = []  # 用于记录每轮平均loss
#     # 外层循环，遍历预设的训练轮数（num_episodes）
#     for episode in trange(num_episodes, desc=data + "__DQN Training"):
#
#         current_epsilon = max(epsilon_end, epsilon_start - episode * epsilon_decay)  # 计算当前ε值，确保不低于最小值
#         # simulation_num = int(simulation_num_start * (decay_rate ** episode))  # 指数递减simulation次数
#         agent.epsilon = current_epsilon  # 更新Agent的探索率
#         CandidateNodesPool = set(CandidateNodes_pos.copy())  # 复制候选节点索引列表，并转换为集合类型,集合类型便于进行元素的删除操作，确保每个动作选择的节点是唯一的
#         state = []  # 初始化当前状态为空列表，后续会随着动作选择逐渐更新
#         episode_return = 0  # 初始化当前训练轮次的总回报为 0
#         last_CompeteInf = 0  # 初始化上一次信息传播后的感染节点数量为 0
#         done = False  # 初始化一个布尔变量，标记当前训练轮次是否结束，初始为 False
#         episode_loss = []  # 记录本轮所有batch的loss
#         # 内层循环，只要当前训练轮次未结束，就持续执行循环体中的操作
#         while not done:
#             # 调用 Compete_DQN 类的 take_action 方法，根据当前状态和候选节点池选择一个动作（即选择一个节点），修改动作选择返回值和经验池存储的action
#             action_node, action_index = agent.take_action(state, list(CandidateNodesPool))  # 接收节点位置和索引
#             CandidateNodesPool.discard(action_node)  # 使用节点位置移除候选池
#
#             CompeteInf = Simulations(
#                 simulation_num,
#                 HG_Diffusion_SICP_pos,
#                 shm.name,  # 共享内存名称
#                 G_matrix.dtype,  # 数据类型
#                 G_matrix.shape,  # 矩阵形状
#                 NodeList,
#                 SeedNodes,
#                 state + [action_node],
#                 beta,
#                 T
#             )
#
#
#             reward = CompeteInf - last_CompeteInf  # 计算奖励，奖励定义为当前感染节点数量与上一次感染节点数量的差值
#             last_CompeteInf = CompeteInf  # 更新上一次感染节点数量
#             next_state = state + [action_node]  # 更新下一个状态，将选择的动作（节点）添加到当前状态中
#             done = (len(next_state) == CompeteNodes_number)  # 判断当前训练轮次是否结束，如果已选择的节点数量达到 CompeteNodes_number，则将 done 设置为 True
#
#             state_encoded = agent._encode_state(state)  # 调用 Compete_DQN 类的 _encode_state 方法，将当前状态编码为适合神经网络输入的格式
#             next_state_encoded = agent._encode_state(next_state)  # 调用 Compete_DQN 类的 _encode_state 方法，将下一个状态编码为适合神经网络输入的格式
#
#             replay_buffer.add(state_encoded, action_index, reward, next_state_encoded, done)  # 将编码后的状态、动作、奖励、下一个状态和结束标志添加到经验回放缓冲区中
#
#             if len(replay_buffer) >= batch_size:  # 检查经验回放缓冲区中的经验数量是否达到 batch_size
#                 batch = replay_buffer.sample(batch_size)  # 从经验回放缓冲区中随机采样 batch_size 个经验
#                 loss = agent.update(zip(*batch))  # 调用 Compete_DQN 类的 update 方法，根据采样的经验更新神经网络的参数
#                 episode_loss.append(loss)  # 收集loss
#
#             if total_steps % target_update_freq == 0:  # 每经过target_update_freq步，将Q网络的参数同步到目标网络
#                 agent.sync_target_net()
#             total_steps += 1
#
#             state = next_state  # 将下一个状态更新为当前状态
#             episode_return += reward  # 将当前步骤的奖励累加到当前训练轮次的总回报中
#
#         # 计算本轮平均loss
#         if episode_loss:
#             avg_loss = np.mean(episode_loss)
#         else:
#             avg_loss = 0  # 无更新时记为0
#         loss_list.append(avg_loss)
#
#         return_list.append(episode_return)  # 将当前训练轮次的总回报添加到 return_list 中
#
#         print('loss:' + str(avg_loss))
#
#
#
#     # #################### 关闭共享内存 ####################
#     shm.close()
#     shm.unlink()
#
#
#     # #################### 保存最优策略 ####################
#     # 提取最佳动的索引
#     Indicator_selection_RL_pos = []
#     current_state = []
#     CandidateNodesPool = set(CandidateNodes_pos.copy())  # 初始化候选池
#     for _ in range(CompeteNodes_number):
#         available_actions = list(CandidateNodesPool)
#         if not available_actions:
#             break  # 候选池已空
#         action = agent.best_action(current_state, available_actions)
#         CandidateNodesPool.discard(action)
#         current_state.append(action)
#         Indicator_selection_RL_pos.append(action)
#
#     # 转换为节点序号并保存
#     Indicator_selection_RL = [NodeList[idx] for idx in Indicator_selection_RL_pos]
#     np.save(f'./New_List/indicator_selection_RL_{serial_number}_({data})_(T_{T})_(beta_{beta})_(ratio_{CandidateNodes_Ratio}).npy', Indicator_selection_RL)
#
#     end = time.time()
#     print(f"算法在{data}上的运行时间：{end - start}")
#
#     # #################### 可视化奖励和损失曲线 ####################
#     plt.figure(figsize=(12, 5))
#
#     # 奖励曲线
#     plt.subplot(1, 2, 1)
#     plt.plot(return_list)
#     plt.xlabel('Episode')
#     plt.ylabel('Total Reward')
#     plt.title('Training Reward_' + data + str(end - start))
#
#     # 损失曲线
#     plt.subplot(1, 2, 2)
#     plt.plot(loss_list)
#     plt.xlabel('Episode')
#     plt.ylabel('Q-Net Loss')
#     plt.title('Training Loss_' + data)
#
#     plt.tight_layout()
#     # plt.savefig(f'./New_pic/RL_{serial_number}_Training_Reward&Loss_({data})_(ratio_{CandidateNodes_Ratio}).png')
#     plt.show()
#
#
#
# if __name__ == '__main__':
#     freeze_support()
#     main()
