import numpy as np
import torch
import os
from tqdm import trange
from multiprocessing.shared_memory import SharedMemory

# 导入你的环境和底层模块
from CompeteAlgorithm import Compete_DQN, ReplayBuffer
from Indicator import Competition_composite_indicator_new, get_node_hyperedge_avgsize, get_avg_HigerOrder_dis
from SeedNodesGenerator import H_Degree_SG
from Diffusion_In_HyperGraph import HG_Diffusion_SICP_pos, Simulations

def get_CSIQ_seeds(G_matrix, NodeList, data_name, num_v, beta, T, device):
    """
    封装好的 CSIQ 算法。
    自动执行候选集筛选 -> DQN 训练 -> 保存并返回最优种子节点序列。
    """
    # 1. 初始化超参数
    Indicator_Order = 10
    Indicator_Alpha = 0.5
    CandidateNodes_Ratio = 0.2
    num_episodes = 600
    alpha = 0.03
    gamma = 0.9
    epsilon_start = 0.5
    epsilon_end = 0.01
    epsilon_decay = (epsilon_start - epsilon_end) / num_episodes
    simulation_num = 500
    batch_size = 64
    buffer_capacity = 30000
    state_dim = 2 * num_v
    target_update_freq = 100

    # 2. 生成初始竞争种子节点
    SeedNodes = H_Degree_SG(G_matrix, NodeList, num_v)
    SeedNodes_pos = [NodeList.index(node) for node in SeedNodes]

    # 3. CSI 筛选：生成候选节点池
    CandidateNodes_pos = Competition_composite_indicator_new(
        SeedNodes_pos, G_matrix, data_name,
        int(CandidateNodes_Ratio * len(NodeList)), Indicator_Order, Indicator_Alpha
    ).tolist()

    # 4. 初始化 DQN Agent 与回放池
    agent = Compete_DQN(
        epsilon=epsilon_start, alpha=alpha, gamma=gamma, state_dim=state_dim,
        action_dim=len(CandidateNodes_pos),
        node_hyperedge_avgsize=get_node_hyperedge_avgsize(G_matrix),
        avg_HigerOrder_dis_seednodes=get_avg_HigerOrder_dis(SeedNodes_pos, G_matrix, data_name, Indicator_Order, Indicator_Alpha),
        candidate_nodes_pos=CandidateNodes_pos, device=device
    )
    replay_buffer = ReplayBuffer(buffer_capacity)

    # 5. 开辟共享内存
    shm = SharedMemory(create=True, size=G_matrix.nbytes)
    shm_np = np.ndarray(G_matrix.shape, dtype=G_matrix.dtype, buffer=shm.buf)
    np.copyto(shm_np, G_matrix)

    # 6. DQN 训练循环
    total_steps = 0
    for episode in trange(num_episodes, desc=f"  -> 正在训练 {data_name} 的 CSIQ (DQN)"):
        current_epsilon = max(epsilon_end, epsilon_start - episode * epsilon_decay)
        agent.epsilon = current_epsilon
        CandidateNodesPool = set(CandidateNodes_pos.copy())
        state = []
        last_CompeteInf = 0
        done = False

        while not done:
            action_node, action_index = agent.take_action(state, list(CandidateNodesPool))
            CandidateNodesPool.discard(action_node)

            CompeteInf = Simulations(
                simulation_num, HG_Diffusion_SICP_pos, shm.name,
                G_matrix.dtype, G_matrix.shape, NodeList, SeedNodes,
                state + [action_node], beta, T
            )

            reward = CompeteInf - last_CompeteInf
            last_CompeteInf = CompeteInf
            next_state = state + [action_node]
            done = (len(next_state) == num_v)

            state_encoded = agent._encode_state(state)
            next_state_encoded = agent._encode_state(next_state)

            replay_buffer.add(state_encoded, action_index, reward, next_state_encoded, done)

            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                agent.update(zip(*batch))

            if total_steps % target_update_freq == 0:
                agent.sync_target_net()
            total_steps += 1
            state = next_state

    # 释放内存
    shm.close()
    shm.unlink()

    # 7. 策略推理 (获取最优动作序列)
    Indicator_selection_RL_pos = []
    current_state = []
    CandidateNodesPool = set(CandidateNodes_pos.copy())
    for _ in range(num_v):
        available_actions = list(CandidateNodesPool)
        if not available_actions: break
        action = agent.best_action(current_state, available_actions)
        CandidateNodesPool.discard(action)
        current_state.append(action)
        Indicator_selection_RL_pos.append(action)

    # 将节点序号转为具体名称
    Indicator_selection_RL = [NodeList[idx] for idx in Indicator_selection_RL_pos]

    # -----------------------------------------------------
    # 【新增】：保存结果到本地 ./New_List/ 文件夹
    # -----------------------------------------------------
    save_dir = 'HCL_synthetic_comparison/New_list'
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/indicator_selection_RL_11_({data_name})_(T_{T})_(beta_{beta})_(ratio_{CandidateNodes_Ratio}).npy'
    np.save(save_path, Indicator_selection_RL)
    print(f"\n    💾 训练完成！CSIQ 结果已成功保存至: {save_path}\n")

    return Indicator_selection_RL