import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import random
import time
from multiprocessing.shared_memory import SharedMemory
from tqdm import trange

# ==========================================
# 导入你的底层核心算法模块
# ==========================================
# 【已删除】：不需要导入 Data 模块，避开 Algebra.txt 报错
from SeedNodesGenerator import H_Degree_SG
from CompeteAlgorithm import Compete_DQN, ReplayBuffer
from Indicator import Competition_composite_indicator_new, get_avg_HigerOrder_dis, get_node_hyperedge_avgsize
from Diffusion_In_HyperGraph import HG_Diffusion_SICP_pos, Simulations

# 统一使用绝对路径，防止路径找不到
BASE_PATH = r"D:\YX\HyperGraphMatrix"


def train_and_save_CSIQ(data_name, beta=0.03, T=30):
    """核心训练函数：执行 DQN 训练并保存种子节点"""
    start = time.time()

    # 1. 基础超参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    CandidateNodes_Ratio = 0.2
    Indicator_Order = 10
    Indicator_Alpha = 0.5

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

    # 2. 数据加载 (直接读取 .npy 文件)
    m_p = os.path.join(BASE_PATH, f"{data_name}.npy")
    n_p = os.path.join(BASE_PATH, f"{data_name}_nodelist.npy")

    if not os.path.exists(m_p):
        print(f"❌ 找不到数据文件: {m_p}")
        return

    G_matrix = np.load(m_p, allow_pickle=True)
    NodeList = np.load(n_p, allow_pickle=True).tolist()

    num_v = int(0.01 * len(NodeList))
    state_dim = 2 * num_v

    # 3. 初始化种子节点与候选池
    SeedNodes = H_Degree_SG(G_matrix, NodeList, num_v)
    SeedNodes_pos = [NodeList.index(node) for node in SeedNodes]

    print(f"\n🚀 开始训练 CSIQ: {data_name} | 预算 b={num_v}")
    CandidateNodes_pos = Competition_composite_indicator_new(
        SeedNodes_pos, G_matrix, data_name,
        int(CandidateNodes_Ratio * len(NodeList)), Indicator_Order, Indicator_Alpha
    ).tolist()

    # 4. 初始化 RL Agent 和 回放池
    agent = Compete_DQN(
        epsilon=epsilon_start, alpha=alpha, gamma=gamma, state_dim=state_dim,
        action_dim=len(CandidateNodes_pos),
        node_hyperedge_avgsize=get_node_hyperedge_avgsize(G_matrix),
        avg_HigerOrder_dis_seednodes=get_avg_HigerOrder_dis(SeedNodes_pos, G_matrix, data_name, Indicator_Order,
                                                            Indicator_Alpha),
        candidate_nodes_pos=CandidateNodes_pos, device=device
    )
    replay_buffer = ReplayBuffer(buffer_capacity)

    # 5. 开辟共享内存
    shm = SharedMemory(create=True, size=G_matrix.nbytes)

    # 💥 加入强制保障机制：即使报错也绝对回收内存
    try:
        shm_np = np.ndarray(G_matrix.shape, dtype=G_matrix.dtype, buffer=shm.buf)
        np.copyto(shm_np, G_matrix)

        # 6. DQN 训练主循环
        total_steps = 0
        for episode in trange(num_episodes, desc=f"{data_name} 训练进度"):
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

        # 7. 策略推理：获取最优节点序列
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

        Indicator_selection_RL = [NodeList[idx] for idx in Indicator_selection_RL_pos]

        # 8. 保存结果 (统一保存格式)
        save_dir = 'HCL_synthetic_comparison/New_list'
        os.makedirs(save_dir, exist_ok=True)
        save_path = f'{save_dir}/indicator_selection_RL_11_({data_name})_(T_{T})_(beta_{beta})_(ratio_0.2).npy'
        np.save(save_path, Indicator_selection_RL)

        end = time.time()
        print(f"✅ 训练完成！耗时: {end - start:.2f}s | 结果已保存至: {save_path}\n")

    finally:
        # 🛡️ 终极防泄漏补丁：不管上面代码是成功跑完还是中途崩溃报错，一定会执行这里
        if 'shm' in locals():
            shm.close()
            shm.unlink()


if __name__ == "__main__":
    # 固定随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # 你需要跑的数据集
    datasets_to_train = [
        "synthetic_HCL_v1000_e1000_exp2.5_maxdeg10",
    ]

    # 依次执行训练并保存
    for data in datasets_to_train:
        # 注意：这里的 beta=0.03 要和画图脚本里的 BETA2 保持完全一致！
        train_and_save_CSIQ(data, beta=0.06, T=30)