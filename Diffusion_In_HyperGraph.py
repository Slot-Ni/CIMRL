import numpy as np
from joblib import Parallel, delayed
from multiprocessing.shared_memory import SharedMemory
from tqdm import tqdm
import os

SIM_N_JOBS = max(1, int(os.environ.get("CIM_SIM_N_JOBS", "4")))
SIM_BACKEND = os.environ.get("CIM_SIM_BACKEND", "threading")
_PREPROCESS_CACHE = {}
_CURRENT_GRAPH_CACHE_KEY = None


def _get_preprocessed_graph(G_matrix):
    cache_key = _CURRENT_GRAPH_CACHE_KEY or (id(G_matrix), G_matrix.shape, G_matrix.dtype.str)
    cached = _PREPROCESS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    num_nodes, num_hyperedges = G_matrix.shape
    hyperedge_nodes = [np.where(G_matrix[:, he] == 1)[0] for he in range(num_hyperedges)]
    node_hyperedges = [np.where(G_matrix[node_idx] == 1)[0] for node_idx in range(num_nodes)]
    cached = (num_nodes, num_hyperedges, hyperedge_nodes, node_hyperedges)
    _PREPROCESS_CACHE[cache_key] = cached
    return cached


def HG_Diffusion_SICP(G_matrix, NodeList, SeedNodes, BlockNodes, beta, T, seed=None):
    """Synchronous irreversible ACHPD; competitor wins simultaneous attempts."""
    target = list(BlockNodes) if False else [NodeList.index(v) for v in BlockNodes]
    rival = [NodeList.index(v) for v in SeedNodes]
    return _paper_diffusion(G_matrix, rival, target, beta, T, seed)




def HG_Diffusion_SICP_pos(G_matrix, NodeList, SeedNodes, BlockNodes_pos, beta, T, seed=None):
    """Synchronous irreversible ACHPD; competitor wins simultaneous attempts."""
    target = list(BlockNodes_pos) if True else [NodeList.index(v) for v in BlockNodes_pos]
    rival = [NodeList.index(v) for v in SeedNodes]
    return _paper_diffusion(G_matrix, rival, target, beta, T, seed)



def _run_simulation(diffusion_func, task_args):
    global _CURRENT_GRAPH_CACHE_KEY
    """全局辅助函数，处理共享内存并调用扩散函数（带强力防崩溃保护）"""
    if len(task_args) == 9:
        shm_name, dtype, shape, NodeList, SeedNodes, BlockNodes, beta, T, seed = task_args
    else:
        shm_name, dtype, shape, NodeList, SeedNodes, BlockNodes, beta, T = task_args
        seed = None

    shm = None
    G_matrix = None  # 提前声明，防止 finally 报错
    try:
        shm = SharedMemory(name=shm_name)
        G_matrix = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        _CURRENT_GRAPH_CACHE_KEY = (shm_name, shape, np.dtype(dtype).str)
        res = diffusion_func(G_matrix, NodeList, SeedNodes, BlockNodes, beta, T, seed=seed)
        return res
    finally:
        _CURRENT_GRAPH_CACHE_KEY = None
        if G_matrix is not None:
            del G_matrix  # 显式删除 numpy 数组对象
        if shm is not None:
            shm.close()  # 关闭共享内存连接

        # 强制触发 Python 垃圾回收器，清理底层 C 语言层面的 mmap 句柄
        import gc
        gc.collect()


def Simulations(simulation_num, diffusion_func, shm_name, dtype, shape, *args, seed_base=42):
    """并行执行多次扩散模拟"""
    base_seed = seed_base
    seeds = [base_seed + i for i in range(simulation_num)]
    tasks = [(shm_name, dtype, shape) + args + (seed,) for seed in seeds]

    # 【关键修复3】：绝对不能用 n_jobs=-1！将其固定为 4。
    # 如果您的电脑依然报 WinError 1455，请将这里的 4 改为 2。
    try:
        results = Parallel(n_jobs=SIM_N_JOBS, backend=SIM_BACKEND)(
            delayed(_run_simulation)(diffusion_func, task)
            for task in tasks
        )
    except OSError as exc:
        if getattr(exc, "winerror", None) != 1455:
            raise
        print("⚠️  WinError 1455: 页面文件不足，自动切换为单线程模拟以继续运行。")
        results = [_run_simulation(diffusion_func, task) for task in tasks]

    if results and isinstance(results[0], tuple):
        blocked_counts = [result[3] for result in results]
    else:
        blocked_counts = results

    return np.mean(blocked_counts)



def Simulations_tqdm(simulation_num, diffusion_func, shm_name, dtype, shape, *args):
    """
    并行执行多次扩散模拟（使用 joblib + 共享内存优化）
    """
    # 生成任务列表
    tasks = [(shm_name, dtype, shape) + args for _ in range(simulation_num)]

    # 使用 joblib 并行执行
    results = Parallel(n_jobs=-1)(
        delayed(_run_simulation)(diffusion_func, task)
        for task in tqdm(tasks, desc="模拟进度", unit="次")
    )

    # 计算平均阻塞节点数
    blocked_counts = [result[3] for result in results]
    return np.mean(blocked_counts)



def HG_Diffusion_SICP_track(G_matrix, NodeList, SeedNodes, BlockNodes, beta, T, interval):
    """
    该函数实现了HG_Diffusion_SICP_pos优化的扩散过程，并跟踪每个间隔时间的BlockNodes数量。

    参数:
    G_matrix (numpy.ndarray): 超图的邻接矩阵，形状为(num_nodes, num_hyperedges)
    NodeList (list): 节点名称列表
    SeedNodes (list): 初始谣言节点名称列表
    BlockNodes_pos (list): 初始阻塞节点的索引列表
    beta (float): 激活概率
    T (int): 扩散的时间步数
    interval (int): 记录BlockNodes数量的时间间隔

    返回:
    tuple: 包含节点总数、未激活节点数、谣言节点数、阻塞节点数和各时间间隔BlockNodes数量列表的元组
    """
    num_nodes, num_hyperedges = G_matrix.shape
    hyperedge_nodes = [np.where(G_matrix[:, he] == 1)[0] for he in range(num_hyperedges)]
    node_hyperedges = [np.where(G_matrix[node_idx] == 1)[0] for node_idx in range(num_nodes)]

    hyperedge_choices = np.full((num_nodes, T), -1, dtype=int)
    for idx in range(num_nodes):
        hes = node_hyperedges[idx]
        if len(hes) > 0:
            hyperedge_choices[idx] = np.random.choice(hes, size=T)


    # 转换节点名为索引
    # 将阻塞节点名称转换为对应的索引
    BlockNodes_index = [NodeList.index(node) for node in BlockNodes]
    # 将谣言节点名称转换为对应的索引
    RumorNodes_index = [NodeList.index(node) for node in SeedNodes]

    StateList = np.zeros(num_nodes, dtype=np.int8)
    StateList[BlockNodes_index] = 1
    StateList[RumorNodes_index] = -1

    BlockNodes_set = set(BlockNodes_index)
    RumorNodes_set = set(RumorNodes_index)

    track_counts = [len(BlockNodes_set)]  # 初始状态

    for t in range(T):
        # 处理阻塞节点传播
        if BlockNodes_set:
            active_block = np.array(list(BlockNodes_set))
            current_hyper = hyperedge_choices[active_block, t]
            valid_mask = current_hyper != -1
            valid_actives = active_block[valid_mask]
            valid_hes = current_hyper[valid_mask]

            if valid_actives.size > 0:
                all_neighbors = np.concatenate([hyperedge_nodes[he] for he in valid_hes])
                unactivated = all_neighbors[StateList[all_neighbors] == 0]
                if unactivated.size > 0:
                    activated = unactivated[np.random.random(unactivated.size) <= beta]
                    StateList[activated] = 1
                    BlockNodes_set.update(activated.tolist())

        # 处理谣言节点传播
        if RumorNodes_set:
            active_rumor = np.array(list(RumorNodes_set))
            current_hyper = hyperedge_choices[active_rumor, t]
            valid_mask = current_hyper != -1
            valid_actives = active_rumor[valid_mask]
            valid_hes = current_hyper[valid_mask]

            if valid_actives.size > 0:
                all_neighbors = np.concatenate([hyperedge_nodes[he] for he in valid_hes])
                unactivated = all_neighbors[StateList[all_neighbors] == 0]
                if unactivated.size > 0:
                    activated = unactivated[np.random.random(unactivated.size) <= beta]
                    StateList[activated] = -1
                    RumorNodes_set.update(activated.tolist())

        # 记录间隔时间点的BlockNodes数量
        current_time = t + 1
        if current_time % interval == 0 or current_time == T:
            track_counts.append(len(BlockNodes_set))

    unactivated_number = np.sum(StateList == 0)
    return len(NodeList), unactivated_number, len(RumorNodes_set), track_counts



def HG_Diffusion_SICP_pos_track(G_matrix, NodeList, SeedNodes, BlockNodes_pos, beta, T, interval):
    """
    该函数实现了HG_Diffusion_SICP_pos优化的扩散过程，并跟踪每个间隔时间的BlockNodes数量。

    参数:
    G_matrix (numpy.ndarray): 超图的邻接矩阵，形状为(num_nodes, num_hyperedges)
    NodeList (list): 节点名称列表
    SeedNodes (list): 初始谣言节点名称列表
    BlockNodes_pos (list): 初始阻塞节点的索引列表
    beta (float): 激活概率
    T (int): 扩散的时间步数
    interval (int): 记录BlockNodes数量的时间间隔

    返回:
    tuple: 包含节点总数、未激活节点数、谣言节点数、阻塞节点数和各时间间隔BlockNodes数量列表的元组
    """
    num_nodes, num_hyperedges = G_matrix.shape
    hyperedge_nodes = [np.where(G_matrix[:, he] == 1)[0] for he in range(num_hyperedges)]
    node_hyperedges = [np.where(G_matrix[node_idx] == 1)[0] for node_idx in range(num_nodes)]

    hyperedge_choices = np.full((num_nodes, T), -1, dtype=int)
    for idx in range(num_nodes):
        hes = node_hyperedges[idx]
        if len(hes) > 0:
            hyperedge_choices[idx] = np.random.choice(hes, size=T)

    BlockNodes_index = BlockNodes_pos
    RumorNodes_index = [NodeList.index(node) for node in SeedNodes]

    StateList = np.zeros(num_nodes, dtype=np.int8)
    StateList[BlockNodes_index] = 1
    StateList[RumorNodes_index] = -1

    BlockNodes_set = set(BlockNodes_index)
    RumorNodes_set = set(RumorNodes_index)

    track_counts = [len(BlockNodes_set)]  # 初始状态

    for t in range(T):
        # 处理阻塞节点传播
        if BlockNodes_set:
            active_block = np.array(list(BlockNodes_set))
            current_hyper = hyperedge_choices[active_block, t]
            valid_mask = current_hyper != -1
            valid_actives = active_block[valid_mask]
            valid_hes = current_hyper[valid_mask]

            if valid_actives.size > 0:
                all_neighbors = np.concatenate([hyperedge_nodes[he] for he in valid_hes])
                unactivated = all_neighbors[StateList[all_neighbors] == 0]
                if unactivated.size > 0:
                    activated = unactivated[np.random.random(unactivated.size) <= beta]
                    StateList[activated] = 1
                    BlockNodes_set.update(activated.tolist())

        # 处理谣言节点传播
        if RumorNodes_set:
            active_rumor = np.array(list(RumorNodes_set))
            current_hyper = hyperedge_choices[active_rumor, t]
            valid_mask = current_hyper != -1
            valid_actives = active_rumor[valid_mask]
            valid_hes = current_hyper[valid_mask]

            if valid_actives.size > 0:
                all_neighbors = np.concatenate([hyperedge_nodes[he] for he in valid_hes])
                unactivated = all_neighbors[StateList[all_neighbors] == 0]
                if unactivated.size > 0:
                    activated = unactivated[np.random.random(unactivated.size) <= beta]
                    StateList[activated] = -1
                    RumorNodes_set.update(activated.tolist())

        # 记录间隔时间点的BlockNodes数量
        current_time = t + 1
        if current_time % interval == 0 or current_time == T:
            track_counts.append(len(BlockNodes_set))

    unactivated_number = np.sum(StateList == 0)
    return len(NodeList), unactivated_number, len(RumorNodes_set), track_counts



def _run_simulation_track(diffusion_func, task_args):
    """处理共享内存并调用跟踪型扩散函数的辅助函数"""
    shm_name, dtype, shape, NodeList, SeedNodes, BlockNodes, beta, T, interval = task_args
    try:
        shm = SharedMemory(name=shm_name)
        G_matrix = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        return diffusion_func(G_matrix, NodeList, SeedNodes, BlockNodes, beta, T, interval)
    finally:
        shm.close()


def Simulations_track(simulation_num, diffusion_func, shm_name, dtype, shape, *args):
    """
    并行执行多次跟踪型扩散模拟，返回各时间间隔的平均阻塞节点数量

    参数:
    simulation_num (int): 模拟次数
    diffusion_func (callable): 跟踪型扩散函数（如HG_Diffusion_SICP_pos_track）
    shm_name (str): 共享内存名称
    dtype: 矩阵数据类型
    shape: 矩阵形状
    *args: 扩散函数参数 (NodeList, SeedNodes, BlockNodes, beta, T, interval)

    返回:
    tuple: (平均最终阻塞数, 各时间点平均阻塞数列表)
    """
    tasks = [(shm_name, dtype, shape) + args for _ in range(simulation_num)]

    results = Parallel(n_jobs=-1)(
        delayed(_run_simulation_track)(diffusion_func, task)
        for task in tasks
    )

    # 提取各时间点数据
    time_series = [result[3] for result in results]

    # 计算各时间点平均值
    avg_series = np.mean(time_series, axis=0).tolist()

    return avg_series



def Simulations_track_tqdm(simulation_num, diffusion_func, shm_name, dtype, shape, *args):
    """
    并行执行多次跟踪型扩散模拟，返回各时间间隔的平均阻塞节点数量

    参数:
    simulation_num (int): 模拟次数
    diffusion_func (callable): 跟踪型扩散函数（如HG_Diffusion_SICP_pos_track）
    shm_name (str): 共享内存名称
    dtype: 矩阵数据类型
    shape: 矩阵形状
    *args: 扩散函数参数 (NodeList, SeedNodes, BlockNodes, beta, T, interval)

    返回:
    tuple: (平均最终阻塞数, 各时间点平均阻塞数列表)
    """
    tasks = [(shm_name, dtype, shape) + args for _ in range(simulation_num)]

    results = Parallel(n_jobs=-1)(
        delayed(_run_simulation_track)(diffusion_func, task)
        for task in tqdm(tasks, desc="跟踪模拟进度", unit="次")
    )

    # 提取各时间点数据
    time_series = [result[3] for result in results]

    # 计算各时间点平均值
    avg_series = np.mean(time_series, axis=0).tolist()

    return avg_series



def _paper_diffusion(matrix, rivals, targets, beta, horizon, seed=None):
    if not 0 <= beta <= 1 or horizon < 0:
        raise ValueError('Invalid propagation parameters')
    if set(rivals) & set(targets):
        raise ValueError('Competing and target seeds must be disjoint')
    rng = np.random.default_rng(seed)
    n = len(matrix)
    incident = [np.flatnonzero(row) for row in matrix]
    members = [np.flatnonzero(col) for col in matrix.T]
    state = np.zeros(n, dtype=np.int8)
    state[rivals] = -1
    state[targets] = 1
    for _ in range(horizon):
        susceptible = state == 0
        rival_hits = np.zeros(n, dtype=bool)
        target_hits = np.zeros(n, dtype=bool)
        for source in np.flatnonzero(state):
            if not len(incident[source]):
                continue
            edge = rng.choice(incident[source])
            available = members[edge][susceptible[members[edge]]]
            hit = available[rng.random(len(available)) < beta]
            (rival_hits if state[source] == -1 else target_hits)[hit] = True
        # Both masks were computed from the same beginning-of-round state.
        state[susceptible & target_hits & ~rival_hits] = 1
        state[susceptible & rival_hits] = -1
    return n, int(np.sum(state == 0)), int(np.sum(state == -1)), int(np.sum(state == 1))
