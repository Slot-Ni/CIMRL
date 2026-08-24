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
    """
    该函数实现了HG_Diffusion_SICP优化的扩散过程。

    参数:
    G_matrix (numpy.ndarray): 超图的邻接矩阵，形状为(num_nodes, num_hyperedges)
    NodeList (list): 节点名称列表
    SeedNodes (list): 初始谣言节点名称列表
    BlockNodes (list): 初始阻塞节点名称列表
    beta (float): 激活概率
    T (int): 扩散的时间步数

    返回:
    tuple: 包含节点总数、未激活节点数、谣言节点数和阻塞节点数的元组
    """

    # 添加seed参数
    if seed is not None:
        np.random.seed(seed)  # 固定随机种子
    # 预处理：预计算所有超边的节点列表和节点的超边列表
    # 获取节点数量和超边数量
    num_nodes, num_hyperedges, hyperedge_nodes, node_hyperedges = _get_preprocessed_graph(G_matrix)
    # 对于每个超边，找出与之相连的节点的索引
    # 对于每个节点，找出与之相连的超边的索引

    # 预生成所有节点的超边选择（形状：[num_nodes, T]）
    # 初始化超边选择数组，初始值为 -1
    hyperedge_choices = np.full((num_nodes, T), -1, dtype=int)
    # 遍历每个节点
    for idx in range(num_nodes):
        # 获取该节点相连的超边索引
        hes = node_hyperedges[idx]
        # 如果该节点有相连的超边
        if len(hes) > 0:
            rng = np.random.RandomState(seed) if seed is not None else np.random
            # 为该节点在每个时间步随机选择一个超边
            hyperedge_choices[idx] = np.random.choice(hes, size=T)

    # 转换节点名为索引
    # 将阻塞节点名称转换为对应的索引
    BlockNodes_index = [NodeList.index(node) for node in BlockNodes]
    # 将谣言节点名称转换为对应的索引
    RumorNodes_index = [NodeList.index(node) for node in SeedNodes]

    # 初始化状态（1: 阻塞, -1: 谣言, 0: 未激活）
    # 初始化节点状态数组，初始值为 0
    StateList = np.zeros(num_nodes, dtype=np.int8)
    # 将阻塞节点的状态设置为 1
    StateList[BlockNodes_index] = 1
    # 将谣言节点的状态设置为 -1
    StateList[RumorNodes_index] = -1

    # 使用集合存储激活节点
    # 将阻塞节点索引转换为集合
    BlockNodes_set = set(BlockNodes_index)
    # 将谣言节点索引转换为集合
    RumorNodes_set = set(RumorNodes_index)

    # 进行 T 个时间步的扩散
    for t in range(T):
        # 处理阻塞节点传播
        # 如果有阻塞节点
        if BlockNodes_set:
            # 将阻塞节点集合转换为数组
            active_block = np.array(list(BlockNodes_set))
            # 获取当前时间步这些阻塞节点选择的超边
            current_hyper = hyperedge_choices[active_block, t]
            # 筛选出选择了有效超边的节点
            valid_mask = current_hyper != -1
            valid_actives = active_block[valid_mask]
            valid_hes = current_hyper[valid_mask]

            # 如果有有效激活的阻塞节点
            if valid_actives.size > 0:
                # 获取这些有效超边所连接的所有邻居节点
                all_neighbors = np.concatenate([hyperedge_nodes[he] for he in valid_hes])
                # 筛选出未激活的邻居节点
                unactivated = all_neighbors[StateList[all_neighbors] == 0]
                # 如果有未激活的邻居节点
                if unactivated.size > 0:
                    # 根据激活概率随机选择要激活的节点
                    activated = unactivated[np.random.random(unactivated.size) <= beta]
                    # 将这些激活的节点状态设置为 1
                    StateList[activated] = 1
                    # 将这些激活的节点添加到阻塞节点集合中
                    BlockNodes_set.update(activated.tolist())

        # 处理谣言节点传播
        # 如果有谣言节点
        if RumorNodes_set:
            # 将谣言节点集合转换为数组
            active_rumor = np.array(list(RumorNodes_set))
            # 获取当前时间步这些谣言节点选择的超边
            current_hyper = hyperedge_choices[active_rumor, t]
            # 筛选出选择了有效超边的节点
            valid_mask = current_hyper != -1
            valid_actives = active_rumor[valid_mask]
            valid_hes = current_hyper[valid_mask]

            # 如果有有效激活的谣言节点
            if valid_actives.size > 0:
                # 获取这些有效超边所连接的所有邻居节点
                all_neighbors = np.concatenate([hyperedge_nodes[he] for he in valid_hes])
                # 筛选出未激活的邻居节点
                unactivated = all_neighbors[StateList[all_neighbors] == 0]
                # 如果有未激活的邻居节点
                if unactivated.size > 0:
                    # 根据激活概率随机选择要激活的节点
                    activated = unactivated[np.random.random(unactivated.size) <= beta]
                    # 将这些激活的节点状态设置为 -1
                    StateList[activated] = -1
                    # 将这些激活的节点添加到谣言节点集合中
                    RumorNodes_set.update(activated.tolist())

    # 统计结果
    # 统计未激活节点的数量
    unactivated_number = np.sum(StateList == 0)
    # 返回节点总数、未激活节点数、谣言节点数和阻塞节点数
    return len(NodeList), unactivated_number, len(RumorNodes_set), len(BlockNodes_set)



def HG_Diffusion_SICP_pos(G_matrix, NodeList, SeedNodes, BlockNodes_pos, beta, T, seed=None):
    """
    该函数实现了HG_Diffusion_SICP_pos优化的扩散过程。

    参数:
    G_matrix (numpy.ndarray): 超图的邻接矩阵，形状为(num_nodes, num_hyperedges)
    NodeList (list): 节点名称列表
    SeedNodes (list): 初始谣言节点名称列表
    BlockNodes_pos (list): 初始阻塞节点的索引列表
    beta (float): 激活概率
    T (int): 扩散的时间步数

    返回:
    tuple: 包含节点总数、未激活节点数、谣言节点数和阻塞节点数的元组
    """

    # 添加seed参数
    if seed is not None:
        np.random.seed(seed)  # 固定随机种子
    # 预处理与上述函数相同，仅BlockNodes_index处理不同
    # 获取节点数量和超边数量
    num_nodes, num_hyperedges, hyperedge_nodes, node_hyperedges = _get_preprocessed_graph(G_matrix)
    # 对于每个超边，找出与之相连的节点的索引
    # 对于每个节点，找出与之相连的超边的索引

    # 预生成所有节点的超边选择（形状：[num_nodes, T]）
    # 初始化超边选择数组，初始值为 -1
    hyperedge_choices = np.full((num_nodes, T), -1, dtype=int)
    # 遍历每个节点
    for idx in range(num_nodes):
        # 获取该节点相连的超边索引
        hes = node_hyperedges[idx]
        # 如果该节点有相连的超边
        if len(hes) > 0:
            # 使用固定种子生成随机选择
            rng = np.random.RandomState(seed) if seed is not None else np.random
            # 为该节点在每个时间步随机选择一个超边
            hyperedge_choices[idx] = np.random.choice(hes, size=T)

    # 这里直接使用传入的阻塞节点索引
    BlockNodes_index = BlockNodes_pos
    # 将谣言节点名称转换为对应的索引
    RumorNodes_index = [NodeList.index(node) for node in SeedNodes]

    # 初始化状态（1: 阻塞, -1: 谣言, 0: 未激活）
    # 初始化节点状态数组，初始值为 0
    StateList = np.zeros(num_nodes, dtype=np.int8)
    # 将阻塞节点的状态设置为 1
    StateList[BlockNodes_index] = 1
    # 将谣言节点的状态设置为 -1
    StateList[RumorNodes_index] = -1

    # 使用集合存储激活节点
    # 将阻塞节点索引转换为集合
    BlockNodes_set = set(BlockNodes_index)
    # 将谣言节点索引转换为集合
    RumorNodes_set = set(RumorNodes_index)

    # 进行 T 个时间步的扩散
    for t in range(T):
        # 处理阻塞节点传播
        # 如果有阻塞节点
        if BlockNodes_set:
            # 将阻塞节点集合转换为数组
            active_block = np.array(list(BlockNodes_set))
            # 获取当前时间步这些阻塞节点选择的超边
            current_hyper = hyperedge_choices[active_block, t]
            # 筛选出选择了有效超边的节点
            valid_mask = current_hyper != -1
            valid_actives = active_block[valid_mask]
            valid_hes = current_hyper[valid_mask]

            # 如果有有效激活的阻塞节点
            if valid_actives.size > 0:
                # 获取这些有效超边所连接的所有邻居节点
                all_neighbors = np.concatenate([hyperedge_nodes[he] for he in valid_hes])
                # 筛选出未激活的邻居节点
                unactivated = all_neighbors[StateList[all_neighbors] == 0]
                # 如果有未激活的邻居节点
                if unactivated.size > 0:
                    # 根据激活概率随机选择要激活的节点
                    activated = unactivated[np.random.random(unactivated.size) <= beta]
                    # 将这些激活的节点状态设置为 1
                    StateList[activated] = 1
                    # 将这些激活的节点添加到阻塞节点集合中
                    BlockNodes_set.update(activated.tolist())

        # 处理谣言节点传播
        # 如果有谣言节点
        if RumorNodes_set:
            # 将谣言节点集合转换为数组
            active_rumor = np.array(list(RumorNodes_set))
            # 获取当前时间步这些谣言节点选择的超边
            current_hyper = hyperedge_choices[active_rumor, t]
            # 筛选出选择了有效超边的节点
            valid_mask = current_hyper != -1
            valid_actives = active_rumor[valid_mask]
            valid_hes = current_hyper[valid_mask]

            # 如果有有效激活的谣言节点
            if valid_actives.size > 0:
                # 获取这些有效超边所连接的所有邻居节点
                all_neighbors = np.concatenate([hyperedge_nodes[he] for he in valid_hes])
                # 筛选出未激活的邻居节点
                unactivated = all_neighbors[StateList[all_neighbors] == 0]
                # 如果有未激活的邻居节点
                if unactivated.size > 0:
                    # 根据激活概率随机选择要激活的节点
                    activated = unactivated[np.random.random(unactivated.size) <= beta]
                    # 将这些激活的节点状态设置为 -1
                    StateList[activated] = -1
                    # 将这些激活的节点添加到谣言节点集合中
                    RumorNodes_set.update(activated.tolist())

    # 统计结果
    # 统计未激活节点的数量
    unactivated_number = np.sum(StateList == 0)
    # 返回节点总数、未激活节点数、谣言节点数和阻塞节点数
    return len(NodeList), unactivated_number, len(RumorNodes_set), len(BlockNodes_set)


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


def Simulations(simulation_num, diffusion_func, shm_name, dtype, shape, *args):
    """并行执行多次扩散模拟"""
    base_seed = 42
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

