import numpy as np
import os
import json
import math
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import random
import matplotlib.pyplot as plt
from multiprocessing.shared_memory import SharedMemory

# 瀵煎叆鏍稿績妯″潡
from Indicator import Competition_composite_indicator_new
from Diffusion_In_HyperGraph import Simulations, HG_Diffusion_SICP, HG_Diffusion_SICP_pos
from SeedNodesGenerator import H_Degree_SG
from Baseline_Algorithm import Indicator_selection_Greedy, HADP, H_Degree, CI
from Baseline_S2V import Compete_S2V_DQN, S2VReplayBuffer

# ==========================================
# 1. 瀹為獙鍏ㄥ眬閰嶇疆
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = Path(r"D:\YX\HyperGraphMatrix")
RESULT_DIR = SCRIPT_DIR / "HCL_Final_Comparison"
EVAL_CACHE_DIR = RESULT_DIR / "eval_cache"
SEED_CACHE_DIR = RESULT_DIR / "seed_cache"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(EVAL_CACHE_DIR, exist_ok=True)
os.makedirs(SEED_CACHE_DIR, exist_ok=True)

HCL_DATASETS = [
    "synthetic_HCL_v2000_e2000_exp2.0_maxdeg10",
    "synthetic_HCL_v2000_e2000_exp2.5_maxdeg10",
    "synthetic_HCL_v2000_e2000_exp3.0_maxdeg10"
]

BETA2, BETA1, T_MAX, SIM_TIMES, SEED = 0.03, 0.05, 30, 1000, 42
S2V_TRAIN_EPISODES = 30
S2V_TRAIN_SIM_TIMES = 20
S2V_BATCH_SIZE = 64
S2V_BUFFER_CAPACITY = 30000
S2V_TARGET_UPDATE_FREQ = 100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

global_results_collector = {}
PLOT_ORDER = ['CSI-Greedy', 'HDegree', 'HADP', 'CI', 'S2V-DQN', 'CSIQ']
DISPLAY_NAME = {
    'CSI-Greedy': 'HCDP-Greedy',
    'HDegree': 'HDgree',
    'HADP': 'HADP',
    'CI': 'CI',
    'S2V-DQN': 'S2V-DQN',
    'CSIQ': 'HCDP-DQN',
}

try:
    import numpy.core as _numpy_core
    sys.modules.setdefault("numpy._core", _numpy_core)
    sys.modules.setdefault("numpy._core.multiarray", _numpy_core.multiarray)
    sys.modules.setdefault("numpy._core._multiarray_umath", _numpy_core._multiarray_umath)
except Exception:
    pass


# ==========================================
# 宸ュ叿鍑芥暟
# ==========================================
def hypergraph_to_graph(hypergraph_matrix):
    OR_matrix = (hypergraph_matrix @ hypergraph_matrix.T) > 0
    np.fill_diagonal(OR_matrix, False)
    return OR_matrix.astype(np.float32)


def resolve_csiq_path(data_name):
    filename = (
        f"indicator_selection_RL_11_({data_name})_(T_{T_MAX})_"
        f"(beta_{BETA2})_(ratio_0.2).npy"
    )
    search_dirs = [
        SCRIPT_DIR / "HCL_synthetic_comparison" / "New_list",
        SCRIPT_DIR / "HCL_synthetic_comparison" / "New_List",
        SCRIPT_DIR / "New_list",
        SCRIPT_DIR / "New_List",
        Path.cwd() / "HCL_synthetic_comparison" / "New_list",
        Path.cwd() / "HCL_synthetic_comparison" / "New_List",
        Path.cwd() / "New_list",
        Path.cwd() / "New_List",
    ]
    for directory in search_dirs:
        path = directory / filename
        if path.exists():
            return path
    return search_dirs[0] / filename


def eval_cache_path(data_name, method_name):
    safe_method = method_name.replace("/", "_")
    return EVAL_CACHE_DIR / (
        f"{data_name}_{safe_method}_sim{SIM_TIMES}_T{T_MAX}_beta{BETA2}.npy"
    )


def seed_cache_path(data_name, method_name):
    safe_method = method_name.replace("/", "_")
    return SEED_CACHE_DIR / f"{data_name}_{safe_method}_budget{int(0.01 * int(data_name.split('_v')[1].split('_')[0]))}_T{T_MAX}_beta{BETA2}.npy"


def load_seed_cache(data_name, method_name):
    path = seed_cache_path(data_name, method_name)
    if path.exists():
        print(f"  - 澶嶇敤 {method_name} 绉嶅瓙缂撳瓨: {path}")
        try:
            return np.load(path, allow_pickle=True).tolist()
        except ModuleNotFoundError as error:
            print(f"    缂撳瓨涓嶅吋瀹硅繖涓寖鍥达細{error}，灏嗛噸鏂拌绠?")
            return None
    return None


def save_seed_cache(data_name, method_name, seeds):
    path = seed_cache_path(data_name, method_name)
    np.save(path, np.array(seeds, dtype=object))
    return seeds


def run_eval(G_matrix, NodeList, seed_nodes, target_seeds_list, data_name=None, method_name=None):
    cache_path = eval_cache_path(data_name, method_name) if data_name and method_name else None
    if cache_path is not None and cache_path.exists():
        try:
            return np.load(cache_path, allow_pickle=True).tolist()
        except ModuleNotFoundError as error:
            print(f"    璇勪及缂撳瓨涓嶅吋瀹硅繖涓寖鍥达細{error}，灏嗛噸鏂拌绠?")

    shm = SharedMemory(create=True, size=G_matrix.nbytes)
    try:
        shm_np = np.ndarray(G_matrix.shape, dtype=G_matrix.dtype, buffer=shm.buf)
        np.copyto(shm_np, G_matrix)
        results = []
        for i in range(1, len(target_seeds_list) + 1):
            subset = target_seeds_list[:i]
            inf = Simulations(SIM_TIMES, HG_Diffusion_SICP, shm.name, G_matrix.dtype, G_matrix.shape,
                              NodeList, seed_nodes, subset, BETA2, T_MAX)
            results.append(inf)
    finally:
        shm.close()
        shm.unlink()

    if cache_path is not None:
        np.save(cache_path, np.array(results, dtype=np.float64))
    return results


def train_light_s2v(G_matrix, NodeList, seed_nodes, seed_nodes_pos, graph_adj, budget, device):
    candidate_pos = [node for node in range(len(NodeList)) if node not in seed_nodes_pos]
    agent = Compete_S2V_DQN(
        epsilon=0.5,
        alpha=0.01,
        gamma=0.8,
        adj_matrix=graph_adj,
        edge_weights=graph_adj,
        nodes_number=len(NodeList),
        candidate_pos=candidate_pos,
        device=device,
        embedding_dim=64
    )
    agent.s2v = agent.s2v.to(device)

    replay_buffer = S2VReplayBuffer(S2V_BUFFER_CAPACITY)
    epsilon_end = 0.01
    epsilon_decay = (agent.epsilon - epsilon_end) / S2V_TRAIN_EPISODES
    total_steps = 0

    shm = SharedMemory(create=True, size=G_matrix.nbytes)
    try:
        shm_np = np.ndarray(G_matrix.shape, dtype=G_matrix.dtype, buffer=shm.buf)
        np.copyto(shm_np, G_matrix)

        for episode in range(S2V_TRAIN_EPISODES):
            agent.epsilon = max(epsilon_end, 0.5 - episode * epsilon_decay)
            candidate_pool = set(candidate_pos.copy())
            state = []
            last_influence = 0
            done = False

            while not done and candidate_pool:
                action = agent.take_action(state, list(candidate_pool))
                candidate_pool.discard(action)

                influence = Simulations(
                    S2V_TRAIN_SIM_TIMES,
                    HG_Diffusion_SICP_pos,
                    shm.name,
                    G_matrix.dtype,
                    G_matrix.shape,
                    NodeList,
                    seed_nodes,
                    state + [action],
                    BETA2,
                    T_MAX
                )
                reward = influence - last_influence
                last_influence = influence

                next_state = state + [action]
                done = len(next_state) >= budget
                replay_buffer.add(state, action, reward, next_state, done)

                if len(replay_buffer) >= S2V_BATCH_SIZE:
                    agent.update(replay_buffer.sample(S2V_BATCH_SIZE))

                if total_steps % S2V_TARGET_UPDATE_FREQ == 0:
                    agent.sync_target_net()
                total_steps += 1
                state = next_state
    finally:
        shm.close()
        shm.unlink()

    final_policy = []
    candidate_pool = candidate_pos.copy()
    while len(final_policy) < budget and candidate_pool:
        best_node = agent.best_action(final_policy, candidate_pool)
        final_policy.append(best_node)
        candidate_pool.remove(best_node)

    return [NodeList[index] for index in final_policy]


def get_style(name):
    styles = {
        'CSI-Greedy': {'color': '#ff7f0e', 'marker': 's', 'z': 5},
        'HDegree': {'color': '#EC6E66', 'marker': 'd', 'z': 2},
        'HADP': {'color': '#FAC7B3', 'marker': '^', 'z': 2},
        'CI': {'color': '#91CCC0', 'marker': '*', 'z': 2},
        'S2V-DQN': {'color': '#2ca02c', 'marker': 'v', 'z': 5},
        'CSIQ': {'color': '#1f77b4', 'marker': '8', 'z': 10}
    }
    return styles.get(name, {'color': 'gray', 'marker': 'o', 'z': 1})


# ==========================================
# 涓绘祦绋?
# ==========================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for idx, data_name in enumerate(HCL_DATASETS):
        print(f"\n馃殌 璇勪及澶勭悊涓?({idx + 1}/{len(HCL_DATASETS)}): {data_name}")
        m_p = BASE_PATH / f"{data_name}.npy"
        n_p = BASE_PATH / f"{data_name}_nodelist.npy"
        if not m_p.exists():
            print(f"  鉂?鎵句笉鍒版暟鎹泦鏂囦欢锛岃烦杩? {data_name}")
            continue

        G_matrix = np.load(m_p, allow_pickle=True)
        NodeList = np.load(n_p, allow_pickle=True).tolist()
        node_to_pos = {node: pos for pos, node in enumerate(NodeList)}
        num_v = int(0.01 * len(NodeList))

        Vs_names = H_Degree_SG(G_matrix, NodeList, num_v)
        Vs_pos = [node_to_pos[node] for node in Vs_names]

        algo_seeds = {}

        print("  - 姝ｅ湪璁＄畻 CSI 鎸囨爣鍙婅繍琛?CSI-Greedy...")
        cached = load_seed_cache(data_name, 'CSI-Greedy')
        if cached is None:
            cand_pos_list = Competition_composite_indicator_new(Vs_pos, G_matrix, data_name,
                                                                int(0.2 * len(NodeList)), 10, 0.5).tolist()
            algo_seeds['CSI-Greedy'] = save_seed_cache(
                data_name,
                'CSI-Greedy',
                Indicator_selection_Greedy(
                    G_matrix, NodeList, Vs_names, cand_pos_list, num_v, 100, BETA2, T_MAX,
                    checkpoint_path=seed_cache_path(data_name, 'CSI-Greedy')
                )
            )
        else:
            algo_seeds['CSI-Greedy'] = cached

        print("  - 姝ｅ湪杩愯鍚彂寮忓熀绾?(HDegree, HADP, CI)...")
        for method_name, method_func in [('HDegree', H_Degree), ('HADP', HADP), ('CI', CI)]:
            cached = load_seed_cache(data_name, method_name)
            algo_seeds[method_name] = cached if cached is not None else save_seed_cache(
                data_name, method_name, method_func(G_matrix, NodeList, Vs_names, num_v)
            )

        print("  - 姝ｅ湪杩愯杞婚噺璁粌鐗?S2V-DQN...")
        cached = load_seed_cache(data_name, 'S2V-DQN')
        if cached is None:
            OR_matrix = hypergraph_to_graph(G_matrix)
            algo_seeds['S2V-DQN'] = save_seed_cache(
                data_name,
                'S2V-DQN',
                train_light_s2v(G_matrix, NodeList, Vs_names, Vs_pos, OR_matrix, num_v, device)
            )
        else:
            algo_seeds['S2V-DQN'] = cached

        # -----------------------------------------------------
        # 鐩存帴璇诲彇棰勮绠楀ソ鐨?CSIQ (RL) 绉嶅瓙鏂囦欢
        # -----------------------------------------------------
        print("  - 姝ｅ湪鍔犺浇 CSIQ 棰勮绠楃粨鏋?..")
        csiq_file_path = resolve_csiq_path(data_name)

        if csiq_file_path.exists():
            algo_seeds['CSIQ'] = np.load(csiq_file_path, allow_pickle=True).tolist()
            print(f"    鉁?鎴愬姛鍔犺浇: {csiq_file_path}")
        else:
            print(f"    鉂?鏈壘鍒版枃浠? {csiq_file_path}")
            print("    CSIQ result file is missing; skip CSIQ evaluation for this dataset.")

        # -----------------------------------------------------
        # 璇勪及鍚勭畻娉曠殑褰卞搷鍔?
        # -----------------------------------------------------
        current_data_results = {}
        for name in PLOT_ORDER:
            if name in algo_seeds:
                print(f"  - 璇勪及 {name}...")
                current_data_results[name] = run_eval(
                    G_matrix, NodeList, Vs_names, algo_seeds[name], data_name, name
                )

        global_results_collector[data_name] = current_data_results

        # -----------------------------------------------------
        # 銆愭柊澧炪€戝皢璇ユ暟鎹泦鐨勭敾鍥炬暟鎹寔涔呭寲淇濆瓨涓?JSON
        # -----------------------------------------------------
        # 纭繚 numpy 鐨?float 绫诲瀷鑳借 json 搴忓垪鍖?
        json_friendly_results = {k: [float(val) for val in v] for k, v in current_data_results.items()}
        json_path = RESULT_DIR / f"{data_name}_plot_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_friendly_results, f, indent=4)
        print(f"  馃捑 璇ユ暟鎹泦鐨勭敾鍥炬暟鎹凡淇濆瓨鑷? {json_path}")

    # -----------------------------------------------------
    # 姹囨€诲ぇ鍥剧粯鍒?(鏀寔浠绘剰鏁伴噺鐨勬暟鎹泦鍔ㄦ€佹帓鐗?
    # -----------------------------------------------------
    if not global_results_collector:
        print("No evaluation data collected; skip plotting.")
        return

    print("\n馃搱 姝ｅ湪缁樺埗瀵规瘮鍥捐〃...")

    # 鍔ㄦ€佽绠楁墍闇€鐨勮鏁板拰鍒楁暟锛堟瘡琛屾渶澶氭斁 3 涓瓙鍥撅級
    num_plots = len(global_results_collector)
    cols = 3
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, (data_name, data_results) in enumerate(global_results_collector.items()):
        ax = axes[idx]

        # 鎻愬彇鏍囬涓尯鍒嗗害鐨勯儴鍒?(濡?v1000_exp3.0)
        parts = data_name.split('_')
        title_suffix = f"{parts[2]}_{parts[3]}" if len(parts) > 3 else data_name
        ax.set_title(f"HCL ({title_suffix})", fontsize=14, fontweight='bold')

        # 涓轰簡淇濆瓨鐙珛瀛愬浘锛屽崟鐙柊寤轰竴涓敾甯?
        fig_single, ax_single = plt.subplots(figsize=(6, 5))
        ax_single.set_title(f"HCL ({title_suffix})", fontsize=14, fontweight='bold')

        for algo_name in PLOT_ORDER:
            if algo_name in data_results:
                y_values = data_results[algo_name]
                x_values = range(1, len(y_values) + 1)
                style = get_style(algo_name)

                # 鐢诲湪澶ч暱鍥句笂
                ax.plot(x_values, y_values,
                        label=DISPLAY_NAME.get(algo_name, algo_name),
                        color=style['color'], marker=style['marker'],
                        linewidth=1.5,
                        markersize=9 if algo_name == 'CSIQ' else 7,
                        zorder=style['z'])

                # 鐢诲湪鐙珛瀛愬浘涓?
                ax_single.plot(x_values, y_values,
                               label=DISPLAY_NAME.get(algo_name, algo_name),
                               color=style['color'], marker=style['marker'],
                               linewidth=1.5,
                               markersize=9 if algo_name == 'CSIQ' else 7,
                               zorder=style['z'])

        # 澶у浘瀛愬浘淇グ
        ax.set_xlabel('Budget (b)', fontsize=12)
        ax.set_ylabel('Target Expected Influence', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        if idx == 0: ax.legend(loc='best', fontsize=11, framealpha=0.9)

        # 鐙珛瀛愬浘淇グ涓庝繚瀛?
        ax_single.set_xlabel('Budget (b)', fontsize=12)
        ax_single.set_ylabel('Target Expected Influence', fontsize=12)
        ax_single.grid(True, linestyle='--', alpha=0.6)
        ax_single.legend(loc='best', fontsize=11, framealpha=0.9)
        fig_single.tight_layout()
        single_save_path = RESULT_DIR / f"{data_name}_Individual_Plot.pdf"
        fig_single.savefig(single_save_path, format='pdf', bbox_inches='tight')
        plt.close(fig_single)  # 閲婃斁鍐呭瓨

    # 闅愯棌缃戞牸涓浣欑殑绌虹櫧瀛愬浘
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    combined_save_path = RESULT_DIR / 'HCL_All_Algorithms_Comparison.pdf'
    plt.savefig(combined_save_path, format='pdf', bbox_inches='tight')
    print(f"鉁?缁樺浘瀹屾垚锛佸ぇ缁勫悎鍥捐〃宸蹭繚瀛樿嚦: {combined_save_path}")
    print(f"鉁?鎵€鏈夌嫭绔嬪瓙鍥惧強鍘熷 JSON 鏁版嵁鍧囧凡淇濆瓨鍦?{RESULT_DIR} 鐩綍涓紒")

    plt.show()


# ==========================================
# 鎵ц鍏ュ彛
# ==========================================
if __name__ == "__main__":
    main()
