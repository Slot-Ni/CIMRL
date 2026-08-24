import gc
import json
import os
import random
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from joblib.externals.loky import get_reusable_executor
from tqdm import trange

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from Baseline_S2V import Compete_S2V_DQN, S2VReplayBuffer
from Diffusion_In_HyperGraph import HG_Diffusion_SICP, HG_Diffusion_SICP_pos, Simulations
from SeedNodesGenerator import H_Degree_SG


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_NAME = "synthetic_HCL_v1000_e1000_exp2.5_maxdeg10"
BETAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
T_MAX = 30
SEED = 42

# Evaluation is kept identical to the previous comparison.
EVAL_SIM_TIMES = 1000

# Training follows Baseline_Main_S2V.py, but is isolated and cached.
TRAIN_EPISODES = 100
TRAIN_SIM_TIMES = 35
ALPHA = 0.01
GAMMA = 0.8
EPSILON_START = 0.5
EPSILON_END = 0.01
BATCH_SIZE = 64
BUFFER_CAPACITY = 30000
TARGET_UPDATE_FREQ = 100
EMBEDDING_DIM = 64

# Keep CPU by default. Baseline_S2V.py originally leaves the s2v module on CPU;
# using CPU avoids mixed-device errors and keeps runs reproducible.
DEVICE = "cpu"

BASE_PATH = PROJECT_ROOT / "HyperGraphMatrix"
BASE_RESULT_DIR = SCRIPT_DIR / "HCL1000_Beta_Method_Comparison"
BASE_JSON = BASE_RESULT_DIR / f"{DATA_NAME}_methods_beta_comparison.json"

# Separate output directory: this script never overwrites the existing figures.
RESULT_DIR = SCRIPT_DIR / "HCL1000_Beta_Method_Comparison_S2VOpt"
CACHE_DIR = RESULT_DIR / "cache"
EVAL_CACHE_DIR = RESULT_DIR / "eval_cache"
RESULT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
EVAL_CACHE_DIR.mkdir(exist_ok=True)

OPT_METHOD = "S2V-DQN-Opt"
METHOD_ORDER = ["CSI-Greedy", "HDegree", "HADP", "CI", "S2V-DQN", OPT_METHOD, "CSIQ"]
DISPLAY_NAME = {
    "CSIQ": "HCDP-DQN",
    "HDegree": "HDgree",
    "HADP": "HADP",
    "CI": "CI",
    "S2V-DQN": "S2V-DQN",
    OPT_METHOD: "S2V-DQN-Opt",
    "CSI-Greedy": "HCDP-Greedy",
}
STYLE = {
    "CSIQ": {"color": "#1f77b4", "marker": "o", "linewidth": 1.5, "z": 10},
    "HDegree": {"color": "#EC6E66", "marker": "D", "linewidth": 1.5, "z": 4},
    "HADP": {"color": "#FAC7B3", "marker": "^", "linewidth": 1.5, "z": 3},
    "CI": {"color": "#91CCC0", "marker": "*", "linewidth": 1.5, "z": 3},
    "S2V-DQN": {"color": "#2ca02c", "marker": "v", "linewidth": 1.5, "z": 5},
    OPT_METHOD: {"color": "#8c564b", "marker": "P", "linewidth": 1.5, "z": 9},
    "CSI-Greedy": {"color": "#ff7f0e", "marker": "s", "linewidth": 1.5, "z": 6},
}


def beta_label(beta):
    return f"{beta:g}"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def cleanup_parallel_workers():
    try:
        get_reusable_executor().shutdown(wait=True, kill_workers=True)
    except TypeError:
        get_reusable_executor().shutdown(wait=True)
    gc.collect()


def load_graph(data_name):
    matrix_path = BASE_PATH / f"{data_name}.npy"
    nodelist_path = BASE_PATH / f"{data_name}_nodelist.npy"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing graph matrix: {matrix_path}")
    if not nodelist_path.exists():
        raise FileNotFoundError(f"Missing node list: {nodelist_path}")
    graph_matrix = np.load(matrix_path, allow_pickle=True)
    node_list = np.load(nodelist_path, allow_pickle=True).tolist()
    return graph_matrix, node_list


def hypergraph_to_graph(hypergraph_matrix):
    num_nodes = hypergraph_matrix.shape[0]
    graph_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for edge in range(hypergraph_matrix.shape[1]):
        nodes_in_edge = np.where(hypergraph_matrix[:, edge] > 0)[0]
        graph_matrix[np.ix_(nodes_in_edge, nodes_in_edge)] = 1.0
    np.fill_diagonal(graph_matrix, 0.0)
    return graph_matrix


def train_cache_path(beta):
    return CACHE_DIR / f"{OPT_METHOD}_nodes_beta_{beta_label(beta)}.npy"


def train_log_path(beta):
    return CACHE_DIR / f"{OPT_METHOD}_train_log_beta_{beta_label(beta)}.json"


def load_trained_nodes(beta):
    path = train_cache_path(beta)
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True).tolist()


def save_trained_nodes(beta, nodes):
    np.save(train_cache_path(beta), np.array(nodes, dtype=object))


def eval_cache_path(beta):
    return EVAL_CACHE_DIR / f"{OPT_METHOD}_eval_beta_{beta_label(beta)}.npy"


def load_eval_cache(beta):
    path = eval_cache_path(beta)
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True).astype(float).tolist()


def save_eval_cache(beta, values):
    np.save(eval_cache_path(beta), np.array(values, dtype=float))


def run_simulation(graph_matrix, node_list, rumor_seeds, block_nodes, beta, sim_times):
    shm = SharedMemory(create=True, size=graph_matrix.nbytes)
    try:
        shm_np = np.ndarray(graph_matrix.shape, dtype=graph_matrix.dtype, buffer=shm.buf)
        np.copyto(shm_np, graph_matrix)
        return float(
            Simulations(
                sim_times,
                HG_Diffusion_SICP_pos,
                shm.name,
                graph_matrix.dtype,
                graph_matrix.shape,
                node_list,
                rumor_seeds,
                block_nodes,
                beta,
                T_MAX,
            )
        )
    finally:
        shm.close()
        shm.unlink()
        cleanup_parallel_workers()


def train_s2v_for_beta(graph_matrix, node_list, graph_adj, rumor_seeds, rumor_seed_pos, budget, beta):
    cached_nodes = load_trained_nodes(beta)
    if cached_nodes is not None:
        print(f"  Using cached {OPT_METHOD} nodes for beta={beta_label(beta)}")
        return cached_nodes[:budget]

    candidate_pos = [node for node in range(len(node_list)) if node not in rumor_seed_pos]
    agent = Compete_S2V_DQN(
        epsilon=EPSILON_START,
        alpha=ALPHA,
        gamma=GAMMA,
        adj_matrix=graph_adj,
        edge_weights=graph_adj,
        nodes_number=len(node_list),
        candidate_pos=candidate_pos,
        device=DEVICE,
        embedding_dim=EMBEDDING_DIM,
    )
    agent.s2v = agent.s2v.to(DEVICE)
    replay_buffer = S2VReplayBuffer(BUFFER_CAPACITY)
    epsilon_decay = (EPSILON_START - EPSILON_END) / TRAIN_EPISODES

    returns = []
    losses = []
    total_steps = 0

    for episode in trange(TRAIN_EPISODES, desc=f"Training {OPT_METHOD} beta={beta_label(beta)}"):
        agent.epsilon = max(EPSILON_END, EPSILON_START - episode * epsilon_decay)
        candidate_pool = set(candidate_pos.copy())
        state = []
        last_influence = 0.0
        episode_return = 0.0
        episode_losses = []
        done = False

        while not done and candidate_pool:
            action = agent.take_action(state, list(candidate_pool))
            candidate_pool.discard(action)

            influence = run_simulation(
                graph_matrix,
                node_list,
                rumor_seeds,
                state + [action],
                beta,
                TRAIN_SIM_TIMES,
            )
            reward = influence - last_influence
            last_influence = influence
            episode_return += reward

            next_state = state + [action]
            done = len(next_state) >= budget
            replay_buffer.add(state, action, reward, next_state, done)

            if len(replay_buffer) >= BATCH_SIZE:
                batch = replay_buffer.sample(BATCH_SIZE)
                episode_losses.append(float(agent.update(batch)))

            if total_steps % TARGET_UPDATE_FREQ == 0:
                agent.sync_target_net()
            total_steps += 1
            state = next_state

        returns.append(float(episode_return))
        losses.append(float(np.mean(episode_losses)) if episode_losses else 0.0)

    final_policy = []
    candidate_pool = candidate_pos.copy()
    while len(final_policy) < budget and candidate_pool:
        best_node = agent.best_action(final_policy, candidate_pool)
        final_policy.append(best_node)
        candidate_pool.remove(best_node)

    nodes = [node_list[index] for index in final_policy]
    save_trained_nodes(beta, nodes)
    with train_log_path(beta).open("w", encoding="utf-8") as file:
        json.dump(
            {
                "beta": beta,
                "train_episodes": TRAIN_EPISODES,
                "train_sim_times": TRAIN_SIM_TIMES,
                "returns": returns,
                "losses": losses,
                "nodes": nodes,
            },
            file,
            indent=4,
        )
    return nodes


def evaluate_prefixes(graph_matrix, node_list, rumor_seeds, block_nodes, beta):
    cached_values = load_eval_cache(beta)
    if cached_values is not None:
        print(
            f"  Using cached {OPT_METHOD} evaluation for beta={beta_label(beta)} "
            f"(final influence: {cached_values[-1]:.4f})"
        )
        return cached_values

    values = []
    for budget in range(1, len(block_nodes) + 1):
        shm = SharedMemory(create=True, size=graph_matrix.nbytes)
        try:
            shm_np = np.ndarray(graph_matrix.shape, dtype=graph_matrix.dtype, buffer=shm.buf)
            np.copyto(shm_np, graph_matrix)
            influence = Simulations(
                EVAL_SIM_TIMES,
                HG_Diffusion_SICP,
                shm.name,
                graph_matrix.dtype,
                graph_matrix.shape,
                node_list,
                rumor_seeds,
                block_nodes[:budget],
                beta,
                T_MAX,
            )
            values.append(float(influence))
        finally:
            shm.close()
            shm.unlink()
            cleanup_parallel_workers()

    save_eval_cache(beta, values)
    return values


def load_base_results():
    if not BASE_JSON.exists():
        return {}
    with BASE_JSON.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    loaded = {}
    for method, beta_results in payload.get("results", {}).items():
        loaded[method] = {float(beta): values for beta, values in beta_results.items()}
    return loaded


def save_outputs(results):
    payload = {
        "data_name": DATA_NAME,
        "t": T_MAX,
        "eval_simulation_times": EVAL_SIM_TIMES,
        "optimized_method": OPT_METHOD,
        "training_config": {
            "train_episodes": TRAIN_EPISODES,
            "train_sim_times": TRAIN_SIM_TIMES,
            "alpha": ALPHA,
            "gamma": GAMMA,
            "epsilon_start": EPSILON_START,
            "epsilon_end": EPSILON_END,
            "batch_size": BATCH_SIZE,
            "embedding_dim": EMBEDDING_DIM,
            "device": DEVICE,
        },
        "results": {
            method: {beta_label(beta): values for beta, values in beta_results.items()}
            for method, beta_results in results.items()
        },
        "final_budget_results": {
            method: {beta_label(beta): values[-1] for beta, values in beta_results.items()}
            for method, beta_results in results.items()
        },
    }
    json_path = RESULT_DIR / f"{DATA_NAME}_methods_beta_comparison_with_{OPT_METHOD}.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)

    methods = [method for method in METHOD_ORDER if method in results]
    csv_path = RESULT_DIR / f"{DATA_NAME}_methods_beta_final_with_{OPT_METHOD}.csv"
    with csv_path.open("w", encoding="utf-8") as file:
        file.write("beta," + ",".join(methods) + "\n")
        for beta in BETAS:
            row = [beta_label(beta)]
            for method in methods:
                values = results.get(method, {}).get(beta)
                row.append(str(values[-1]) if values else "")
            file.write(",".join(row) + "\n")

    return json_path, csv_path


def plot_final_beta_curves(results):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for method in METHOD_ORDER:
        if method not in results:
            continue
        betas = [beta for beta in BETAS if beta in results[method]]
        values = [results[method][beta][-1] for beta in betas]
        style = STYLE[method]
        ax.plot(
            betas,
            values,
            label=DISPLAY_NAME.get(method, method),
            color=style["color"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=8,
            zorder=style["z"],
        )

    ax.set_title(f"HCL-1000 Methods with {OPT_METHOD}", fontsize=14, fontweight="bold")
    ax.set_xlabel(r"$\beta$", fontsize=12)
    ax.set_ylabel("Target Expected Influence at b=10", fontsize=12)
    ax.set_xticks(BETAS)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    fig.tight_layout()

    pdf_path = RESULT_DIR / f"{DATA_NAME}_methods_beta_final_curve_with_{OPT_METHOD}.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def plot_budget_curves_by_beta(results):
    paths = []
    for beta in BETAS:
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        for method in METHOD_ORDER:
            if method not in results or beta not in results[method]:
                continue
            values = results[method][beta]
            style = STYLE[method]
            ax.plot(
                range(1, len(values) + 1),
                values,
                label=DISPLAY_NAME.get(method, method),
                color=style["color"],
                marker=style["marker"],
                linewidth=style["linewidth"],
                markersize=8,
                zorder=style["z"],
            )

        ax.set_title(rf"HCL-1000 Budget Curves with {OPT_METHOD} ($\beta={beta_label(beta)}$)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Budget (b)", fontsize=12)
        ax.set_ylabel("Target Expected Influence", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best", fontsize=10, framealpha=0.9)
        fig.tight_layout()

        path = RESULT_DIR / f"{DATA_NAME}_budget_curve_beta_{beta_label(beta)}_with_{OPT_METHOD}.pdf"
        fig.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def main():
    set_seed(SEED)
    graph_matrix, node_list = load_graph(DATA_NAME)
    graph_adj = hypergraph_to_graph(graph_matrix)
    budget = int(0.01 * len(node_list))
    rumor_seeds = H_Degree_SG(graph_matrix, node_list, budget)
    rumor_seed_pos = [node_list.index(node) for node in rumor_seeds]

    print(f"Dataset: {DATA_NAME}")
    print(f"Optimized method: {OPT_METHOD}")
    print(f"Output directory: {RESULT_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Training episodes: {TRAIN_EPISODES}, training simulations: {TRAIN_SIM_TIMES}")
    print(f"Evaluation simulations: {EVAL_SIM_TIMES}")

    results = load_base_results()
    results.setdefault(OPT_METHOD, {})

    for beta in BETAS:
        print(f"\nProcessing {OPT_METHOD} beta={beta_label(beta)}")
        nodes = train_s2v_for_beta(
            graph_matrix,
            node_list,
            graph_adj,
            rumor_seeds,
            rumor_seed_pos,
            budget,
            beta,
        )
        results[OPT_METHOD][beta] = evaluate_prefixes(
            graph_matrix,
            node_list,
            rumor_seeds,
            nodes[:budget],
            beta,
        )
        save_outputs(results)
        print(f"  {OPT_METHOD} final influence: {results[OPT_METHOD][beta][-1]:.4f}")

    json_path, csv_path = save_outputs(results)
    final_curve_path = plot_final_beta_curves(results)
    budget_curve_paths = plot_budget_curves_by_beta(results)

    print("\nSaved optimized S2V comparison outputs:")
    print(f"  JSON: {json_path}")
    print(f"  Final CSV: {csv_path}")
    print(f"  Final beta curve: {final_curve_path}")
    for path in budget_curve_paths:
        print(f"  Budget curve: {path}")


if __name__ == "__main__":
    main()
