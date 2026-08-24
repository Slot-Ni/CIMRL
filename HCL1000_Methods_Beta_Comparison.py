import json
import os
import random
import gc
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from joblib.externals.loky import get_reusable_executor
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from Baseline_Algorithm import CI, HADP, H_Degree, Indicator_selection_Greedy
from Baseline_S2V import Compete_S2V_DQN
from Diffusion_In_HyperGraph import HG_Diffusion_SICP, Simulations
from Indicator import Competition_composite_indicator_new
from SeedNodesGenerator import H_Degree_SG


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_NAME = "synthetic_HCL_v1000_e1000_exp2.5_maxdeg10"
BETAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
T_MAX = 30
RATIO = 0.2
SIM_TIMES = 1000
SEED = 42

# CSI-Greedy is much more expensive because it runs diffusion repeatedly while
# selecting nodes. Enable it only when you need the full HCL_evaluate baseline.
RUN_CSI_GREEDY = True
CSI_GREEDY_TIMES = 100

BASE_PATH = PROJECT_ROOT / "HyperGraphMatrix"
NEW_LIST_DIRS = [SCRIPT_DIR / "New_list", SCRIPT_DIR / "New_List"]
RESULT_DIR = SCRIPT_DIR / "HCL1000_Beta_Method_Comparison"
CACHE_DIR = RESULT_DIR / "cache"
EVAL_CACHE_DIR = RESULT_DIR / "eval_cache"
RESULT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
EVAL_CACHE_DIR.mkdir(exist_ok=True)

METHOD_ORDER = ["CSI-Greedy", "HDegree", "HADP", "CI", "S2V-DQN", "CSIQ"]
DISPLAY_NAME = {
    "CSIQ": "HCDP-DQN",
    "HDegree": "HDgree",
    "HADP": "HADP",
    "CI": "CI",
    "S2V-DQN": "S2V-DQN",
    "CSI-Greedy": "HCDP-Greedy",
}
STYLE = {
    "CSIQ": {"color": "#1f77b4", "marker": "o", "linewidth": 1.5, "z": 10},
    "HDegree": {"color": "#EC6E66", "marker": "D", "linewidth": 1.5, "z": 4},
    "HADP": {"color": "#FAC7B3", "marker": "^", "linewidth": 1.5, "z": 3},
    "CI": {"color": "#91CCC0", "marker": "*", "linewidth": 1.5, "z": 3},
    "S2V-DQN": {"color": "#2ca02c", "marker": "v", "linewidth": 1.5, "z": 5},
    "CSI-Greedy": {"color": "#ff7f0e", "marker": "s", "linewidth": 1.5, "z": 6},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def beta_label(beta):
    return f"{beta:g}"


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
    num_hyperedges = hypergraph_matrix.shape[1]
    for edge in range(num_hyperedges):
        nodes_in_edge = np.where(hypergraph_matrix[:, edge] > 0)[0]
        for src in nodes_in_edge:
            for dst in nodes_in_edge:
                if src != dst:
                    graph_matrix[src, dst] = 1.0
    return graph_matrix


def find_csiq_file(data_name, beta):
    filename = (
        f"indicator_selection_RL_11_({data_name})_(T_{T_MAX})_"
        f"(beta_{beta_label(beta)})_(ratio_{RATIO}).npy"
    )
    for directory in NEW_LIST_DIRS:
        file_path = directory / filename
        if file_path.exists():
            return file_path
    raise FileNotFoundError(f"Missing CSIQ result file: {filename}")


def save_node_cache(method, beta, nodes):
    cache_path = CACHE_DIR / f"{method}_nodes_beta_{beta_label(beta)}.npy"
    np.save(cache_path, np.array(nodes, dtype=object))
    return cache_path


def load_node_cache(method, beta):
    cache_path = CACHE_DIR / f"{method}_nodes_beta_{beta_label(beta)}.npy"
    if not cache_path.exists():
        return None
    return np.load(cache_path, allow_pickle=True).tolist()


def s2v_cache_path():
    return CACHE_DIR / "S2V-DQN_nodes.npy"


def load_s2v_cache():
    cache_path = s2v_cache_path()
    if not cache_path.exists():
        return None
    return np.load(cache_path, allow_pickle=True).tolist()


def save_s2v_cache(nodes):
    np.save(s2v_cache_path(), np.array(nodes, dtype=object))


def eval_cache_path(method, beta):
    safe_method = method.replace("/", "_")
    return EVAL_CACHE_DIR / f"{safe_method}_eval_beta_{beta_label(beta)}.npy"


def load_eval_cache(method, beta):
    cache_path = eval_cache_path(method, beta)
    if not cache_path.exists():
        return None
    return np.load(cache_path, allow_pickle=True).astype(float).tolist()


def save_eval_cache(method, beta, values):
    cache_path = eval_cache_path(method, beta)
    np.save(cache_path, np.array(values, dtype=float))
    return cache_path


def cleanup_parallel_workers():
    try:
        get_reusable_executor().shutdown(wait=True, kill_workers=True)
    except TypeError:
        get_reusable_executor().shutdown(wait=True)
    gc.collect()


def get_s2v_nodes(graph_matrix, node_list, rumor_seed_pos, budget, device):
    cached = load_s2v_cache()
    if cached is not None:
        return cached[:budget]

    graph_adj = hypergraph_to_graph(graph_matrix)
    candidate_pos = [node for node in range(len(node_list)) if node not in rumor_seed_pos]
    s2v_agent = Compete_S2V_DQN(
        epsilon=0,
        alpha=0.01,
        gamma=0.8,
        adj_matrix=graph_adj,
        edge_weights=graph_adj,
        nodes_number=len(node_list),
        candidate_pos=candidate_pos,
        device=device,
        embedding_dim=64,
    )

    selected_pos = []
    temp_candidates = candidate_pos.copy()
    for _ in range(budget):
        best_node = s2v_agent.best_action(selected_pos, temp_candidates)
        selected_pos.append(best_node)
        temp_candidates.remove(best_node)

    nodes = [node_list[index] for index in selected_pos]
    save_s2v_cache(nodes)
    return nodes


def prepare_method_nodes(graph_matrix, node_list, rumor_seeds, rumor_seed_pos, budget, beta, device):
    methods = {}

    csiq_path = find_csiq_file(DATA_NAME, beta)
    methods["CSIQ"] = np.load(csiq_path, allow_pickle=True).tolist()[:budget]

    cached = load_node_cache("HDegree", beta)
    methods["HDegree"] = cached if cached is not None else H_Degree(graph_matrix, node_list, rumor_seeds, budget)
    save_node_cache("HDegree", beta, methods["HDegree"])

    cached = load_node_cache("HADP", beta)
    methods["HADP"] = cached if cached is not None else HADP(graph_matrix, node_list, rumor_seeds, budget)
    save_node_cache("HADP", beta, methods["HADP"])

    cached = load_node_cache("CI", beta)
    methods["CI"] = cached if cached is not None else CI(graph_matrix, node_list, rumor_seeds, budget)
    save_node_cache("CI", beta, methods["CI"])

    methods["S2V-DQN"] = get_s2v_nodes(graph_matrix, node_list, rumor_seed_pos, budget, device)

    if RUN_CSI_GREEDY:
        cached = load_node_cache("CSI-Greedy", beta)
        if cached is None:
            candidate_pos = Competition_composite_indicator_new(
                rumor_seed_pos,
                graph_matrix,
                DATA_NAME,
                int(RATIO * len(node_list)),
                10,
                0.5,
            ).tolist()
            cached = Indicator_selection_Greedy(
                graph_matrix,
                node_list,
                rumor_seeds,
                candidate_pos,
                budget,
                CSI_GREEDY_TIMES,
                beta,
                T_MAX,
            )
            save_node_cache("CSI-Greedy", beta, cached)
        methods["CSI-Greedy"] = cached

    return methods


def evaluate_prefixes(graph_matrix, node_list, rumor_seeds, block_nodes, beta):
    values = []
    for budget in range(1, len(block_nodes) + 1):
        shm = SharedMemory(create=True, size=graph_matrix.nbytes)
        try:
            shm_np = np.ndarray(graph_matrix.shape, dtype=graph_matrix.dtype, buffer=shm.buf)
            np.copyto(shm_np, graph_matrix)
            influence = Simulations(
                SIM_TIMES,
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
    return values


def save_outputs(results):
    payload = {
        "data_name": DATA_NAME,
        "t": T_MAX,
        "ratio": RATIO,
        "simulation_times": SIM_TIMES,
        "run_csi_greedy": RUN_CSI_GREEDY,
        "results": {
            method: {beta_label(beta): values for beta, values in beta_results.items()}
            for method, beta_results in results.items()
        },
        "final_budget_results": {
            method: {beta_label(beta): values[-1] for beta, values in beta_results.items()}
            for method, beta_results in results.items()
        },
    }
    json_path = RESULT_DIR / f"{DATA_NAME}_methods_beta_comparison.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)

    csv_path = RESULT_DIR / f"{DATA_NAME}_methods_beta_final.csv"
    methods = [method for method in METHOD_ORDER if method in results]
    with csv_path.open("w", encoding="utf-8") as file:
        file.write("beta," + ",".join(methods) + "\n")
        for beta in BETAS:
            row = [beta_label(beta)]
            for method in methods:
                values = results[method].get(beta)
                row.append(str(values[-1]) if values else "")
            file.write(",".join(row) + "\n")

    return json_path, csv_path


def load_existing_results():
    json_path = RESULT_DIR / f"{DATA_NAME}_methods_beta_comparison.json"
    if not json_path.exists():
        return None

    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    results = {}
    for method, beta_results in payload.get("results", {}).items():
        results[method] = {float(beta): values for beta, values in beta_results.items()}
    return results


def plot_final_beta_curves(results):
    fig, ax = plt.subplots(figsize=(7, 5))
    x_values = BETAS
    for method in METHOD_ORDER:
        if method not in results:
            continue
        y_values = [results[method][beta][-1] for beta in BETAS if beta in results[method]]
        method_betas = [beta for beta in BETAS if beta in results[method]]
        style = STYLE[method]
        ax.plot(
            method_betas,
            y_values,
        label=DISPLAY_NAME.get(method, method),
            color=style["color"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=8,
            zorder=style["z"],
        )

    ax.set_title("HCL-1000 Methods under Different Betas", fontsize=14, fontweight="bold")
    ax.set_xlabel(r"$\beta$", fontsize=12)
    ax.set_ylabel("Target Expected Influence at b=10", fontsize=12)
    ax.set_xticks(x_values)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", fontsize=11, framealpha=0.9)
    fig.tight_layout()

    pdf_path = RESULT_DIR / f"{DATA_NAME}_methods_beta_final_curve.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def plot_budget_curves_by_beta(results):
    pdf_paths = []
    for beta in BETAS:
        fig, ax = plt.subplots(figsize=(7, 5))
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

        ax.set_title(rf"HCL-1000 Budget Curves ($\beta={beta_label(beta)}$)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Budget (b)", fontsize=12)
        ax.set_ylabel("Target Expected Influence", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best", fontsize=11, framealpha=0.9)
        fig.tight_layout()

        pdf_path = RESULT_DIR / f"{DATA_NAME}_budget_curve_beta_{beta_label(beta)}.pdf"
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        pdf_paths.append(pdf_path)
    return pdf_paths


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph_matrix, node_list = load_graph(DATA_NAME)
    budget = int(0.01 * len(node_list))
    rumor_seeds = H_Degree_SG(graph_matrix, node_list, budget)
    rumor_seed_pos = [node_list.index(node) for node in rumor_seeds]

    print(f"Dataset: {DATA_NAME}")
    print(f"Nodes: {len(node_list)}, hyperedges: {graph_matrix.shape[1]}, budget: {budget}")
    print(f"Betas: {', '.join(beta_label(beta) for beta in BETAS)}")
    print(f"Simulation times per method/budget point: {SIM_TIMES}")
    print(f"RUN_CSI_GREEDY: {RUN_CSI_GREEDY}")
    print(f"S2V device: {device}")

    results = load_existing_results()
    if results is None:
        results = {}
        for beta in BETAS:
            print(f"\nPreparing methods for beta={beta_label(beta)}")
            method_nodes = prepare_method_nodes(
                graph_matrix,
                node_list,
                rumor_seeds,
                rumor_seed_pos,
                budget,
                beta,
                device,
            )

            for method, block_nodes in method_nodes.items():
                results.setdefault(method, {})
                cached_values = load_eval_cache(method, beta)
                if cached_values is not None:
                    results[method][beta] = cached_values
                    print(
                        f"  Using cached evaluation for {method} at beta={beta_label(beta)} "
                        f"(final influence: {cached_values[-1]:.4f})"
                    )
                    continue

                print(f"  Evaluating {method} at beta={beta_label(beta)}")
                results[method][beta] = evaluate_prefixes(
                    graph_matrix,
                    node_list,
                    rumor_seeds,
                    block_nodes,
                    beta,
                )
                save_eval_cache(method, beta, results[method][beta])
                save_outputs(results)
                print(f"    final influence: {results[method][beta][-1]:.4f}")
    else:
        print(f"Loaded existing results JSON for {DATA_NAME}; regenerating plots only.")

    json_path, csv_path = save_outputs(results)
    final_curve_path = plot_final_beta_curves(results)
    budget_curve_paths = plot_budget_curves_by_beta(results)

    print("\nSaved comparison outputs:")
    print(f"  JSON: {json_path}")
    print(f"  Final CSV: {csv_path}")
    print(f"  Final beta curve: {final_curve_path}")
    for path in tqdm(budget_curve_paths, desc="Budget curve files"):
        print(f"  Budget curve: {path}")


if __name__ == "__main__":
    main()
