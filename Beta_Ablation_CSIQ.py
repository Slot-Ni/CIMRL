import json
import os
import random
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
except ImportError:
    torch = None

from Diffusion_In_HyperGraph import HG_Diffusion_SICP, Simulations
from SeedNodesGenerator import H_Degree_SG


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_NAME = "synthetic_HCL_v1000_e1000_exp2.5_maxdeg10"
BETAS = [0.01, 0.02, 0.03, 0.04, 0.05]
T_MAX = 30
RATIO = 0.2
SIM_TIMES = 1000
SEED = 42

BASE_PATH = PROJECT_ROOT / "HyperGraphMatrix"
NEW_LIST_DIR = SCRIPT_DIR / "New_list"
if not NEW_LIST_DIR.exists():
    NEW_LIST_DIR = SCRIPT_DIR / "New_List"

RESULT_DIR = SCRIPT_DIR / "Beta_Ablation_CSIQ"
RESULT_DIR.mkdir(exist_ok=True)
BASE_COMPARE_JSON = SCRIPT_DIR / "HCL1000_Beta_Method_Comparison" / f"{DATA_NAME}_methods_beta_comparison.json"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
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


def load_csiq_nodes(data_name, beta):
    file_path = NEW_LIST_DIR / (
        f"indicator_selection_RL_11_({data_name})_(T_{T_MAX})_"
        f"(beta_{beta_label(beta)})_(ratio_{RATIO}).npy"
    )
    if not file_path.exists():
        raise FileNotFoundError(f"Missing CSIQ result: {file_path}")
    return np.load(file_path, allow_pickle=True).tolist(), file_path


def load_csiq_results_from_compare_json(data_name):
    if not BASE_COMPARE_JSON.exists():
        return None
    with BASE_COMPARE_JSON.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    results = payload.get("results", {}).get("CSIQ", {})
    return {float(beta): values for beta, values in results.items()}


def evaluate_seed_prefixes(graph_matrix, node_list, rumor_seeds, block_nodes, beta):
    shm = SharedMemory(create=True, size=graph_matrix.nbytes)
    try:
        shm_np = np.ndarray(graph_matrix.shape, dtype=graph_matrix.dtype, buffer=shm.buf)
        np.copyto(shm_np, graph_matrix)

        results = []
        for budget in range(1, len(block_nodes) + 1):
            subset = block_nodes[:budget]
            influence = Simulations(
                SIM_TIMES,
                HG_Diffusion_SICP,
                shm.name,
                graph_matrix.dtype,
                graph_matrix.shape,
                node_list,
                rumor_seeds,
                subset,
                beta,
                T_MAX,
            )
            results.append(float(influence))
        return results
    finally:
        shm.close()
        shm.unlink()


def save_json(results, file_paths):
    payload = {
        "data_name": DATA_NAME,
        "t": T_MAX,
        "ratio": RATIO,
        "simulation_times": SIM_TIMES,
        "evaluation_mode": "matched_beta",
        "source_files": {beta_label(beta): str(path) for beta, path in file_paths.items()},
        "results": {beta_label(beta): values for beta, values in results.items()},
        "final_budget_results": {
            beta_label(beta): values[-1] if values else None for beta, values in results.items()
        },
    }

    json_path = RESULT_DIR / f"{DATA_NAME}_beta_ablation_plot_data.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)
    return json_path


def save_csv(results):
    csv_path = RESULT_DIR / f"{DATA_NAME}_beta_ablation.csv"
    max_budget = max(len(values) for values in results.values())
    with csv_path.open("w", encoding="utf-8") as file:
        file.write("budget," + ",".join(f"beta_{beta_label(beta)}" for beta in BETAS) + "\n")
        for budget in range(1, max_budget + 1):
            row = [str(budget)]
            for beta in BETAS:
                values = results.get(beta, [])
                row.append(str(values[budget - 1]) if budget <= len(values) else "")
            file.write(",".join(row) + "\n")
    return csv_path


def plot_results(results):
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = ["o", "s", "^", "D", "v"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, beta in enumerate(BETAS):
        values = results[beta]
        budgets = range(1, len(values) + 1)
        ax.plot(
            budgets,
            values,
            label=rf"$\beta={beta_label(beta)}$",
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            linewidth=1.5,
            markersize=7,
        )

    ax.set_title("HCDP-DQN Beta Ablation on HCL", fontsize=14, fontweight="bold")
    ax.set_xlabel("Budget (b)", fontsize=12)
    ax.set_ylabel("Target Expected Influence", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", fontsize=11, framealpha=0.9)
    fig.tight_layout()

    curve_path = RESULT_DIR / f"{DATA_NAME}_beta_ablation_curve.pdf"
    fig.savefig(curve_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    final_values = [results[beta][-1] for beta in BETAS]
    labels = [beta_label(beta) for beta in BETAS]
    ax.bar(labels, final_values, color=colors[: len(BETAS)], width=0.6)
    ax.set_title("Final Budget Performance", fontsize=14, fontweight="bold")
    ax.set_xlabel(r"$\beta$", fontsize=12)
    ax.set_ylabel("Target Expected Influence", fontsize=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()

    bar_path = RESULT_DIR / f"{DATA_NAME}_beta_ablation_final_bar.pdf"
    fig.savefig(bar_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return curve_path, bar_path


def main():
    set_seed(SEED)
    graph_matrix, node_list = load_graph(DATA_NAME)
    budget = int(0.01 * len(node_list))
    rumor_seeds = H_Degree_SG(graph_matrix, node_list, budget)

    print(f"Dataset: {DATA_NAME}")
    print(f"Nodes: {len(node_list)}, hyperedges: {graph_matrix.shape[1]}, budget: {budget}")
    print(f"Simulation times per prefix: {SIM_TIMES}")

    compare_results = load_csiq_results_from_compare_json(DATA_NAME)
    results = {}
    file_paths = {}
    for beta in BETAS:
        if compare_results and beta in compare_results:
            results[beta] = compare_results[beta]
            print(f"Using cached beta curve from comparison JSON for beta={beta_label(beta)}")
            continue

        block_nodes, file_path = load_csiq_nodes(DATA_NAME, beta)
        block_nodes = block_nodes[:budget]
        file_paths[beta] = file_path
        print(f"Evaluating beta={beta_label(beta)} from {file_path.name}")
        results[beta] = evaluate_seed_prefixes(
            graph_matrix,
            node_list,
            rumor_seeds,
            block_nodes,
            beta,
        )
        print(f"  final influence: {results[beta][-1]:.4f}")

    json_path = save_json(results, file_paths)
    csv_path = save_csv(results)
    curve_path, bar_path = plot_results(results)

    print("\nSaved beta ablation outputs:")
    print(f"  JSON: {json_path}")
    print(f"  CSV: {csv_path}")
    print(f"  Curve: {curve_path}")
    print(f"  Final bar: {bar_path}")


if __name__ == "__main__":
    main()
