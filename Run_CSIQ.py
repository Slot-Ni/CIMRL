"""Main HCDP-DQN experiment runner.

Run from the repository root:
    python Run_CSIQ.py --group all
    python Run_CSIQ.py --group real
    python Run_CSIQ.py --group synthetic
    python Run_CSIQ.py --dataset Algebra

Main-paper settings are centralized in PAPER_CONFIG.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from multiprocessing.shared_memory import SharedMemory

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from tqdm import trange

from CompeteAlgorithm import Compete_DQN, ReplayBuffer
from Diffusion_In_HyperGraph import HG_Diffusion_SICP, HG_Diffusion_SICP_pos, Simulations
from Indicator import (
    Competition_composite_indicator_new,
    get_avg_HigerOrder_dis,
    get_node_hyperedge_avgsize,
)
from SeedNodesGenerator import H_Degree_SG

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
GRAPH_DIR = PROJECT_ROOT / "HyperGraphMatrix"
RESULT_DIR = SCRIPT_DIR / "results" / "main_hcdp_dqn"

REAL_DATASETS = [
    "Algebra", "Restaurants-Rev", "Geometry", "Music-Rev",
    "NDC-classes", "Bars-Rev", "iAF1260b", "iJO1366", "soc-hamsterster",
]
SYNTHETIC_DATASETS = [
    f"synthetic_HCL_v{n}_e{n}_exp{r:.1f}_maxdeg10"
    for n in (1000, 2000, 3000)
    for r in (2.0, 2.5, 3.0)
]

# Main experimental settings transcribed from Section 5 and Table 3.
PAPER_CONFIG = {
    "random_seed": 42,
    "seed_budget_ratio": 0.01,  # matches the repository's main-run convention
    "candidate_ratio": 0.20,
    "max_structural_order": 10,
    "order_weight_alpha": 0.5,
    "propagation_horizon": 30,
    "reward_simulations": 50,
    "final_evaluation_simulations": 10_000,
    "hidden_layer_widths": [256, 128],
    "learning_rate": 0.03,
    "discount_factor": 0.9,
    "training_episodes": 600,
    "replay_buffer_capacity": 30_000,
    "batch_size": 64,
    "target_update_interval": 100,
    "epsilon_start": 0.5,
    "epsilon_end": 0.01,
    "real_world_betas": [0.025, 0.05],
    "synthetic_beta": 0.025,
}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_graph(dataset: str):
    matrix_path = GRAPH_DIR / f"{dataset}.npy"
    nodes_path = GRAPH_DIR / f"{dataset}_nodelist.npy"
    if not matrix_path.is_file() or not nodes_path.is_file():
        raise FileNotFoundError(
            f"Missing inputs for {dataset}: expected {matrix_path} and {nodes_path}"
        )
    matrix = np.load(matrix_path, allow_pickle=True)
    nodes = np.load(nodes_path, allow_pickle=True).tolist()
    if matrix.ndim != 2 or matrix.shape[0] != len(nodes):
        raise ValueError(f"Invalid incidence matrix/node-list dimensions: {dataset}")
    return matrix, nodes

def run_one(dataset: str, beta: float, target_budget=None) -> dict:
    cfg = PAPER_CONFIG
    set_seed(cfg["random_seed"])
    started = time.perf_counter()

    graph, nodes = load_graph(dataset)
    n = len(nodes)
    budget = target_budget if target_budget is not None else max(1, int(cfg["seed_budget_ratio"] * n))
    if not 1 <= budget < n: raise ValueError("Invalid target budget")
    competing_budget = max(1, int(cfg["seed_budget_ratio"] * n))
    competing_seeds = H_Degree_SG(graph, nodes, competing_budget)
    competing_pos = {nodes.index(v) for v in competing_seeds}

    # Calculate HCDP features once for this competing seed set.
    avg_edge_size = get_node_hyperedge_avgsize(graph)
    avg_higher_order = get_avg_HigerOrder_dis(
        [nodes.index(v) for v in competing_seeds],
        graph, dataset, cfg["max_structural_order"], cfg["order_weight_alpha"]
    )

    # Rank all vertices, then explicitly remove competing seeds.
    ranked = Competition_composite_indicator_new(
        [nodes.index(v) for v in competing_seeds],
        graph, dataset, n, cfg["max_structural_order"], cfg["order_weight_alpha"]
    )
    candidate_count = min(n - len(competing_pos),
                          max(budget, int(cfg["candidate_ratio"] * n)))
    candidates = [int(v) for v in ranked if int(v) not in competing_pos][:candidate_count]
    if len(candidates) < budget:
        raise RuntimeError(f"{dataset}: candidate pool smaller than target budget.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = Compete_DQN(
        epsilon=cfg["epsilon_start"],
        alpha=cfg["learning_rate"],
        gamma=cfg["discount_factor"],
        state_dim=2 * budget,
        action_dim=len(candidates),
        node_hyperedge_avgsize=avg_edge_size,
        avg_HigerOrder_dis_seednodes=avg_higher_order,
        candidate_nodes_pos=candidates,
        device=device,
    )
    memory = ReplayBuffer(cfg["replay_buffer_capacity"])
    epsilon_decay = (cfg["epsilon_start"] - cfg["epsilon_end"]) / max(1, cfg["training_episodes"] - 1)

    shm = SharedMemory(create=True, size=graph.nbytes)
    try:
        shared = np.ndarray(graph.shape, dtype=graph.dtype, buffer=shm.buf)
        np.copyto(shared, graph)
        total_steps = 0

        for episode in trange(cfg["training_episodes"], desc=f"{dataset}, beta={beta:g}"):
            agent.epsilon = max(
                cfg["epsilon_end"], cfg["epsilon_start"] - episode * epsilon_decay
            )
            available = set(candidates)
            state = []
            previous_spread = 0.0

            for _ in range(budget):
                action, action_index = agent.take_action(state, list(available))
                available.remove(action)
                spread = Simulations(
                    cfg["reward_simulations"], HG_Diffusion_SICP_pos,
                    shm.name, graph.dtype, graph.shape,
                    nodes, competing_seeds, state + [action],
                    beta, cfg["propagation_horizon"],
                )
                reward = float(spread - previous_spread)
                previous_spread = float(spread)
                next_state = state + [action]
                done = len(next_state) == budget
                memory.add(
                    agent._encode_state(state), action_index, reward,
                    agent._encode_state(next_state), done,
                    np.array([v in available for v in candidates], dtype=bool)
                )
                if len(memory) >= cfg["batch_size"]:
                    agent.update(zip(*memory.sample(cfg["batch_size"])))
                total_steps += 1
                if total_steps % cfg["target_update_interval"] == 0:
                    agent.sync_target_net()
                state = next_state

        # Greedy inference; no exploration during final selection.
        chosen = []
        available = set(candidates)
        state = []
        for _ in range(budget):
            action = agent.best_action(state, list(available))
            chosen.append(int(action))
            available.remove(action)
            state.append(action)
    finally:
        shm.close()
        shm.unlink()

    selected_nodes = [nodes[i] for i in chosen]

    # Final held-out evaluation: 10,000 simulations for each reported prefix.
    eval_shm = SharedMemory(create=True, size=graph.nbytes)
    try:
        shared = np.ndarray(graph.shape, dtype=graph.dtype, buffer=eval_shm.buf)
        np.copyto(shared, graph)
        influence_curve = []
        for k in range(1, len(selected_nodes) + 1):
            value = Simulations(
                cfg["final_evaluation_simulations"], HG_Diffusion_SICP,
                eval_shm.name, graph.dtype, graph.shape,
                nodes, competing_seeds, selected_nodes[:k],
                beta, cfg["propagation_horizon"], seed_base=1_000_000,
            )
            influence_curve.append(float(value))
    finally:
        eval_shm.close()
        eval_shm.unlink()

    result = {
        "dataset": dataset,
        "beta1": beta,
        "beta2": beta,
        "target_budget": budget,
        "candidate_count": len(candidates),
        "selected_nodes": [v.item() if isinstance(v, np.generic) else v for v in selected_nodes],
        "tie_priority": "competitor",
        "influence_by_budget_prefix": influence_curve,
        "elapsed_seconds_including_final_evaluation": time.perf_counter() - started,
        "configuration": cfg,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    beta_tag = str(beta).replace(".", "p")
    output = RESULT_DIR / f"{dataset}_beta_{beta_tag}.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output}")
    return result

def main():
    parser = argparse.ArgumentParser(description="HCDP-DQN main-paper experiments")
    parser.add_argument("--budget", type=int, help="Explicit target budget; default is floor(1 percent of nodes)")
    parser.add_argument("--group", choices=["all", "real", "synthetic"], default="all")
    parser.add_argument("--dataset", help="Run one dataset only")
    parser.add_argument("--beta", type=float, help="Override beta for one dataset")
    args = parser.parse_args()

    # Indicator.py expects the repository's sibling Hyper_Distance directory.
    os.chdir(SCRIPT_DIR)

    if args.dataset:
        datasets = [args.dataset]
    elif args.group == "real":
        datasets = REAL_DATASETS
    elif args.group == "synthetic":
        datasets = SYNTHETIC_DATASETS
    else:
        datasets = REAL_DATASETS + SYNTHETIC_DATASETS

    for dataset in datasets:
        if args.beta is not None:
            betas = [args.beta]
        elif dataset in REAL_DATASETS:
            betas = PAPER_CONFIG["real_world_betas"]
        else:
            betas = [PAPER_CONFIG["synthetic_beta"]]
        for beta in betas:
            run_one(dataset, beta, args.budget)

if __name__ == "__main__":
    main()
