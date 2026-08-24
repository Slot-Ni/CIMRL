import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from SeedNodesGenerator import H_Degree_SG


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

GRAPH_DIR = PROJECT_ROOT / "HyperGraphMatrix"
DISTANCE_DIR = PROJECT_ROOT / "Hyper_Distance"
CSIQ_DIRS = [
    SCRIPT_DIR / "New_list",
    SCRIPT_DIR / "New_List",
    SCRIPT_DIR / "HCL_synthetic_comparison" / "New_list",
]
RESULT_DIR = SCRIPT_DIR / "CSI_Feature_Cluster_Analysis"

DEFAULT_DATA = "synthetic_HCL_v1000_e1000_exp2.5_maxdeg10"
DEFAULT_BETAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]


def beta_label(beta):
    return f"{beta:g}"


def load_graph(data_name):
    matrix_path = GRAPH_DIR / f"{data_name}.npy"
    nodelist_path = GRAPH_DIR / f"{data_name}_nodelist.npy"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing graph matrix: {matrix_path}")
    if not nodelist_path.exists():
        raise FileNotFoundError(f"Missing node list: {nodelist_path}")
    graph_matrix = np.load(matrix_path, allow_pickle=True)
    node_list = np.load(nodelist_path, allow_pickle=True).tolist()
    return graph_matrix, node_list


def get_node_hyperedge_avgsize(graph_matrix):
    hyperedge_size = np.sum(graph_matrix, axis=0)
    values = np.zeros(graph_matrix.shape[0], dtype=float)
    for node in range(graph_matrix.shape[0]):
        edge_pos = np.where(graph_matrix[node] == 1)[0]
        values[node] = float(np.mean(hyperedge_size[edge_pos])) if len(edge_pos) else 0.0
    return values


def get_node_hyperedge_size_stats(graph_matrix):
    hyperedge_size = np.sum(graph_matrix, axis=0)
    stats = np.zeros((graph_matrix.shape[0], 5), dtype=float)
    for node in range(graph_matrix.shape[0]):
        edge_pos = np.where(graph_matrix[node] == 1)[0]
        if len(edge_pos) == 0:
            continue
        sizes = hyperedge_size[edge_pos].astype(float)
        stats[node] = [
            float(np.mean(sizes)),
            float(np.std(sizes)),
            float(np.min(sizes)),
            float(np.max(sizes)),
            float(len(sizes)),
        ]
    return stats


def get_avg_higher_order_distance(seed_positions, data_name, node_count, max_order, alpha):
    values = np.zeros(node_count, dtype=float)
    seed_positions = list(seed_positions)
    for order in range(1, max_order + 1):
        distance_path = DISTANCE_DIR / f"{data_name}_distance_{order}.npy"
        if not distance_path.exists():
            raise FileNotFoundError(f"Missing higher-order distance matrix: {distance_path}")
        order_matrix = np.load(distance_path, allow_pickle=True)
        seed_distances = order_matrix[:, seed_positions].astype(float)
        seed_distances[np.isinf(seed_distances)] = 0.0
        values += (1 / pow(alpha, order)) * np.mean(seed_distances, axis=1)
    values[seed_positions] = 0.0
    return values


def get_per_order_distance_features(seed_positions, data_name, node_count, max_order):
    features = []
    seed_positions = list(seed_positions)
    for order in range(1, max_order + 1):
        distance_path = DISTANCE_DIR / f"{data_name}_distance_{order}.npy"
        if not distance_path.exists():
            raise FileNotFoundError(f"Missing higher-order distance matrix: {distance_path}")
        order_matrix = np.load(distance_path, allow_pickle=True)
        seed_distances = order_matrix[:, seed_positions].astype(float)
        seed_distances[np.isinf(seed_distances)] = 0.0
        features.append(np.mean(seed_distances, axis=1))
    return np.column_stack(features)


def compute_csi_features(graph_matrix, node_list, data_name, max_order, alpha, candidate_ratio):
    budget = max(1, int(0.01 * len(node_list)))
    rumor_seeds = H_Degree_SG(graph_matrix, node_list, budget)
    rumor_seed_pos = [node_list.index(node) for node in rumor_seeds]

    hyperedge_size_stats = get_node_hyperedge_size_stats(graph_matrix)
    avg_size = hyperedge_size_stats[:, 0]
    higher_order_distance = get_avg_higher_order_distance(
        rumor_seed_pos,
        data_name,
        len(node_list),
        max_order,
        alpha,
    )
    csi_score = avg_size * higher_order_distance
    csi_score[rumor_seed_pos] = -np.inf
    per_order_distance = get_per_order_distance_features(
        rumor_seed_pos,
        data_name,
        len(node_list),
        max_order,
    )

    candidate_count = max(budget, int(candidate_ratio * len(node_list)))
    candidate_pos = np.argsort(-csi_score)[:candidate_count]
    feature_matrix = np.column_stack([avg_size[candidate_pos], higher_order_distance[candidate_pos]])
    multiorder_feature_matrix = np.column_stack(
        [
            hyperedge_size_stats[candidate_pos],
            per_order_distance[candidate_pos],
            csi_score[candidate_pos],
        ]
    )

    return {
        "budget": budget,
        "rumor_seeds": rumor_seeds,
        "rumor_seed_pos": rumor_seed_pos,
        "candidate_pos": candidate_pos,
        "features": feature_matrix,
        "tsne_features": multiorder_feature_matrix,
        "csi_score": csi_score[candidate_pos],
    }


def find_csiq_file(data_name, beta, ratio):
    filename = (
        f"indicator_selection_RL_11_({data_name})_(T_30)_"
        f"(beta_{beta_label(beta)})_(ratio_{beta_label(ratio)}).npy"
    )
    for directory in CSIQ_DIRS:
        file_path = directory / filename
        if file_path.exists():
            return file_path
    raise FileNotFoundError(f"Missing CSIQ high-influence node file: {filename}")


def load_high_influence_positions(data_name, beta, ratio, node_list, budget):
    csiq_file = find_csiq_file(data_name, beta, ratio)
    high_nodes = np.load(csiq_file, allow_pickle=True).tolist()[:budget]
    node_to_pos = {node: index for index, node in enumerate(node_list)}
    missing = [node for node in high_nodes if node not in node_to_pos]
    if missing:
        raise ValueError(f"{len(missing)} CSIQ nodes are not in node_list. Example: {missing[:5]}")
    return {node_to_pos[node] for node in high_nodes}, csiq_file


def hypergeom_sf(total, success_total, draws, observed):
    if observed <= 0:
        return 1.0
    denominator = math.comb(total, draws)
    upper = min(success_total, draws)
    probability = 0.0
    for value in range(observed, upper + 1):
        probability += (
            math.comb(success_total, value)
            * math.comb(total - success_total, draws - value)
            / denominator
        )
    return min(1.0, probability)


def fit_clusters(method, scaled_features, cluster_count, seed):
    if method == "kmeans":
        model = KMeans(n_clusters=cluster_count, random_state=seed, n_init=20)
        labels = model.fit_predict(scaled_features)
        return labels, float(model.inertia_), None
    if method == "gmm":
        model = GaussianMixture(
            n_components=cluster_count,
            covariance_type="full",
            random_state=seed,
            n_init=10,
        )
        labels = model.fit_predict(scaled_features)
        return labels, None, float(model.bic(scaled_features))
    raise ValueError(f"Unknown cluster method: {method}")


def summarize_clusters(labels, features, csi_scores, candidate_pos, high_pos):
    rows = []
    high_mask = np.array([pos in high_pos for pos in candidate_pos], dtype=bool)
    total = len(candidate_pos)
    success_total = int(np.sum(high_mask))
    global_ratio = success_total / total if total else 0.0

    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        size = int(np.sum(mask))
        high_count = int(np.sum(high_mask[mask]))
        high_ratio = high_count / size if size else 0.0
        enrichment = high_ratio / global_ratio if global_ratio > 0 else 0.0
        p_value = hypergeom_sf(total, success_total, size, high_count)
        rows.append(
            {
                "cluster": int(cluster_id),
                "size": size,
                "high_count": high_count,
                "high_ratio": high_ratio,
                "global_high_ratio": global_ratio,
                "enrichment": enrichment,
                "hypergeom_p": p_value,
                "minus_log10_p": -math.log10(max(p_value, 1e-300)),
                "mean_hyperedge_avgsize": float(np.mean(features[mask, 0])),
                "mean_higher_order_distance": float(np.mean(features[mask, 1])),
                "mean_csi_score": float(np.mean(csi_scores[mask])),
            }
        )
    rows.sort(key=lambda item: (item["high_ratio"], item["high_count"]), reverse=True)
    return rows, high_mask


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as file:
        file.write(",".join(keys) + "\n")
        for row in rows:
            file.write(",".join(str(row[key]) for key in keys) + "\n")


def save_figure(fig, path):
    try:
        fig.savefig(path, format="pdf", bbox_inches="tight")
        return path
    except PermissionError:
        for index in range(1, 100):
            fallback_path = path.with_name(f"{path.stem}_new{index}{path.suffix}")
            if fallback_path.exists():
                continue
            fig.savefig(fallback_path, format="pdf", bbox_inches="tight")
            print(f"Permission denied for {path}; saved {fallback_path} instead.")
            return fallback_path
        raise


def plot_clusters(path, features, labels, high_mask, title):
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    scatter = ax.scatter(
        features[:, 0],
        features[:, 1],
        c=labels,
        cmap="tab10",
        s=34,
        alpha=0.78,
        edgecolors="none",
    )
    if np.any(high_mask):
        ax.scatter(
            features[high_mask, 0],
            features[high_mask, 1],
            marker="x",
            c="black",
            s=70,
            linewidths=1.8,
            label="High-influence nodes",
        )
    ax.set_xlabel("Node hyperedge avg size")
    ax.set_ylabel("Avg higher-order distance to rumor seeds")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def compute_tsne_embedding(scaled_features, perplexity, seed):
    effective_perplexity = min(perplexity, max(2, (len(scaled_features) - 1) // 3))
    kwargs = {
        "n_components": 2,
        "perplexity": effective_perplexity,
        "learning_rate": 200.0,
        "init": "random",
        "random_state": seed,
    }
    try:
        return TSNE(max_iter=1000, **kwargs).fit_transform(scaled_features)
    except TypeError:
        return TSNE(n_iter=1000, **kwargs).fit_transform(scaled_features)


def plot_tsne_highlight(path, embedding, high_mask, title):
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.scatter(
        embedding[~high_mask, 0],
        embedding[~high_mask, 1],
        c="#8f9aa7",
        s=32,
        alpha=0.62,
        edgecolors="none",
        label="Candidate nodes",
    )
    if np.any(high_mask):
        ax.scatter(
            embedding[high_mask, 0],
            embedding[high_mask, 1],
            c="#d62728",
            marker="*",
            s=120,
            alpha=0.95,
            edgecolors="black",
            linewidths=0.6,
            label="High-influence nodes",
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def plot_tsne_ranked(path, embedding, high_mask, csi_scores, title):
    score_values = np.log1p(np.maximum(csi_scores, 0))
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=score_values,
        cmap="viridis",
        s=36,
        alpha=0.76,
        edgecolors="none",
    )
    if np.any(high_mask):
        ax.scatter(
            embedding[high_mask, 0],
            embedding[high_mask, 1],
            facecolors="none",
            edgecolors="#d62728",
            marker="o",
            s=150,
            linewidths=1.8,
            label="High-influence nodes",
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.colorbar(scatter, ax=ax, label="log(1 + CSI score)")
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def plot_tsne_clusters(path, embedding, labels, high_mask, title):
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="tab10",
        s=34,
        alpha=0.76,
        edgecolors="none",
    )
    if np.any(high_mask):
        ax.scatter(
            embedding[high_mask, 0],
            embedding[high_mask, 1],
            marker="x",
            c="black",
            s=78,
            linewidths=1.8,
            label="High-influence nodes",
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def run_experiment(args):
    RESULT_DIR.mkdir(exist_ok=True)
    graph_matrix, node_list = load_graph(args.data)
    feature_payload = compute_csi_features(
        graph_matrix,
        node_list,
        args.data,
        args.max_order,
        args.alpha,
        args.candidate_ratio,
    )
    candidate_pos = feature_payload["candidate_pos"]
    features = feature_payload["features"]
    tsne_features = feature_payload["tsne_features"] if args.tsne_feature_set == "multiorder" else features
    csi_scores = feature_payload["csi_score"]
    scaled_features = StandardScaler().fit_transform(features)
    tsne_embedding = None
    if args.tsne:
        print(f"Computing t-SNE embedding with {args.tsne_feature_set} features...")
        scaled_tsne_features = StandardScaler().fit_transform(np.log1p(np.maximum(tsne_features, 0)))
        tsne_embedding = compute_tsne_embedding(scaled_tsne_features, args.tsne_perplexity, args.seed)

    all_rows = []
    metadata = {
        "data": args.data,
        "candidate_ratio": args.candidate_ratio,
        "candidate_count": int(len(candidate_pos)),
        "budget": int(feature_payload["budget"]),
        "max_order": args.max_order,
        "alpha": args.alpha,
        "betas": [beta_label(beta) for beta in args.betas],
        "cluster_counts": args.clusters,
            "methods": args.methods,
            "tsne_feature_set": args.tsne_feature_set,
        }

    for beta in args.betas:
        high_pos, high_file = load_high_influence_positions(
            args.data,
            beta,
            args.candidate_ratio,
            node_list,
            feature_payload["budget"],
        )
        high_in_candidates = len(set(candidate_pos).intersection(high_pos))
        print(
            f"beta={beta_label(beta)}: high nodes in candidates "
            f"{high_in_candidates}/{feature_payload['budget']} from {high_file}"
        )
        beta_high_mask = np.array([pos in high_pos for pos in candidate_pos], dtype=bool)
        if tsne_embedding is not None:
            tsne_prefix = f"{args.data}_beta_{beta_label(beta)}_tsne_{args.tsne_feature_set}"
            plot_tsne_highlight(
                RESULT_DIR / f"{tsne_prefix}_highlight.pdf",
                tsne_embedding,
                beta_high_mask,
                f"{args.data}, beta={beta_label(beta)}, t-SNE high-influence view ({args.tsne_feature_set})",
            )
            plot_tsne_ranked(
                RESULT_DIR / f"{tsne_prefix}_csi_score.pdf",
                tsne_embedding,
                beta_high_mask,
                csi_scores,
                f"{args.data}, beta={beta_label(beta)}, t-SNE CSI score view ({args.tsne_feature_set})",
            )

        for method in args.methods:
            for cluster_count in args.clusters:
                if cluster_count < 2 or cluster_count >= len(candidate_pos):
                    continue
                labels, inertia, bic = fit_clusters(method, scaled_features, cluster_count, args.seed)
                silhouette = float(silhouette_score(scaled_features, labels))
                rows, high_mask = summarize_clusters(labels, features, csi_scores, candidate_pos, high_pos)

                for row in rows:
                    row.update(
                        {
                            "data": args.data,
                            "beta": beta_label(beta),
                            "method": method,
                            "k": cluster_count,
                            "silhouette": silhouette,
                            "inertia": "" if inertia is None else inertia,
                            "bic": "" if bic is None else bic,
                        }
                    )
                all_rows.extend(rows)

                prefix = f"{args.data}_beta_{beta_label(beta)}_{method}_k{cluster_count}"
                write_csv(RESULT_DIR / f"{prefix}_clusters.csv", rows)
                plot_clusters(
                    RESULT_DIR / f"{prefix}_scatter.pdf",
                    features,
                    labels,
                    high_mask,
                    f"{args.data}, beta={beta_label(beta)}, {method.upper()} k={cluster_count}",
                )
                if tsne_embedding is not None:
                    plot_tsne_clusters(
                        RESULT_DIR / f"{prefix}_tsne_{args.tsne_feature_set}_clusters.pdf",
                        tsne_embedding,
                        labels,
                        high_mask,
                        f"{args.data}, beta={beta_label(beta)}, t-SNE {method.upper()} k={cluster_count} ({args.tsne_feature_set})",
                    )

    summary_csv = RESULT_DIR / f"{args.data}_cluster_enrichment_summary.csv"
    write_csv(summary_csv, all_rows)
    json_path = RESULT_DIR / f"{args.data}_cluster_enrichment_summary.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump({"metadata": metadata, "rows": all_rows}, file, indent=4)

    best_rows = sorted(all_rows, key=lambda item: (item["enrichment"], item["high_ratio"]), reverse=True)[:12]
    print("\nTop enriched clusters:")
    for row in best_rows:
        print(
            f"  beta={row['beta']} {row['method']} k={row['k']} cluster={row['cluster']}: "
            f"high={row['high_count']}/{row['size']} "
            f"ratio={row['high_ratio']:.3f}, enrich={row['enrichment']:.2f}, "
            f"p={row['hypergeom_p']:.3g}"
        )
    print(f"\nSaved summary CSV: {summary_csv}")
    print(f"Saved summary JSON: {json_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster candidate nodes in the 2D CSI feature space.")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--betas", type=float, nargs="+", default=DEFAULT_BETAS)
    parser.add_argument("--candidate-ratio", type=float, default=0.2)
    parser.add_argument("--max-order", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--clusters", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--methods", nargs="+", default=["kmeans", "gmm"], choices=["kmeans", "gmm"])
    parser.add_argument("--tsne", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-feature-set", choices=["multiorder", "csi2"], default="multiorder")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
