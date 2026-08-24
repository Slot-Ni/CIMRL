from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MATRIX_DIR = PROJECT_ROOT / "HyperGraphMatrix" / "HyperGraphMatrix"
OUT_DIR = SCRIPT_DIR / "New_pic"
OUT_DIR.mkdir(exist_ok=True)

DATASETS = [
    "Algebra",
    "Restaurants-Rev",
    "Geometry",
    "Music-Rev",
    "NDC-classes",
    "Bars-Rev",
    "iAF1260b",
    "iJO1366",
    "soc-hamsterster",
]

HCL_DATASETS = [
    "synthetic_HCL_v1000_e1000_exp2.0_maxdeg10",
    "synthetic_HCL_v1000_e1000_exp2.5_maxdeg10",
    "synthetic_HCL_v1000_e1000_exp3.0_maxdeg10",
    "synthetic_HCL_v2000_e2000_exp2.0_maxdeg10",
    "synthetic_HCL_v2000_e2000_exp2.5_maxdeg10",
    "synthetic_HCL_v2000_e2000_exp3.0_maxdeg10",
    "synthetic_HCL_v3000_e3000_exp2.0_maxdeg10",
    "synthetic_HCL_v3000_e3000_exp2.5_maxdeg10",
    "synthetic_HCL_v3000_e3000_exp3.0_maxdeg10",
]


def projected_degrees(matrix):
    degrees = np.zeros(matrix.shape[0], dtype=np.int64)
    for node in range(matrix.shape[0]):
        hyperedges = np.flatnonzero(matrix[node])
        if len(hyperedges) == 0:
            continue
        neighbors = np.flatnonzero(matrix[:, hyperedges].sum(axis=1))
        degrees[node] = max(len(neighbors) - 1, 0)
    return degrees


def stats_for_dataset(data_name, matrix_dir=MATRIX_DIR):
    matrix_path = matrix_dir / f"{data_name}.npy"
    if not matrix_path.exists():
        raise FileNotFoundError(matrix_path)

    matrix = np.load(matrix_path, allow_pickle=True, mmap_mode="r")
    hyperedge_sizes = np.asarray(matrix.sum(axis=0), dtype=float).ravel()
    hyperdegrees = np.asarray(matrix.sum(axis=1), dtype=float).ravel()
    degrees = projected_degrees(matrix)

    mean_hyperdegree = hyperdegrees.mean()
    cv_hyperdegree = (
        hyperdegrees.std(ddof=0) / mean_hyperdegree
        if mean_hyperdegree != 0
        else np.nan
    )

    return {
        "Networks": data_name,
        "n": matrix.shape[0],
        "m": matrix.shape[1],
        "avg_kE": hyperedge_sizes.mean(),
        "avg_kH": mean_hyperdegree,
        "avg_k": degrees.mean(),
        "cv_kH": cv_hyperdegree,
    }


def write_csv(rows):
    path = OUT_DIR / "real_network_hyperdegree_cv.csv"
    with path.open("w", encoding="utf-8") as file:
        file.write("Networks,n,m,avg_kE,avg_kH,avg_k,cv_kH\n")
        for row in rows:
            file.write(
                "{Networks},{n},{m},{avg_kE:.4f},{avg_kH:.4f},{avg_k:.4f},{cv_kH:.4f}\n".format(
                    **row
                )
            )
    return path


def write_named_csv(rows, filename):
    path = OUT_DIR / filename
    with path.open("w", encoding="utf-8") as file:
        file.write("Networks,n,m,avg_kE,avg_kH,avg_k,cv_kH\n")
        for row in rows:
            file.write(
                "{Networks},{n},{m},{avg_kE:.4f},{avg_kH:.4f},{avg_k:.4f},{cv_kH:.4f}\n".format(
                    **row
                )
            )
    return path


def write_latex(rows):
    path = OUT_DIR / "real_network_hyperdegree_cv_table_rows.tex"
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                "{Networks} & {n} & {m} & {avg_kE:.4f} & {avg_kH:.4f} & {avg_k:.4f} & {cv_kH:.4f} \\\\\n".format(
                    **row
                )
            )
    return path


def write_named_latex(rows, filename):
    path = OUT_DIR / filename
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                "{Networks} & {n} & {m} & {avg_kE:.4f} & {avg_kH:.4f} & {avg_k:.4f} & {cv_kH:.4f} \\\\\n".format(
                    **row
                )
            )
    return path


def main():
    rows = []
    for data_name in DATASETS:
        row = stats_for_dataset(data_name)
        rows.append(row)
        print(
            "{Networks}: n={n}, m={m}, avg_kE={avg_kE:.4f}, avg_kH={avg_kH:.4f}, avg_k={avg_k:.4f}, cv_kH={cv_kH:.4f}".format(
                **row
            )
        )

    csv_path = write_csv(rows)
    latex_path = write_latex(rows)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved LaTeX rows: {latex_path}")

    hcl_rows = []
    for data_name in HCL_DATASETS:
        row = stats_for_dataset(data_name, PROJECT_ROOT / "HyperGraphMatrix")
        hcl_rows.append(row)
        print(
            "{Networks}: n={n}, m={m}, avg_kE={avg_kE:.4f}, avg_kH={avg_kH:.4f}, avg_k={avg_k:.4f}, cv_kH={cv_kH:.4f}".format(
                **row
            )
        )

    hcl_csv_path = write_named_csv(hcl_rows, "hcl_network_hyperdegree_cv.csv")
    hcl_latex_path = write_named_latex(
        hcl_rows, "hcl_network_hyperdegree_cv_table_rows.tex"
    )
    print(f"Saved HCL CSV: {hcl_csv_path}")
    print(f"Saved HCL LaTeX rows: {hcl_latex_path}")


if __name__ == "__main__":
    main()
