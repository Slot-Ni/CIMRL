import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_NAME = "synthetic_HCL_v1000_e1000_exp2.5_maxdeg10"
BETAS = ["0.01", "0.02", "0.03", "0.04", "0.05", "0.06"]

CSIQ_JSON = SCRIPT_DIR / "Beta_Ablation_CSIQ" / f"{DATA_NAME}_beta_ablation_plot_data.json"
S2V_OPT_JSON = (
    SCRIPT_DIR
    / "HCL1000_Beta_Method_Comparison_S2VOpt"
    / f"{DATA_NAME}_methods_beta_comparison_with_S2V-DQN-Opt.json"
)
OUT_DIR = SCRIPT_DIR / "Beta_Ablation_CSIQ"


STYLE = {
    "CSIQ": {"color": "#df8a80", "marker": "o"},
    "CSI-Greedy": {"color": "#9fd3df", "marker": "s"},
    "S2V-DQN": {"color": "#f2b179", "marker": "^"},
    "HADP": {"color": "#8bc487", "marker": "D"},
    "HDegree": {"color": "#b7a5d8", "marker": "v"},
    "CI": {"color": "#9db9dc", "marker": "P"},
}
METHOD_ORDER = ["CSI-Greedy", "HDegree", "HADP", "CI", "S2V-DQN", "CSIQ"]
DISPLAY_NAME = {
    "CSIQ": "HCDP-DQN",
    "CSI-Greedy": "HCDP-Greedy",
    "S2V-DQN": "S2V-DQN",
    "HADP": "HADP",
    "HDegree": "HDegree",
    "CI": "CI",
}
PLOT_LINE_WIDTH = 1.5


def load_results():
    if not CSIQ_JSON.exists():
        raise FileNotFoundError(f"Missing CSIQ beta ablation JSON: {CSIQ_JSON}")
    if not S2V_OPT_JSON.exists():
        raise FileNotFoundError(f"Missing optimized S2V JSON: {S2V_OPT_JSON}")

    with CSIQ_JSON.open("r", encoding="utf-8") as file:
        csiq_payload = json.load(file)
    with S2V_OPT_JSON.open("r", encoding="utf-8") as file:
        s2v_payload = json.load(file)

    csiq_results = dict(csiq_payload["results"])
    csiq_fallback = s2v_payload["results"].get("CSIQ", {})
    for beta in BETAS:
        if beta not in csiq_results and beta in csiq_fallback:
            csiq_results[beta] = csiq_fallback[beta]

    results = {
        "CSIQ": csiq_results,
        "HDegree": s2v_payload["results"].get("HDegree", {}),
        "HADP": s2v_payload["results"].get("HADP", {}),
        "CI": s2v_payload["results"].get("CI", {}),
        "S2V-DQN": s2v_payload["results"].get("S2V-DQN-Opt", {}),
        "CSI-Greedy": s2v_payload["results"].get("CSI-Greedy", {}),
    }
    return results


def save_csv(results):
    csv_path = OUT_DIR / f"{DATA_NAME}_beta_ablation_with_s2vopt_final.csv"
    with csv_path.open("w", encoding="utf-8") as file:
        file.write("beta," + ",".join(METHOD_ORDER) + "\n")
        for beta in BETAS:
            row = [beta]
            for method in METHOD_ORDER:
                values = results.get(method, {}).get(beta)
                row.append(str(values[-1]) if values else "")
            file.write(",".join(row) + "\n")
    return csv_path


def plot_budget_grid(results):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9))
    axes = axes.flatten()

    for index, beta in enumerate(BETAS):
        ax = axes[index]
        for method in METHOD_ORDER:
            values = results.get(method, {}).get(beta)
            if not values:
                continue
            style = STYLE[method]
            ax.plot(
                range(1, len(values) + 1),
                values,
                label=DISPLAY_NAME.get(method, method),
                color=style["color"],
                linestyle="-",
                marker=style["marker"],
                linewidth=PLOT_LINE_WIDTH,
                markersize=5.2,
                markerfacecolor="None",
                markeredgecolor=style["color"],
                markeredgewidth=0.8,
                alpha=0.95,
                zorder=3 if method == "CSIQ" else 1,
            )

        ax.text(
            0.08,
            0.86,
            f"({chr(97 + index)})",
            transform=ax.transAxes,
            fontsize=11.5,
            fontweight="bold",
        )
        ax.text(
            0.66,
            0.08,
            rf"$\beta={beta}$",
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight=300,
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis="both", labelsize=8.5, direction="in")
        ax.set_xlabel("b", fontsize=10)
        ax.set_ylabel("Influence", fontsize=10)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        prop={"size": 8.5},
        handlelength=1.5,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.90, wspace=0.36, hspace=0.46)

    pdf_path = OUT_DIR / f"{DATA_NAME}_beta_ablation_budget_grid_with_s2v.pdf"
    fig.savefig(pdf_path, dpi=600, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def main():
    results = load_results()
    csv_path = save_csv(results)
    grid_path = plot_budget_grid(results)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved beta ablation grid: {grid_path}")


if __name__ == "__main__":
    main()
