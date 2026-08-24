import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


os.chdir(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = Path('./New_pic')
OUTPUT_DIR.mkdir(exist_ok=True)


def unique_plot_path(filename):
    candidate = OUTPUT_DIR / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    for index in range(1, 100):
        rerun_candidate = OUTPUT_DIR / f"{stem}_rerun{index}{suffix}"
        if not rerun_candidate.exists():
            return rerun_candidate
    return OUTPUT_DIR / f"{stem}_rerun{suffix}"

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'


dataset = [
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

algorithm_extra_para_dic = {
    # Full HCDPQ: HCDP candidate selection + DQN policy.
    'indicator_selection_RL_11': ['_(ratio_0.2)'],
    # HCDP-only ablation.
    'indicator_selection': [''],
    # DQN-only ablation.
    'indicator_selection_RL_2': ['_(ratio_1)'],
}

label_name_dic = {
    'indicator_selection_RL_11': 'HCDP-DQN',
    'indicator_selection': 'HCDP',
    'indicator_selection_RL_2': 'DQN',
}

plot_order = [
    'indicator_selection_RL_11',
    'indicator_selection',
    'indicator_selection_RL_2',
]

color_map = {
    'indicator_selection_RL_11': '#df8a80',
    'indicator_selection': '#9fd3df',
    'indicator_selection_RL_2': '#f2b179',
}

marker_map = {
    'indicator_selection_RL_11': 'o',
    'indicator_selection': 'o',
    'indicator_selection_RL_2': 'o',
}

beta = 0.05
T = 30


def load_algorithm_result(data, t_max, beta_value):
    algorithm_result_dic = {}
    for algorithm_name, extra_para in algorithm_extra_para_dic.items():
        algorithm_result_dic[algorithm_name] = []
        for para in extra_para:
            file_path = (
                f'./New_result/result_{algorithm_name}_({data})_'
                f'(T_{t_max})_(beta_{beta_value}){para}.npy'
            )
            try:
                algorithm_result_dic[algorithm_name].append(
                    np.load(file_path, allow_pickle=True).tolist()
                )
            except FileNotFoundError:
                print(f"Warning: missing result file: {file_path}")
                algorithm_result_dic[algorithm_name].append([])
    return algorithm_result_dic


def collect_final_scores():
    final_scores = {}
    for data in dataset:
        algorithm_result = load_algorithm_result(data, T, beta)
        final_scores[data] = {}
        for name in plot_order:
            values_group = algorithm_result.get(name, [])
            values = values_group[0] if values_group else []
            final_scores[data][label_name_dic[name]] = values[-1] if values else np.nan
    return final_scores


def save_summary_csv(final_scores):
    csv_path = f'./New_pic/Ablation_summary_final_{beta}.csv'
    with open(csv_path, 'w', encoding='utf-8') as file:
        file.write('dataset,HCDP-DQN,HCDP,DQN,HCDP_relative,DQN_relative\n')
        for data in dataset:
            HCDPq = final_scores[data].get('HCDP-DQN', np.nan)
            HCDP = final_scores[data].get('HCDP', np.nan)
            dqn = final_scores[data].get('DQN', np.nan)
            HCDP_relative = HCDP / HCDPq if HCDPq and not np.isnan(HCDPq) else np.nan
            dqn_relative = dqn / HCDPq if HCDPq and not np.isnan(HCDPq) else np.nan
            file.write(
                f'{data},{HCDPq},{HCDP},{dqn},{HCDP_relative},{dqn_relative}\n'
            )
    return csv_path


def plot_summary(final_scores):
    x_values = np.arange(len(dataset))
    width = 0.24
    methods = ['HCDP-DQN', 'HCDP', 'DQN']
    offsets = [-width, 0, width]
    summary_color_map = {
        'HCDP-DQN': '#df8a80',
        'HCDP': '#9fd3df',
        'DQN': '#f2b179',
    }
    summary_values = {method: [] for method in methods}

    for data in dataset:
        HCDPq = final_scores[data].get('HCDP-DQN', np.nan)
        for method in methods:
            value = final_scores[data].get(method, np.nan)
            relative_value = value / HCDPq if HCDPq and not np.isnan(HCDPq) else np.nan
            summary_values[method].append(relative_value)

    fig, axis = plt.subplots(figsize=(7.2, 3.2))
    for method, offset in zip(methods, offsets):
        axis.bar(
            x_values + offset,
            summary_values[method],
            width,
            label=method,
            color=summary_color_map[method],
            edgecolor='white',
            linewidth=0.8,
            zorder=3 if method == 'HCDP-DQN' else 1,
        )

    axis.axhline(1.0, color='#333333', linestyle='--', linewidth=1.5, alpha=0.7)
    axis.set_ylabel('Influence Ratio to HCDP-DQN', fontsize=10)
    axis.set_xticks(x_values)
    axis.set_xticklabels(dataset, rotation=35, ha='right', fontsize=8.5)
    axis.tick_params(axis='both', labelsize=8.5, direction='in')
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)
    axis.legend(
        loc='upper center',
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.16),
        prop={'size': 8.5},
        handlelength=1.5,
        columnspacing=1.2,
    )
    fig.tight_layout()

    plot_path = OUTPUT_DIR / f'Ablation_summary_relative_{beta}.pdf'
    fig.savefig(plot_path, dpi=600, bbox_inches='tight', format='pdf')
    plt.close(fig)
    return plot_path


def plot_peak_summary(final_scores):
    methods = ['HCDP-DQN', 'HCDP', 'DQN']
    peak_color_map = {
        'HCDP-DQN': '#df8a80',
        'HCDP': '#9fd3df',
        'DQN': '#f2b179',
    }
    x_values = np.arange(len(dataset))
    peak_width = 0.44

    fig, axis = plt.subplots(figsize=(7.2, 3.4))

    for method in methods:
        values = [final_scores[data].get(method, np.nan) for data in dataset]
        for x_pos, value in zip(x_values, values):
            if np.isnan(value):
                continue
            axis.fill(
                [x_pos - peak_width, x_pos, x_pos + peak_width],
                [0, value, 0],
                color=peak_color_map[method],
                alpha=0.72,
                linewidth=0,
                label=method if x_pos == x_values[0] else None,
            )
            axis.plot(
                [x_pos - peak_width, x_pos, x_pos + peak_width],
                [0, value, 0],
                color=peak_color_map[method],
                linewidth=1.5,
                alpha=0.95,
            )

    max_value = max(
        value
        for data in dataset
        for value in final_scores[data].values()
        if not np.isnan(value)
    )
    label_offset = max_value * 0.025
    for x_pos, data in zip(x_values, dataset):
        HCDPq_value = final_scores[data].get('HCDP-DQN', np.nan)
        if not np.isnan(HCDPq_value):
            axis.text(
                x_pos,
                HCDPq_value + label_offset,
                f'{HCDPq_value:.0f}',
                ha='center',
                va='bottom',
                fontsize=7.5,
                fontweight='bold',
            )

    axis.set_xlim(-0.55, len(dataset) - 0.45)
    axis.set_ylim(0, max_value * 1.14)
    axis.set_ylabel('Influence', fontsize=10)
    axis.set_xticks(x_values)
    axis.set_xticklabels(dataset, rotation=25, ha='right', fontsize=8.5)
    axis.tick_params(axis='both', labelsize=8.5, direction='in')
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)
    axis.legend(
        loc='upper center',
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.17),
        prop={'size': 8.5},
        handlelength=1.5,
        columnspacing=1.2,
    )
    fig.tight_layout()

    plot_path = OUTPUT_DIR / f'Ablation_summary_peak_{beta}.pdf'
    fig.savefig(plot_path, dpi=600, bbox_inches='tight', format='pdf')
    plt.close(fig)
    return plot_path


def main():
    labels = [f'({chr(97 + i)})' for i in range(len(dataset))]
    fig = plt.figure(figsize=(7.2, 7.4))

    for pic_index, data in enumerate(dataset, start=1):
        algorithm_result = load_algorithm_result(data, T, beta)
        axis = fig.add_subplot(4, 4, pic_index)

        for name in plot_order:
            data_group = algorithm_result.get(name, [])
            is_core = name == 'indicator_selection_RL_11'
            zorder = 3 if is_core else 1
            for values in data_group:
                if not values:
                    continue
                seed_counts = list(range(1, len(values) + 1))
                axis.plot(
                    seed_counts,
                    values,
                    label=label_name_dic[name],
                    color=color_map[name],
                    linestyle='-',
                    linewidth=1.5,
                    marker=marker_map[name],
                    markersize=5.2,
                    markerfacecolor=color_map[name],
                    markeredgecolor=color_map[name],
                    alpha=0.95,
                    zorder=zorder,
                )

        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.tick_params(axis='both', labelsize=8.5, direction='in')
        axis.set_xlabel(r'$b$', fontsize=10)
        axis.set_ylabel('Influence', fontsize=10)
        axis.text(
            0.08,
            0.86,
            labels[pic_index - 1],
            fontsize=11.5,
            fontweight='bold',
            transform=axis.transAxes,
        )
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)

    handles, legend_labels = fig.axes[0].get_legend_handles_labels()
    by_label = dict(zip(legend_labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc='upper center',
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        prop={'size': 8.5},
        handlelength=1.5,
        columnspacing=1.2,
    )

    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.07, top=0.93, wspace=0.45, hspace=0.65)

    plot_path = unique_plot_path(f'Ablation_result_{beta}.pdf')
    plt.savefig(plot_path, dpi=600, bbox_inches='tight', format='pdf')
    print(f"Saved ablation figure: {plot_path}")
    plt.close(fig)

    final_scores = collect_final_scores()
    csv_path = save_summary_csv(final_scores)
    summary_path = plot_summary(final_scores)
    peak_summary_path = plot_peak_summary(final_scores)
    print(f"Saved ablation summary CSV: {csv_path}")
    print(f"Saved ablation summary figure: {summary_path}")
    print(f"Saved ablation peak summary figure: {peak_summary_path}")


if __name__ == "__main__":
    main()
