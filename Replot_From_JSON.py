import os
import json
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

STYLE_CONFIG = {
    'CSI-Greedy': {'color': '#ff7f0e', 'marker': 's', 'linewidth': 1.5, 'markersize': 7, 'zorder': 5},
    'HDegree': {'color': '#EC6E66', 'marker': 'd', 'linewidth': 1.5, 'markersize': 7, 'zorder': 2},
    'HADP': {'color': '#FAC7B3', 'marker': '^', 'linewidth': 1.5, 'markersize': 7, 'zorder': 2},
    'CI': {'color': '#91CCC0', 'marker': '*', 'linewidth': 1.5, 'markersize': 7, 'zorder': 2},
    'S2V-DQN': {'color': '#2ca02c', 'marker': 'v', 'linewidth': 1.5, 'markersize': 7, 'zorder': 5},
    'CSIQ': {'color': '#1f77b4', 'marker': '8', 'linewidth': 1.5, 'markersize': 10, 'zorder': 10},
}

PLOT_ORDER = ['CSI-Greedy', 'HDegree', 'HADP', 'CI', 'S2V-DQN', 'CSIQ']
DISPLAY_NAME = {
    'CSI-Greedy': 'HCDP-Greedy',
    'HDegree': 'HDgree',
    'HADP': 'HADP',
    'CI': 'CI',
    'S2V-DQN': 'S2V-DQN',
    'CSIQ': 'HCDP-DQN',
}


def replot_single_json(json_filepath, output_pdf_path, title='HCL Expected Influence'):
    """Read one plot-data JSON file and render a PDF figure."""
    if not os.path.exists(json_filepath):
        print(f'Missing file: {json_filepath}')
        return

    with open(json_filepath, 'r', encoding='utf-8') as file:
        data_results = json.load(file)

    fig, ax = plt.subplots(figsize=(6, 5))

    for algo_name in PLOT_ORDER:
        if algo_name not in data_results:
            continue

        y_values = data_results[algo_name]
        x_values = range(1, len(y_values) + 1)
        style = STYLE_CONFIG.get(
            algo_name,
            {'color': 'gray', 'marker': 'o', 'linewidth': 1.5, 'markersize': 6, 'zorder': 1},
        )

        ax.plot(
            x_values,
            y_values,
            label=DISPLAY_NAME.get(algo_name, algo_name),
            color=style['color'],
            marker=style['marker'],
            linewidth=style['linewidth'],
            markersize=style['markersize'],
            zorder=style['zorder'],
        )

    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel('Budget (b)', fontsize=13)
    ax.set_ylabel('Target Expected Influence', fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', fontsize=11, framealpha=0.95, edgecolor='black')

    plt.tight_layout()
    plt.savefig(output_pdf_path, format='pdf', bbox_inches='tight')
    print(f'Saved replot: {output_pdf_path}')
    plt.close()


if __name__ == '__main__':
    result_dir = './HCL_Final_Comparison'
    found_files = False

    for filename in os.listdir(result_dir):
        if not filename.endswith('_plot_data.json'):
            continue

        found_files = True
        json_path = os.path.join(result_dir, filename)
        parts = filename.split('_')
        title = f'HCL ({parts[2]} {parts[3]})' if len(parts) >= 4 else 'HCL Comparison'
        output_name = filename.replace('_plot_data.json', '_Replot.pdf')
        output_path = os.path.join(result_dir, output_name)
        replot_single_json(json_path, output_path, title=title)

    if not found_files:
        print(f'No *_plot_data.json files found in {result_dir}')
