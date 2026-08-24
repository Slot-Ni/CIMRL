import json
import os
import math

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ============================================================
# Global figure settings
# ============================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'


# ============================================================
# Paths
# ============================================================
BASE_DIR = "./HCL_synthetic_comparison"


EXPECTED_FILES = [
    "synthetic_HCL_v1000_e1000_exp2.0_maxdeg10_plot_data.json",
    "synthetic_HCL_v1000_e1000_exp2.5_maxdeg10_plot_data.json",
    "synthetic_HCL_v1000_e1000_exp3.0_maxdeg10_plot_data.json",

    "synthetic_HCL_v2000_e2000_exp2.0_maxdeg10_plot_data.json",
    "synthetic_HCL_v2000_e2000_exp2.5_maxdeg10_plot_data.json",
    "synthetic_HCL_v2000_e2000_exp3.0_maxdeg10_plot_data.json",

    "synthetic_HCL_v3000_e3000_exp2.0_maxdeg10_plot_data.json",
    "synthetic_HCL_v3000_e3000_exp2.5_maxdeg10_plot_data.json",
    "synthetic_HCL_v3000_e3000_exp3.0_maxdeg10_plot_data.json",
]


# ============================================================
# Plot order
# ============================================================
PLOT_ORDER = [
    'CSI-Greedy',
    'HDegree',
    'HADP',
    'CI',
    'S2V-DQN',
    'CSIQ'
]


# ============================================================
# Labels
# ============================================================
LABEL_NAME = {
    'CSIQ': 'HCDP-DQN',
    'CSI-Greedy': 'HCDP-Greedy',
    'S2V-DQN': 'S2V-DQN',
    'HADP': 'HADP',
    'HDegree': 'HDegree',
    'CI': 'CI',
}


# ============================================================
# Colors
# ============================================================
COLOR_MAP = {
    'CSIQ': '#df8a80',
    'CSI-Greedy': '#9fd3df',
    'S2V-DQN': '#f2b179',
    'HADP': '#8bc487',
    'HDegree': '#b7a5d8',
    'CI': '#9db9dc',
}


# ============================================================
# Markers
# ============================================================
MARKER_MAP = {
    'CSIQ': 'o',
    'CSI-Greedy': 's',
    'S2V-DQN': '^',
    'HADP': 'D',
    'HDegree': 'v',
    'CI': 'P',
}


# ============================================================
# Line width
# ============================================================
PLOT_LINE_WIDTH = 1.5


# ============================================================
# Algorithm aliases
# ============================================================
ALGO_NAME_ALIASES = {
    'ComInf-DQN': 'CSIQ',
    'ComInf-Greedy': 'CSI-Greedy',
    'CSI-greedy': 'CSI-Greedy',
    'indicator_selection_RL_11': 'CSIQ',
    'indicator_selection_greedy': 'CSI-Greedy',
    'S2V_DQN_RL': 'S2V-DQN',
}


# ============================================================
# Normalize algorithm names
# ============================================================
def normalize_result_keys(result):

    normalized = {}

    for name, values in result.items():

        normalized_name = ALGO_NAME_ALIASES.get(
            name,
            name
        )

        normalized[normalized_name] = values

    return normalized


# ============================================================
# Load JSON result
# ============================================================
def load_result(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        result = json.load(file)

    return normalize_result_keys(result)


# ============================================================
# Generate clean x-axis
#
# Important:
#   Same row -> same x-axis range
#   Same row -> same x-axis ticks
# ============================================================
def get_uniform_xticks(max_b):
    """
    横坐标设计原则：
    1. 从 0 开始
    2. 横坐标终点严格等于真实 max_b
    3. 不额外扩展横坐标范围
    4. 尽量使用等间距整数刻度
    5. 刻度数量大约控制在 5~6 个
    """

    if max_b <= 5:
        # 例如：
        # max_b = 4 -> 0,1,2,3,4
        # max_b = 5 -> 0,1,2,3,4,5
        step = 1

    elif max_b <= 10:
        # max_b = 10 -> 0,2,4,6,8,10
        step = 2

    elif max_b <= 15:
        # 例如 max_b = 12 -> 0,3,6,9,12
        step = 3

    elif max_b <= 20:
        # max_b = 16 -> 0,4,8,12,16
        # max_b = 20 -> 0,4,8,12,16,20
        step = 4

    elif max_b <= 25:
        # max_b = 24 -> 0,5,10,15,20
        step = 5

    elif max_b <= 30:
        step = 6

    else:
        # 更大范围自动取大约 5 个区间
        step = max(1, round(max_b / 5))

    # 等间距整数刻度
    ticks = list(range(0, max_b + 1, step))

    # 注意：
    # 不强行把 max_b 加入 ticks，
    # 否则可能导致最后一段刻度不等距。
    #
    # 横坐标本身仍然截止到 max_b。
    return ticks, max_b


# ============================================================
# Preload all results
#
# We first load all files so that we can determine the
# maximum b of each ROW before plotting.
# ============================================================
all_results = []

for file_name in EXPECTED_FILES:

    file_path = os.path.join(
        BASE_DIR,
        file_name
    )

    if not os.path.exists(file_path):

        print(
            f"Warning: file not found: {file_path}"
        )

        all_results.append(None)

        continue

    result = load_result(
        file_path
    )

    all_results.append(
        result
    )


# ============================================================
# Determine maximum b for each row
#
# Row 1: (a), (b), (c)
# Row 2: (d), (e), (f)
# Row 3: (g), (h), (i)
#
# This is the KEY modification:
# three figures in the same row share exactly the same
# x-axis range and tick locations.
# ============================================================
row_max_b = []


for row_index in range(3):

    row_start = row_index * 3
    row_end = row_start + 3

    current_row_results = all_results[
        row_start:row_end
    ]

    max_b = 0

    for result in current_row_results:

        if result is None:
            continue

        for algo in PLOT_ORDER:

            if algo not in result:
                continue

            current_b = len(
                result[algo]
            )

            max_b = max(
                max_b,
                current_b
            )

    row_max_b.append(
        max_b
    )


# ============================================================
# Print row x-axis settings
# ============================================================
print("=" * 60)
print("Row-wise x-axis settings")
print("=" * 60)

for row_index, max_b in enumerate(row_max_b):

    ticks, x_max = get_uniform_xticks(
        max_b
    )

    print(
        f"Row {row_index + 1}: "
        f"max_b = {max_b}, "
        f"x_max = {x_max}, "
        f"ticks = {ticks}"
    )

print("=" * 60)


# ============================================================
# Create figure
# ============================================================
fig = plt.figure(
    figsize=(7.2, 7.4)
)


subplot_labels = [
    f'({chr(97 + i)})'
    for i in range(
        len(EXPECTED_FILES)
    )
]


legend_items = {}


# ============================================================
# Plot all 9 subfigures
# ============================================================
for pic_index, (
        file_name,
        result
) in enumerate(
        zip(
            EXPECTED_FILES,
            all_results
        ),
        start=1
):

    axis = fig.add_subplot(
        3,
        3,
        pic_index
    )


    # ========================================================
    # Missing file
    # ========================================================
    if result is None:

        axis.axis(
            'off'
        )

        continue


    # ========================================================
    # Determine current row
    #
    # pic 1,2,3 -> row 0
    # pic 4,5,6 -> row 1
    # pic 7,8,9 -> row 2
    # ========================================================
    row_index = (
        pic_index - 1
    ) // 3


    # ========================================================
    # Get SAME x-axis settings for current row
    # ========================================================
    current_row_max_b = row_max_b[
        row_index
    ]

    xticks, x_max = get_uniform_xticks(
        current_row_max_b
    )


    # ========================================================
    # Plot algorithms
    # ========================================================
    for algo in PLOT_ORDER:

        if algo not in result:
            continue


        original_y_values = result[
            algo
        ]


        # ====================================================
        # Add:
        #
        #     b = 0
        #     Influence = 0
        #
        # ONLY for visualization.
        #
        # Original JSON is NOT modified.
        # ====================================================
        x_values = (
            [0]
            + list(
                range(
                    1,
                    len(original_y_values) + 1
                )
            )
        )


        y_values = (
            [0]
            + list(
                original_y_values
            )
        )


        # ====================================================
        # Core method on top
        # ====================================================
        zorder = (
            3
            if algo == 'CSIQ'
            else 1
        )


        # ====================================================
        # Plot
        # ====================================================
        line, = axis.plot(

            x_values,
            y_values,

            label=LABEL_NAME[
                algo
            ],

            color=COLOR_MAP[
                algo
            ],

            linestyle='-',

            linewidth=PLOT_LINE_WIDTH,

            marker=MARKER_MAP[
                algo
            ],

            markersize=5.2,

            markerfacecolor='None',

            markeredgecolor=COLOR_MAP[
                algo
            ],

            markeredgewidth=0.8,

            alpha=0.95,

            zorder=zorder,
        )


        # ====================================================
        # Store legend item
        # ====================================================
        legend_items.setdefault(
            LABEL_NAME[algo],
            line
        )


    # ========================================================
    # X-axis
    #
    # IMPORTANT:
    # same row has identical limits and ticks
    # ========================================================
    axis.set_xlim(
        0,
        x_max
    )

    axis.set_xticks(
        xticks
    )


    # ========================================================
    # Y-axis
    #
    # Since b=0 -> Influence=0
    # ========================================================
    axis.set_ylim(
        bottom=0
    )


    axis.yaxis.set_major_locator(
        MaxNLocator(
            nbins=4
        )
    )


    # ========================================================
    # Tick style
    # ========================================================
    axis.tick_params(
        axis='both',
        labelsize=8.5,
        direction='in'
    )


    # ========================================================
    # Labels
    # ========================================================
    axis.set_xlabel(
        'b',
        fontsize=10
    )

    axis.set_ylabel(
        'Influence',
        fontsize=10
    )


    # ========================================================
    # Subfigure label
    # ========================================================
    axis.text(

        0.08,
        0.86,

        subplot_labels[
            pic_index - 1
        ],

        fontsize=11.5,

        fontweight='bold',

        transform=axis.transAxes
    )


    # ========================================================
    # Border
    # ========================================================
    for spine in axis.spines.values():

        spine.set_linewidth(
            0.8
        )


# ============================================================
# Check if valid data exists
# ============================================================
if not legend_items:

    raise RuntimeError(
        f"No *_plot_data.json files found in {BASE_DIR}"
    )


# ============================================================
# Global legend
# ============================================================
legend_labels = [
    LABEL_NAME[algo]

    for algo in PLOT_ORDER

    if LABEL_NAME[algo]
    in legend_items
]


handles = [
    legend_items[label]

    for label
    in legend_labels
]


fig.legend(

    handles,
    legend_labels,

    loc='upper center',

    ncol=6,

    frameon=False,

    bbox_to_anchor=(
        0.5,
        0.995
    ),

    prop={
        'size': 8.5
    },

    handlelength=1.5,

    columnspacing=1.2,
)


# ============================================================
# Layout
# ============================================================
plt.subplots_adjust(

    left=0.08,

    right=0.98,

    bottom=0.07,

    top=0.93,

    wspace=0.24,

    hspace=0.24
)


# ============================================================
# Save PDF
# ============================================================
save_path = os.path.join(
    BASE_DIR,
    "FINAL_STYLE_FIXED.pdf"
)


plt.savefig(

    save_path,

    dpi=600,

    bbox_inches='tight',

    format='pdf'
)


print(
    "Saved:",
    save_path
)


plt.close(fig)