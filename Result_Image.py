import math

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ============================================================
# Global figure settings
# ============================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'


# ============================================================
# Dataset
# ============================================================
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


# ============================================================
# Algorithms
# ============================================================
algorithm_extra_para_dic = {
    'indicator_selection_RL_11': ['_(ratio_0.2)'],
    'indicator_selection_greedy': ['_(ratio_0.2)'],
    'S2V_DQN_RL': [''],
    'HADP': [''],
    'HDegree': [''],
    'CI': [''],
}


# ============================================================
# Algorithm names shown in figure
# ============================================================
lable_name_dic = {
    'indicator_selection_RL_11': 'HCDP-DQN',
    'indicator_selection_greedy': 'HCDP-Greedy',
    'S2V_DQN_RL': 'S2V-DQN',
    'HADP': 'HADP',
    'HDegree': 'HDegree',
    'CI': 'CI',
}


# ============================================================
# Color settings
# ============================================================
color_dic = {
    'indicator_selection_RL_11': '#df8a80',
    'indicator_selection_greedy': '#9fd3df',
    'S2V_DQN_RL': '#f2b179',
    'HADP': '#8bc487',
    'HDegree': '#b7a5d8',
    'CI': '#9db9dc',
}


# ============================================================
# Marker settings
# ============================================================
marker_dic = {
    'indicator_selection_RL_11': 'o',
    'indicator_selection_greedy': 's',
    'S2V_DQN_RL': '^',
    'HADP': 'D',
    'HDegree': 'v',
    'CI': 'P',
}


# ============================================================
# Legend order
# ============================================================
LEGEND_ORDER = [
    'HCDP-Greedy',
    'HDegree',
    'HADP',
    'CI',
    'S2V-DQN',
    'HCDP-DQN',
]


# ============================================================
# Subfigure labels
# ============================================================
labels = [
    f'({chr(97 + i)})'
    for i in range(len(dataset))
]


# ============================================================
# Experimental parameters
# ============================================================
beta = 0.025
T = 30

PLOT_LINE_WIDTH = 1.5

ROWS = 3
COLS = 3

PAGE_SIZE = ROWS * COLS


# ============================================================
# Load results
# ============================================================
def load_algorithm_result(
        algorithm_extra_para_dic,
        data,
        T,
        beta
):

    algorithm_result_dic = {}

    for algorithm_name, extra_para in algorithm_extra_para_dic.items():

        for para in extra_para:

            algorithm_result_dic.setdefault(
                algorithm_name,
                []
            )

            file_path = (
                f'./New_result/'
                f'result_{algorithm_name}_'
                f'({data})_'
                f'(T_{T})_'
                f'(beta_{beta})'
                f'{para}.npy'
            )

            result = np.load(
                file_path,
                allow_pickle=True
            ).tolist()

            algorithm_result_dic[
                algorithm_name
            ].append(result)

    return algorithm_result_dic


# ============================================================
# Calculate average performance
# ============================================================
def calculate_avg_performance(data_group):

    all_values = []

    for seed_data in data_group:

        if seed_data:
            all_values.extend(seed_data)

    if all_values:
        return np.mean(all_values)

    return np.nan


# ============================================================
# Print improvement summary
# ============================================================
def print_improvement_summary():

    comparison_algos = [
        'indicator_selection_greedy',
        'S2V_DQN_RL',
        'HADP',
        'HDegree',
        'CI'
    ]

    dataset_improvement = {}

    all_valid_improvements = []

    print("=" * 80)

    print(
        "Start calculating average improvement "
        "of HCDP-DQN over baselines"
    )

    print(
        "Comparison algorithms:",
        comparison_algos
    )

    print("=" * 80)


    for data in dataset:

        algo_results = load_algorithm_result(
            algorithm_extra_para_dic,
            data,
            T,
            beta
        )

        hcdp_avg = calculate_avg_performance(
            algo_results.get(
                'indicator_selection_RL_11',
                []
            )
        )


        if np.isnan(hcdp_avg):

            print(
                f"{data}: "
                f"HCDP-DQN has no valid data, skipped"
            )

            dataset_improvement[data] = np.nan

            continue


        improvements = []


        for algo in comparison_algos:

            algo_avg = calculate_avg_performance(
                algo_results.get(
                    algo,
                    []
                )
            )


            if np.isnan(algo_avg):

                print(
                    f"{data}: "
                    f"{algo} has no valid data, skipped"
                )

                continue


            improvement = (
                (hcdp_avg - algo_avg)
                / algo_avg
                * 100
            )


            improvements.append(
                improvement
            )


            all_valid_improvements.append(
                improvement
            )


            print(
                f"  {data} - {algo}: "
                f"{improvement:.2f}%"
            )


        if improvements:

            dataset_improvement[data] = np.mean(
                improvements
            )

        else:

            dataset_improvement[data] = np.nan


        print("-" * 50)


    if all_valid_improvements:

        print("\n" + "=" * 80)

        print(
            f"Global average improvement: "
            f"{np.mean(all_valid_improvements):.2f}% "
            f"(std: "
            f"{np.std(all_valid_improvements):.2f}%)"
        )

        print("=" * 80)


# ============================================================
# Get max b of one subplot
# ============================================================
def get_result_max_b(algorithm_result):

    max_b = 0

    for data_group in algorithm_result.values():

        for values in data_group:

            max_b = max(
                max_b,
                len(values)
            )

    return max_b


# ============================================================
# Generate x-axis ticks
#
# Main principle:
#
# 1. Start = 0
# 2. End = exact real max_b
# 3. Last result is exactly at the right boundary
# 4. About 5 major ticks
# 5. Last interval is allowed to be non-uniform
#
# Example:
#
# max_b = 11
# -> 0, 3, 6, 9, 11
#
# max_b = 18
# -> 0, 5, 10, 15, 18
#
# max_b = 24
# -> 0, 6, 12, 18, 24
# ============================================================
def get_exact_xticks(max_b):

    if max_b <= 0:
        return [0]


    # ========================================================
    # For very small b, show all integer values
    # ========================================================
    if max_b <= 5:

        return list(
            range(
                0,
                max_b + 1
            )
        )


    # ========================================================
    # Desired step:
    #
    # roughly divide the axis into 4 intervals,
    # giving around 5 tick labels
    # ========================================================
    target_step = max_b / 4.0


    # ========================================================
    # Prefer visually clean integer steps
    # ========================================================
    nice_steps = [
        1,
        2,
        3,
        4,
        5,
        6,
        8,
        10,
        12,
        15,
        20,
        25,
        30,
        40,
        50,
        60,
        80,
        100,
    ]


    # ========================================================
    # Choose first clean step >= target_step
    #
    # Examples:
    #
    # max_b = 11
    # target = 2.75
    # step = 3
    #
    # max_b = 18
    # target = 4.5
    # step = 5
    #
    # max_b = 24
    # target = 6
    # step = 6
    # ========================================================
    step = None

    for candidate in nice_steps:

        if candidate >= target_step:

            step = candidate

            break


    # Fallback
    if step is None:

        step = int(
            math.ceil(
                target_step
            )
        )


    # ========================================================
    # Generate regular ticks first
    # ========================================================
    ticks = list(
        range(
            0,
            max_b + 1,
            step
        )
    )


    # ========================================================
    # IMPORTANT:
    #
    # Always include the REAL endpoint max_b.
    #
    # Therefore the last interval may be shorter.
    #
    # Example:
    #
    # max_b = 18
    #
    # regular:
    # 0, 5, 10, 15
    #
    # final:
    # 0, 5, 10, 15, 18
    # ========================================================
    if ticks[-1] != max_b:

        ticks.append(
            max_b
        )


    return ticks


# ============================================================
# Plot one page
# ============================================================
def plot_page(
        page_dataset,
        page_index,
        start_index
):

    # ========================================================
    # Figure
    # ========================================================
    fig = plt.figure(
        figsize=(7.2, 7.4)
    )


    # ========================================================
    # Legend storage
    # ========================================================
    legend_items = {}


    # ========================================================
    # Plot every subplot
    # ========================================================
    for local_index, data in enumerate(
            page_dataset,
            start=1
    ):

        global_index = (
            start_index
            + local_index
        )


        # ====================================================
        # Load current dataset
        # ====================================================
        algorithm_result = load_algorithm_result(
            algorithm_extra_para_dic,
            data,
            T,
            beta
        )


        # ====================================================
        # Current axis
        # ====================================================
        axis = fig.add_subplot(
            ROWS,
            COLS,
            local_index
        )


        # ====================================================
        # Exact maximum b of CURRENT subplot
        # ====================================================
        max_b = get_result_max_b(
            algorithm_result
        )


        # ====================================================
        # Exact ticks
        # ====================================================
        xticks = get_exact_xticks(
            max_b
        )


        # ====================================================
        # Plot curves
        # ====================================================
        for name, data_group in algorithm_result.items():

            is_core = (
                name
                == 'indicator_selection_RL_11'
            )


            zorder = (
                3
                if is_core
                else 1
            )


            for values in data_group:

                # ============================================
                # IMPORTANT:
                #
                # DO NOT use np.linspace here.
                #
                # Preserve real seed-budget values:
                #
                # b = 1, 2, 3, ..., len(values)
                #
                # Only add:
                #
                # b = 0 -> Influence = 0
                # ============================================
                x_values = (
                    [0]
                    + list(
                        range(
                            1,
                            len(values) + 1
                        )
                    )
                )


                y_values = (
                    [0]
                    + list(values)
                )


                # ============================================
                # Plot
                # ============================================
                line, = axis.plot(

                    x_values,

                    y_values,

                    label=lable_name_dic[
                        name
                    ],

                    color=color_dic[
                        name
                    ],

                    linestyle='-',

                    linewidth=PLOT_LINE_WIDTH,

                    marker=marker_dic[
                        name
                    ],

                    markersize=5.2,

                    markerfacecolor='None',

                    markeredgecolor=color_dic[
                        name
                    ],

                    markeredgewidth=0.8,

                    alpha=0.95,

                    zorder=zorder,
                )


                # ============================================
                # Legend handle
                # ============================================
                legend_items.setdefault(
                    lable_name_dic[name],
                    line
                )


        # ====================================================
        # X-axis
        #
        # KEY:
        #
        # x_max is EXACTLY current real max_b.
        #
        # Therefore:
        #
        # curve endpoint = axis endpoint
        #
        # No blank space on the right.
        # ====================================================

        # First ticks
        axis.set_xticks(
            xticks
        )


        # Then strictly fix the range
        axis.set_xlim(
            0,
            max_b
        )


        # ====================================================
        # Y-axis starts at 0
        # ====================================================
        axis.set_ylim(
            bottom=0
        )


        axis.yaxis.set_major_locator(
            MaxNLocator(
                nbins=4
            )
        )


        # ====================================================
        # Tick style
        # ====================================================
        axis.tick_params(

            axis='both',

            labelsize=8.5,

            direction='in'
        )


        # ====================================================
        # Axis labels
        # ====================================================
        axis.set_xlabel(
            'b',
            fontsize=10
        )


        axis.set_ylabel(
            'Influence',
            fontsize=10
        )


        # ====================================================
        # Subfigure label
        # ====================================================
        axis.text(

            0.08,

            0.86,

            labels[
                global_index - 1
            ],

            fontsize=11.5,

            fontweight='bold',

            transform=axis.transAxes
        )


        # ====================================================
        # Border
        # ====================================================
        for spine in axis.spines.values():

            spine.set_linewidth(
                0.8
            )


        # ====================================================
        # Print x-axis information
        # Useful for checking
        # ====================================================
        print(
            f"{data}: "
            f"max_b = {max_b}, "
            f"xticks = {xticks}"
        )


    # ========================================================
    # Global legend
    # ========================================================
    ordered_labels = [

        label

        for label
        in LEGEND_ORDER

        if label in legend_items
    ]


    ordered_handles = [

        legend_items[
            label
        ]

        for label
        in ordered_labels
    ]


    fig.legend(

        ordered_handles,

        ordered_labels,

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


    # ========================================================
    # Layout
    # ========================================================
    plt.subplots_adjust(

        left=0.075,

        right=0.985,

        bottom=0.065,

        top=0.93,

        wspace=0.24,

        hspace=0.24
    )


    # ========================================================
    # Output path
    # ========================================================
    if len(dataset) > PAGE_SIZE:

        suffix = (
            f'_page{page_index}'
        )

    else:

        suffix = ''


    plot_path = (
        f'./New_pic/'
        f'Influence_result_{beta}'
        f'{suffix}.pdf'
    )


    # ========================================================
    # Save
    # ========================================================
    plt.savefig(

        plot_path,

        dpi=600,

        bbox_inches='tight',

        format='pdf'
    )


    print(
        f"\nSaved: {plot_path}"
    )


    # ========================================================
    # Close
    # ========================================================
    plt.close(
        fig
    )


# ============================================================
# Main
# ============================================================
def main():

    # ========================================================
    # Improvement statistics
    #
    # Added (0,0) does NOT participate in statistics.
    # ========================================================
    print_improvement_summary()


    # ========================================================
    # Number of pages
    # ========================================================
    total_pages = math.ceil(
        len(dataset)
        / PAGE_SIZE
    )


    # ========================================================
    # Plot
    # ========================================================
    for page_index in range(
            total_pages
    ):

        start = (
            page_index
            * PAGE_SIZE
        )


        page_dataset = dataset[
            start:
            start + PAGE_SIZE
        ]


        plot_page(

            page_dataset,

            page_index + 1,

            start
        )


# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":

    main()