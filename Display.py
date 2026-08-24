import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing.shared_memory import SharedMemory
from Data import get_Graph
from SeedNodesGenerator import H_Degree_SG
from Diffusion_In_HyperGraph import Simulations  # 直接引用优化后的并行函数
from Diffusion_In_HyperGraph import HG_Diffusion_SICP
from Diffusion_In_HyperGraph import HG_Diffusion_SICP_pos



# 加载数据函数，返回的algorithm_nodelist字典格式为  列表名称(字符串):节点列表(list)
def load_algorithm_result(algorithm_extra_para_dic, data, T, beta):
    algorithm_result_dic = {}
    for algorithm_name, extra_para in algorithm_extra_para_dic.items():
        for para in extra_para:
            if algorithm_name not in algorithm_result_dic:
                algorithm_result_dic[algorithm_name] = []
            algorithm_result_dic[algorithm_name].append(np.load(f'./New_result/result_{algorithm_name}_({data})_(T_{T})_(beta_{beta}){para}.npy', allow_pickle=True).tolist())

    return algorithm_result_dic




# 主逻辑流程
def main():
    # #################### 参数与数据加载 ####################

    # data = "Algebra"
    # data = "Restaurants-Rev"
    # data = "Geometry"
    # data = "Bars-Rev"
    # data = "Music-Rev"
    # data = "iAF1260b"
    # data = "iJO1366"
    # data = "NDC-classes"
    # data = "soc-hamsterster"
    # data = 'house-committees'

    # data = "synthetic_HCL_v500_e1000_exp2.0_maxdeg50"
    # data = "synthetic_HCL_v500_e1500_exp2.0_maxdeg50"
    # data = "synthetic_HCL_v500_e500_exp2.0_maxdeg50"
    # data = "synthetic_HCL_v1000_e1000_exp2.0_maxdeg100"
    # data = "synthetic_HCL_v1000_e1500_exp2.0_maxdeg100"
    data = "synthetic_HCL_v1000_e500_exp2.0_maxdeg100" # 1

# 传播参数
    beta = 0.025
    T = 30

    # 确定种子节点
    SNG_mode = 'H_Degree'




    # 加载结果
    algorithm_extra_para_dic = {
        'indicator_selection_RL_11': ['_(ratio_0.2)'],
        'indicator_selection': [''],
        'indicator_selection_RL_2': ['_(ratio_1)'],
        'indicator_selection_greedy': ['_(ratio_0.2)'],
        'S2V_DQN_RL': [''],
        # 'degree': [''],
        'HADP': [''],
        'HDegree': [''],
        'CI':['']
    }

    algorithm_result = load_algorithm_result(algorithm_extra_para_dic, data, T, beta)
    print(algorithm_result)

    # #################### 绘制对比图 ####################

    # 准备绘图
    plt.figure(figsize=(12, 6))

    # 为每个算法绘制曲线
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#FAC7B3', '#EC6E66', '#91CCC0', '#F7AC53', '#FFD869', '#BD7795', '#7FABD1']  # 预定义颜色
    linestyles = ['--', '--', '--', '--', '--', '--']  # 预定义线型
    markers = ['8', 's', '>', '^', '<', '*', 'x', 'h', 'd', 'p']

    lable_name_dic = {
        'indicator_selection_RL_11': 'HCDP-DQN',
        'indicator_selection': 'indicator_selection',
        'indicator_selection_RL_2': 'DQN',
        'indicator_selection_greedy': 'HCDP-Greedy',
        'S2V_DQN_RL': 'S2V_DQN',
        'degree': 'degree',
        'HADP': 'HADP',
        'HDegree': 'HDegree',
        'CI': 'CI'
    }

    idx = 0
    for name, data_group in algorithm_result.items():
        for data_index in range(len(data_group)):
            seed_counts = list(range(1, len(data_group[data_index]) + 1))
            # 判断是否为RL11线条，是的话设置更高的zorder
            zorder = 10 if name == 'indicator_selection_RL_11' else 1
            plt.plot(
                seed_counts,
                data_group[data_index],
                label=lable_name_dic[name],
                color=colors[idx % len(colors)],
                linestyle='--',
                linewidth=2,
                marker=markers[idx % len(markers)],
                markersize=6,
                markerfacecolor='None',
                zorder=zorder  # 关键：通过zorder控制层级
            )
            idx += 1

    # 图表装饰
    # plt.title(f'Competitive Nodes Influence ({data})\nβ={beta}, T={T}')
    plt.title(f'Competitive Nodes Influence ({data})')
    # plt.title(f'Ablation ({data})')
    plt.xlabel('Seed Set Size')
    plt.ylabel('Influence')
    plt.grid(True, linestyle='--', alpha=0.7)
    handles, labels = plt.gca().get_legend_handles_labels()
    desired_order = ['HCDP-Greedy', 'HDgree', 'HADP', 'CI', 'S2V_DQN', 'HCDP-DQN']
    legend_map = {label: handle for handle, label in zip(handles, labels)}
    ordered_labels = [label for label in desired_order if label in legend_map]
    ordered_labels.extend([label for label in labels if label not in desired_order])
    plt.legend([legend_map[label] for label in ordered_labels], ordered_labels, loc='best')

    # 自动调整刻度
    # max_seeds = max(len(data) for data in result_data.values()) if result_data else 1
    # plt.xticks(np.arange(0, max_seeds + 1, max(1, max_seeds // 10)))

    # 保存图片
    plot_path = f'./New_pic/Result_({data})_(T_{T})_(beta_{beta}).png'
    # plot_path = f'./New_pic/Ablation_({data})_(T_{T})_(beta_{beta}).png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存至: {plot_path}")

    # 显示图表（如果是在交互环境）
    plt.show()


if __name__ == "__main__":
    main()
