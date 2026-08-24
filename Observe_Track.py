import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing.shared_memory import SharedMemory
from Data import get_Graph
from SeedNodesGenerator import H_Degree_SG
from Diffusion_In_HyperGraph import Simulations_track_tqdm
from Diffusion_In_HyperGraph import HG_Diffusion_SICP_pos_track
from Diffusion_In_HyperGraph import HG_Diffusion_SICP_track



# 加载数据函数，返回的algorithm_nodelist字典格式为  列表名称(字符串):节点列表(list)
def load_algorithm_nodelist(algorithm_extra_para_dic, data, T, beta):
    algorithm_nodelist_dic = {}
    for algorithm_name, extra_para in algorithm_extra_para_dic.items():
        algorithm_nodelist_dic[algorithm_name] = np.load(f'./New_list/{algorithm_name}_({data})_(T_{T})_(beta_{beta}){extra_para}.npy', allow_pickle=True).tolist()
    return algorithm_nodelist_dic




# 主逻辑流程
def main():
    # #################### 参数与数据加载 ####################

    # data = "Algebra"
    # data = "iJO1366"
    # data = "Restaurants-Rev"
    # data = "Geometry"

    # data = "Bars-Rev"
    # data = "Music-Rev"
    # data = "iAF1260b"
    # data = "NDC-classes"
    # data = "soc-hamsterster"
    # data = "fb-tvshow"
    data = "senate-committees"

    if not os.path.exists('../HyperGraphMatrix/' + data + '.npy'):
        print("saving file...")
        G_matrix, NodeList = get_Graph(data)
        np.save('../HyperGraphMatrix/' + data + '.npy', G_matrix)
        np.save('../HyperGraphMatrix/' + data + '_nodelist.npy', NodeList)
        print("new data: " + data + " save successfully！")
    else:
        print("loading file...")
        G_matrix = np.load('../HyperGraphMatrix/' + data + '.npy', allow_pickle=True)
        NodeList = np.load('../HyperGraphMatrix/' + data + '_nodelist.npy', allow_pickle=True)
        NodeList = NodeList.tolist()
        print("load data: " + data + " successfully！")

    NodeNumber = len(NodeList)
    SeedNodes_number = int(0.01 * NodeNumber)
    CompeteNodeNumber = int(0.01 * NodeNumber)


    CandidateNodes_Ratio = 0.05

    # 传播参数
    beta = 0.01
    T = 30
    interval = 1  # 观察间隔

    # 确定种子节点
    SNG_mode = 'H_Degree'
    SeedNodes = H_Degree_SG(G_matrix, NodeList, SeedNodes_number)

    # 模拟次数
    simulation_times = 10000



    # 加载指标数据并生成子集
    # algorithm_dic格式为  列表名称(字符串):使用的传播函数(字符串)
    algorithm_dic = {
        'indicator_selection': 'HG_Diffusion_SICP_pos',
        'indicator_selection_greedy': 'HG_Diffusion_SICP_pos'
    }

    algorithm_extra_para_dic = {
        'indicator_selection': '',
        'indicator_selection_greedy': '_(ratio_' + str(CandidateNodes_Ratio) + ')',
        'indicator_selection_RL_1': '_(ratio_' + str(CandidateNodes_Ratio) + ')',
        # 'indicator_selection_RL_2': '_(ratio_' + str(CandidateNodes_Ratio) + ')',
        # 'indicator_selection_RL_3': '_(ratio_' + str(CandidateNodes_Ratio) + ')',
        'degree': ''
    }

    algorithm_nodelist_dic = load_algorithm_nodelist(algorithm_extra_para_dic, data, T, beta)

    # 添加函数名称到函数对象的映射
    function_mapping = {
        'HG_Diffusion_SICP': HG_Diffusion_SICP_track,
        'HG_Diffusion_SICP_pos': HG_Diffusion_SICP_pos_track
    }

    #使用共享内存优化
    shm = SharedMemory(create=True, size=G_matrix.nbytes)
    shared_matrix = np.ndarray(G_matrix.shape, dtype=G_matrix.dtype, buffer=shm.buf)
    np.copyto(shared_matrix, G_matrix)

    # 批量执行模拟

    for name, nodelist in algorithm_nodelist_dic.items():
        func_name = algorithm_dic[name]
        diffusion_func = function_mapping[func_name]  # 获取实际函数
        track_list = Simulations_track_tqdm(simulation_times,
                                  diffusion_func,
                                  shm.name,  # 共享内存名称
                                  G_matrix.dtype,  # 数据类型
                                  G_matrix.shape,  # 矩阵形状
                                  NodeList,
                                  SeedNodes,
                                  nodelist,
                                  beta,
                                  T,
                                  interval)

        np.save(f'./New_result/track_{name}_({data})_(T_{T})_(beta_{beta}).npy', np.array(track_list))
    # 清理共享内存
    shm.close()
    shm.unlink()
    # 打印节点列表
    for name, nodes_list in algorithm_nodelist_dic.items():
        print(f"{name}:{nodes_list}")



    # #################### 绘制对比图 ####################

    # 获取所有跟踪文件
    track_data = {}
    for name in algorithm_nodelist_dic.keys():
        try:
            file_path = f'./New_result/track_{name}_({data})_(T_{T})_(beta_{beta}).npy'
            track_array = np.load(file_path, allow_pickle=True)
            track_data[name] = track_array
            print(f"成功加载 {name} 的跟踪数据")
        except Exception as e:
            print(f"加载 {name} 数据失败: {str(e)}")
            continue

    # 准备绘图
    plt.figure(figsize=(12, 6))
    time_points = np.arange(0, T + 1, interval)  # 生成时间轴

    # 为每个算法绘制曲线
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 预定义颜色
    linestyles = ['-', '--', ':']  # 预定义线型

    for idx, (name, data_array) in enumerate(track_data.items()):
        if len(data_array) != len(time_points):
            print(f"数据长度不匹配: {name} (期望{len(time_points)} 实际{len(data_array)})")
            continue

        plt.plot(
            time_points,
            data_array,
            label=name,
            color=colors[idx % len(colors)],
            linestyle=linestyles[idx % len(linestyles)],
            linewidth=2,
            marker='o' if idx < 2 else None,
            markersize=4
        )

    # 图表装饰
    plt.title(f'Blocking Nodes Comparison ({data})\nβ={beta}, T={T}, interval={interval}')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Blocked Nodes')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')

    # 自动调整刻度
    max_time = T + (interval - T % interval) if T % interval != 0 else T
    plt.xticks(np.arange(0, max_time + 1, interval * 5))  # 每5个间隔显示一个刻度

    # 保存图片
    plot_path = f'./New_pic/Comparison_({data})_(T_{T})_(beta_{beta})_(inter_{interval}).png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存至: {plot_path}")

    # 显示图表（如果是在交互环境）
    plt.show()



if __name__ == "__main__":
    main()


