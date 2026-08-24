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
def load_algorithm_nodelist(algorithm_extra_para_dic, data, T, beta):
    algorithm_nodelist_dic = {}
    for algorithm_name, extra_para in algorithm_extra_para_dic.items():
        algorithm_nodelist_dic[algorithm_name] = np.load(
            f'HCL_synthetic_comparison/New_list/{algorithm_name}_({data})_(T_{T})_(beta_{beta}){extra_para}.npy', allow_pickle=True).tolist()
    return algorithm_nodelist_dic


# 主逻辑流程
def main():
    # #################### 设置全局随机种子 ####################
    seed = 42  # 固定种子
    import random
    import numpy as np
    import torch  # 如果代码中使用了PyTorch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 如果有使用PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    # #################### 所有待处理的数据列表 ####################
    all_datas = [
        # "Algebra",
        # "Restaurants-Rev",
        # "Geometry",
        # "Bars-Rev",
        # "Music-Rev",
        # "iAF1260b",
        # "iJO1366",
        # "NDC-classes",
        # "soc-hamsterster",
        # "house-committees",
        # "synthetic_HCL_v500_e1000_exp2.0_maxdeg50",
        # "synthetic_HCL_v500_e1500_exp2.0_maxdeg50",
        # "synthetic_HCL_v500_e500_exp2.0_maxdeg50",
        # "synthetic_HCL_v1000_e1000_exp2.0_maxdeg100",
        # "synthetic_HCL_v1000_e1500_exp2.0_maxdeg100",
        "synthetic_HCL_v1000_e500_exp2.0_maxdeg100"
    ]

    # 传播参数（全局共享，若不同数据需要不同参数可在循环内单独设置）
    beta = 0.025
    T = 30
    simulation_times = 10000

    # 算法配置（全局共享，无需重复定义）
    algorithm_dic = {
        # 'indicator_selection': 'HG_Diffusion_SICP_pos',
        # 'indicator_selection_greedy': 'HG_Diffusion_SICP_pos',
        # 'indicator_selection_RL_2': 'HG_Diffusion_SICP',
        'indicator_selection_RL_11': 'HG_Diffusion_SICP',
        # 'S2V_DQN_RL': 'HG_Diffusion_SICP',
        # 'degree': 'HG_Diffusion_SICP',
        # 'HADP': 'HG_Diffusion_SICP',
        # 'HDegree': 'HG_Diffusion_SICP',
        # 'CI': 'HG_Diffusion_SICP'

    }

    algorithm_extra_para_dic = {
        # 'indicator_selection': '',
        # 'indicator_selection_greedy': '_(ratio_0.2)',
        # 'indicator_selection_RL_2': '_(ratio_1)',
        'indicator_selection_RL_11': '_(ratio_0.2)',
        # 'S2V_DQN_RL': '',
        # 'degree': '',
        # 'HADP': '',
        # 'HDegree': '',
        # 'CI': ''
        # 其他算法参数保持原样...
    }

    function_mapping = {
        'HG_Diffusion_SICP': HG_Diffusion_SICP,
        'HG_Diffusion_SICP_pos': HG_Diffusion_SICP_pos
    }

    # #################### 循环处理所有数据 ####################
    for data in all_datas:  # 遍历所有数据
        print(f"\n===== 开始处理数据: {data} =====")

        # 加载当前数据的超图矩阵
        if not os.path.exists('../HyperGraphMatrix/' + data + '.npy'):
            print("saving file...")
            G_matrix, NodeList = get_Graph(data)
            np.save('../HyperGraphMatrix/' + data + '.npy', G_matrix)
            np.save('../HyperGraphMatrix/' + data + '_nodelist.npy', NodeList)
            print(f"new data: {data} save successfully！")
        else:
            print("loading file...")
            G_matrix = np.load('../HyperGraphMatrix/' + data + '.npy', allow_pickle=True)
            NodeList = np.load('../HyperGraphMatrix/' + data + '_nodelist.npy', allow_pickle=True)
            NodeList = NodeList.tolist()
            print(f"load data: {data} successfully！")

        # 计算当前数据的节点相关参数
        NodeNumber = len(NodeList)
        SeedNodes_number = int(0.01 * NodeNumber)
        CompeteNodeNumber = int(0.01 * NodeNumber)
        SeedNodes = H_Degree_SG(G_matrix, NodeList, SeedNodes_number)

        # 加载当前数据的算法节点列表
        algorithm_nodelist_dic = load_algorithm_nodelist(algorithm_extra_para_dic, data, T, beta)
        algorithm_nodelist_subsets = {name: [nodes_list[:i + 1] for i in range(len(nodes_list))] for name, nodes_list in
                                      algorithm_nodelist_dic.items()}

        # 使用共享内存优化当前数据的矩阵
        shm = SharedMemory(create=True, size=G_matrix.nbytes)
        shared_matrix = np.ndarray(G_matrix.shape, dtype=G_matrix.dtype, buffer=shm.buf)
        np.copyto(shared_matrix, G_matrix)

        # 批量执行当前数据的模拟
        for name, subsets in algorithm_nodelist_subsets.items():
            result = []
            func_name = algorithm_dic[name]
            diffusion_func = function_mapping[func_name]
            for subset in tqdm(subsets, desc=f"{data} - {name}"):
                compete_num = Simulations(
                    simulation_times,
                    diffusion_func,
                    shm.name,
                    G_matrix.dtype,
                    G_matrix.shape,
                    NodeList,
                    SeedNodes,
                    subset,
                    beta,
                    T)
                result.append(compete_num)
            np.save(f'./New_result/result_{name}_({data})_(T_{T})_(beta_{beta}){algorithm_extra_para_dic[name]}.npy',
                    np.array(result))

        # 清理当前数据的共享内存
        shm.close()
        shm.unlink()

        # 打印当前数据的结果
        for name, nodes_list in algorithm_nodelist_dic.items():
            print(f"{data} - {name}: {nodes_list}")
        print(f"===== 数据 {data} 处理完成 =====")
        print('=====================================\n')


if __name__ == "__main__":
    main()





# import numpy as np
# import os
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from multiprocessing.shared_memory import SharedMemory
# from Data import get_Graph
# from SeedNodesGenerator import H_Degree_SG
# from Diffusion_In_HyperGraph import Simulations  # 直接引用优化后的并行函数
# from Diffusion_In_HyperGraph import HG_Diffusion_SICP
# from Diffusion_In_HyperGraph import HG_Diffusion_SICP_pos
#
#
#
# # 加载数据函数，返回的algorithm_nodelist字典格式为  列表名称(字符串):节点列表(list)
# def load_algorithm_nodelist(algorithm_extra_para_dic, data, T, beta):
#     algorithm_nodelist_dic = {}
#     for algorithm_name, extra_para in algorithm_extra_para_dic.items():
#         algorithm_nodelist_dic[algorithm_name] = np.load(f'./New_list/{algorithm_name}_({data})_(T_{T})_(beta_{beta}){extra_para}.npy', allow_pickle=True).tolist()
#     return algorithm_nodelist_dic
#
#
#
#
# # 主逻辑流程
# def main():
#     # #################### 设置全局随机种子 ####################
#     seed = 42  # 固定种子
#     import random
#     import numpy as np
#     import torch  # 如果代码中使用了PyTorch
#
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)  # 如果有使用PyTorch
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     os.environ['PYTHONHASHSEED'] = str(seed)
#
#
#     # #################### 参数与数据加载 ####################
#
#     # data = "Algebra"
#     # data = "Restaurants-Rev"
#     # data = "Geometry"
#     # data = "Bars-Rev"
#     data = "Music-Rev"
#     # data = "iAF1260b"
#     # data = "iJO1366"
#     # data = "NDC-classes"  # 1
#     # data = "soc-hamsterster"
#     # data = 'house-committees'
#
#     # data = "synthetic_HCL_v500_e1000_exp2.0_maxdeg50"
#     # data = "synthetic_HCL_v500_e1500_exp2.0_maxdeg50"
#     # data = "synthetic_HCL_v500_e500_exp2.0_maxdeg50"
#     # data = "synthetic_HCL_v1000_e1000_exp2.0_maxdeg100"
#     # data = "synthetic_HCL_v1000_e1500_exp2.0_maxdeg100"
#     # data = "synthetic_HCL_v1000_e500_exp2.0_maxdeg100"
#
#
#
#
#     if not os.path.exists('../HyperGraphMatrix/' + data + '.npy'):
#         print("saving file...")
#         G_matrix, NodeList = get_Graph(data)
#         np.save('../HyperGraphMatrix/' + data + '.npy', G_matrix)
#         np.save('../HyperGraphMatrix/' + data + '_nodelist.npy', NodeList)
#         print("new data: " + data + " save successfully！")
#     else:
#         print("loading file...")
#         G_matrix = np.load('../HyperGraphMatrix/' + data + '.npy', allow_pickle=True)
#         NodeList = np.load('../HyperGraphMatrix/' + data + '_nodelist.npy', allow_pickle=True)
#         NodeList = NodeList.tolist()
#         print("load data: " + data + " successfully！")
#
#     NodeNumber = len(NodeList)
#     SeedNodes_number = int(0.01 * NodeNumber)
#     CompeteNodeNumber = int(0.01 * NodeNumber)
#
#
#     # 传播参数
#     beta = 0.025
#     T = 30
#
#
#     # 确定种子节点
#     SNG_mode = 'H_Degree'
#     SeedNodes = H_Degree_SG(G_matrix, NodeList, SeedNodes_number)
#
#     # 模拟次数
#     simulation_times = 10000
#
#
#     # 加载指标数据并生成子集
#     # algorithm_dic格式为  列表名称(字符串):使用的传播函数(字符串)
#     algorithm_dic = {
#         # 'indicator_selection': 'HG_Diffusion_SICP_pos',
#         # 'indicator_selection_greedy': 'HG_Diffusion_SICP_pos',
#         'indicator_selection_RL_2': 'HG_Diffusion_SICP',
#         # 'indicator_selection_RL_11': 'HG_Diffusion_SICP',
#         # 'S2V_DQN_RL': 'HG_Diffusion_SICP',
#         # 'degree': 'HG_Diffusion_SICP',
#         # 'HADP': 'HG_Diffusion_SICP',
#         # 'HDegree': 'HG_Diffusion_SICP',
#         # 'CI': 'HG_Diffusion_SICP'
#
#     }
#
#     algorithm_extra_para_dic = {
#         # 'indicator_selection': '',
#         # 'indicator_selection_greedy': '_(ratio_0.2)',
#         'indicator_selection_RL_2': '_(ratio_1)',
#         # 'indicator_selection_RL_11': '_(ratio_0.2)',
#         # 'S2V_DQN_RL': '',
#         # 'degree': '',
#         # 'HADP': '',
#         # 'HDegree': '',
#         # 'CI': ''
#     }
#
#     algorithm_nodelist_dic = load_algorithm_nodelist(algorithm_extra_para_dic, data, T, beta)
#     algorithm_nodelist_subsets = {name: [nodes_list[:i + 1] for i in range(len(nodes_list))] for name, nodes_list in algorithm_nodelist_dic.items()}
#
#     # 添加函数名称到函数对象的映射
#     function_mapping = {
#         'HG_Diffusion_SICP': HG_Diffusion_SICP,
#         'HG_Diffusion_SICP_pos': HG_Diffusion_SICP_pos
#     }
#
#
#     #使用共享内存优化
#     shm = SharedMemory(create=True, size=G_matrix.nbytes)
#     shared_matrix = np.ndarray(G_matrix.shape, dtype=G_matrix.dtype, buffer=shm.buf)
#     np.copyto(shared_matrix, G_matrix)
#
#     # 批量执行模拟
#     for name, subsets in algorithm_nodelist_subsets.items():
#         result = []
#         func_name = algorithm_dic[name]
#         diffusion_func = function_mapping[func_name]  # 获取实际函数
#         for subset in tqdm(subsets, desc=name):
#             compete_num = Simulations(
#                 simulation_times,
#                 diffusion_func,
#                 shm.name,  # 共享内存名称
#                 G_matrix.dtype,  # 数据类型
#                 G_matrix.shape,  # 矩阵形状
#                 NodeList,
#                 SeedNodes,
#                 subset,
#                 beta,
#                 T)
#             result.append(compete_num)
#         np.save(f'./New_result/result_{name}_({data})_(T_{T})_(beta_{beta}){algorithm_extra_para_dic[name]}.npy', np.array(result))
#     # 清理共享内存
#     shm.close()
#     shm.unlink()
#
#     # a = np.array([143.3106, 219.5638, 268.1452, 302.471, 327.4378])
#     # np.save(f'./New_result/result_indicator_selection_({data})_(T_{T})_(beta_{beta}){algorithm_extra_para_dic["indicator_selection"]}.npy',np.array(a))
#     # 打印结果
#     for name, nodes_list in algorithm_nodelist_dic.items():
#         print(f"{name}:{nodes_list}")
#     print('=====================================')
#     for name in algorithm_nodelist_dic.keys():
#         final_result = np.load(f'./New_result/result_{name}_({data})_(T_{T})_(beta_{beta}){algorithm_extra_para_dic[name]}.npy', allow_pickle=True).tolist()
#         print(f"result_{name}: {final_result}")
#
#
#
# if __name__ == "__main__":
#     main()