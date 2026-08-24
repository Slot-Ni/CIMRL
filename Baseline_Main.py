import time

import numpy as np
import os

from Data import get_Graph
from SeedNodesGenerator import H_Degree_SG
from Diffusion_In_HyperGraph import HG_Diffusion_SICP
from Baseline_Algorithm import Indicator_selection_Greedy
from Baseline_Algorithm import Degree
from Baseline_Algorithm import HADP
from Baseline_Algorithm import H_Degree
from Baseline_Algorithm import CI
from Indicator import Competition_composite_indicator_new
#
#
def main():
    # 定义所有需要处理的数据列表
    data_list = [
        # "Algebra",
        # "Restaurants-Rev",
        # "Geometry",
        # "Bars-Rev",
        # "Music-Rev",
        "iAF1260b",
        "iJO1366",
        "NDC-classes",
        "soc-hamsterster",
        "house-committees",
        "synthetic_HCL_v500_e1000_exp2.0_maxdeg50",
        "synthetic_HCL_v500_e1500_exp2.0_maxdeg50",
        "synthetic_HCL_v500_e500_exp2.0_maxdeg50",
        "synthetic_HCL_v1000_e1000_exp2.0_maxdeg100",
        "synthetic_HCL_v1000_e1500_exp2.0_maxdeg100",
        "synthetic_HCL_v1000_e500_exp2.0_maxdeg100"
    ]

    for data in data_list:
        print(f"\n开始处理数据: {data}")  # 打印当前处理的数据，方便跟踪进度

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
        CompeteNodes_number = int(0.01 * NodeNumber)
        CandidateNodes_Ratio = 0.2

        # 传播参数
        beta = 0.025
        T = 30
        Indicator_Order = 10
        Indicator_Alpha = 0.5

        # 确定种子节点
        SNG_mode = 'H_Degree'
        SeedNodes = H_Degree_SG(G_matrix, NodeList, SeedNodes_number)

        SeedNodes_pos = []
        for i in SeedNodes:
            SeedNodes_pos.append(NodeList.index(i))

        #################### 指标筛选 返回节点索引 ####################
        start = time.time()
        Indicator_selection_pos = Competition_composite_indicator_new(SeedNodes_pos, G_matrix, data,
                                                                      CompeteNodes_number,
                                                                      Indicator_Order, Indicator_Alpha)
        np.save(f'./New_List/indicator_selection_({data})_(T_{T})_(beta_{beta}).npy', Indicator_selection_pos)
        print("indicator_selection is finish!")
        print("time:" + str(time.time() - start))
        #################### 指标筛选后贪婪算法 返回节点索引 ####################
        start = time.time()
        greedy_times = 1000  # 贪婪模拟次数
        CandidateNodes_pos = Competition_composite_indicator_new(SeedNodes_pos, G_matrix, data,
                                                                 int(CandidateNodes_Ratio * NodeNumber),
                                                                 Indicator_Order,
                                                                 Indicator_Alpha).tolist()
        Indicator_selection_greedy_pos = Indicator_selection_Greedy(G_matrix, NodeList, SeedNodes, CandidateNodes_pos,
                                                                    CompeteNodes_number, greedy_times, beta, T)
        np.save(
            f'./New_List/indicator_selection_greedy_({data})_(T_{T})_(beta_{beta})_(ratio_{CandidateNodes_Ratio}).npy',
            Indicator_selection_greedy_pos)
        print("indicator_selection_greedy is finish!")
        print("time:" + str(time.time() - start))
        # #################### 最大度算法 返回节序号 ####################
        start = time.time()
        Degree_list = Degree(G_matrix, NodeList, SeedNodes, CompeteNodes_number)
        np.save(f'./New_List/degree_({data})_(T_{T})_(beta_{beta}).npy', Degree_list)
        print("degree is finish!")
        print("time:" + str(time.time() - start))
        # #################### 最大超度算法 返回节序号 ####################
        start = time.time()
        HDegree_list = H_Degree(G_matrix, NodeList, SeedNodes, CompeteNodes_number)
        np.save(f'./New_List/HDegree_({data})_(T_{T})_(beta_{beta}).npy', HDegree_list)
        print("hdegree is finish!")
        print("time:" + str(time.time() - start))
        print(HDegree_list)
        # #################### HADP扩展算法 返回节序号 ####################
        start = time.time()
        HADP_list = HADP(G_matrix, NodeList, SeedNodes, CompeteNodes_number)
        np.save(f'./New_List/HADP_({data})_(T_{T})_(beta_{beta}).npy', HADP_list)
        print("HADP is finish!")
        print("time:" + str(time.time() - start))
        print(HADP_list)
        # #################### CI扩展算法 返回节序号 ####################
        start = time.time()
        CI_list = CI(G_matrix, NodeList, SeedNodes, CompeteNodes_number)
        np.save(f'./New_List/CI_({data})_(T_{T})_(beta_{beta}).npy', CI_list)
        print("CI is finish!")
        print("time:" + str(time.time() - start))
        print(CI_list)
        #################### 展示结果 ####################
        # print(f'indicator_selelction:{Indicator_selection_pos}')
        # print(f'Indicator_selection_greedy:{Indicator_selection_greedy_pos}')
        # print(f'degree:{Degree_list}')


if __name__ == '__main__':
    main()


