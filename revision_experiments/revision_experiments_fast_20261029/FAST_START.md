# 2026-10-29加速版与任务分配

## 已保留的实验设置
600训练回合、奖励50次、最终评估1000次、5独立重复、32维/32轮S2V、实际回放batch64全部不变。
本机3个完整结果已验真复用；原S2V271回合检查点已备份并接续。原目录不覆盖。
总计划仍为140项，3项已完成，剩余137项分配为本机99项与服务器38项。

## 加速实现与限制
1. 固定传播随机种子的逐节点超边选择只准备一次，保存选择后的精确RNG状态；训练传播样本与参考实现逐项一致。
2. 对相同种子集合、相同随机种子库复用完整奖励样本，最多缓存20000集合。参数固定在各独立simulator中，不跨数据集复用。
3. S2V采用稀疏投影邻接、每次前向中提取循环不变量。有效batch64不变，以microbatch8累计梯度，单次Adam更新；LayerNorm按节点特征计算，不使用BatchNorm。
4. 稀疏归约和梯度累加会改变浮点加法顺序；小图输出/梯度容差检查通过，不承诺整个训练逐位一致。
5. 本机2任务并行、最多1个S2V。服务器可2任务并行且最多2个S2V；没有足够CPU/内存时降低并发。不能保证不同硬件上的加速比。
6. 任务细分tqdm与错误写入每项任务console.log，总入口显示任务进度。每回合检查点和每条评估样本即时保存不变。
7. 原3项结果manifest保留原代码/参数；完成收据注明execution_plan_id和adopted_result。旧结果只复用品质结果，速度按执行后端和主机分层。
8. 接续原271回合的S2V为mixed-reference-fast；其训练总时间包含旧阶段，不用它估计纯加速版速度。

## 本机
实际路径D:/YX/CIM/revision_experiments_fast_20261029。
```
powershell -ExecutionPolicy Bypass -File D:/YX/CIM/revision_experiments_fast_20261029/run_local_fast.ps1
```
原run_local.ps1也已转向本入口。已在后台运行时不要再执行第二次。
本机环境使用已测试的原.runtime313；不搬运虚拟环境。

## 服务器
在现有YX目录下新建，将本压缩包内容放入该目录。
数据矩阵位于YX，缓存位于YX/Hyper_Distance，不重新上传数据、不重新计算距离。
可直接打开新版experiments.ipynb，选之前通过工程测试的CNN内核，先测试与check，再运行正式单元。
命令行等价方式（python换成该CNN环境解释器）：
```bash
cd ~/Datasets/Hypergraph_Network/YX/revision_experiments_fast_20261029
python run_suite.py check --data-root .. --cache-root ../Hyper_Distance
mkdir -p logs
nohup python -u run_suite.py run --data-root .. --cache-root ../Hyper_Distance --task-file tasks_server.json --output results/server_fast --workers 2 --s2v-workers 2 > logs/server.log 2>&1 &
```
服务器只执行tasks_server.json的38项，不要使用旧版140任务命令。
若服务器内存32GB以上且CPU核数足够，可在确认2个S2V均稳定后，以停止/回合恢复方式提高为--workers 4 --s2v-workers 3；不要并行重复启动已有任务。

## 合并
服务器完整results/server_fast搬回本机加速版目录下的results/server_fast后：
```powershell
D:/YX/CIM/revision_experiments/.runtime313/Scripts/python.exe run_suite.py merge --inputs results/local_fast results/server_fast --output merged_final
```
汇总输出必须不存在，缺任务则拒绝正式合并。timing_by_host.csv还按execution_backend分层，旧后端与加速后端不能混算同一速度结论。

## 24小时目标的证据边界
DQN在1000节点真实文件的8回合工程计时约1.17秒/回合，原完整任务平均16.46秒/回合；短测不能直接承诺600回合耗时。
S2V经验回放未满64前很快，更新开始后明显变慢。不能把前8回合平均值当作稳态速度预测。
本版保留32轮S2V；用户已接受仍可能超过一天。先看服务器更新开始后的每回合速度，再调整硬件并发；不暗减预算或重复。
工程验证13项测试通过，并行夹具分片/续跑/合并通过。尚未在服务器实际验收加速版；不承诺24小时完成全量任务。
