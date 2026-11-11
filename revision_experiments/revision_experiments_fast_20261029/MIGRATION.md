> 当前决定：本机代码直接部署在D:/YX/CIM/revision_experiments；默认单端运行。本文两端各70任务仅为可选迁移示例，尚未确定服务器分配。

# 三组返修实验：Windows / Linux 迁移与双端分工

## 1. 包的用途与边界

针对 INS-D-26-11982：C04候选池敏感性、C05分量消融与学习增益、C07公平预算比较。
默认使用 Algebra 与 synthetic_HCL_v1000_e1000_exp2.5_maxdeg10，预算10；不是18数据集全量重跑。
C04为5种比例的DQN；C05为3种筛选×3种选择方法；C07为乘积筛选p=.2下DQN/Greedy/S2V。
各5重复，去重后140任务（两端各70）。任务数不代表耗时相等。

正式参数：600训练回合，奖励50次，最终评估1000次，5独立重复，beta=.03，T=30，最高阶10，alpha=.5。
只读取旧缓存，不生成、修复或覆盖距离。合成旧缓存已有异常，结果必须披露这一限制。
传播采用同beta、目标方平局优先、同步下一步传播。论文的异beta/竞争方优先机制尚未统一，不能混称。

## 2. 文件夹必须这样放（Windows和Linux完全相同）

```text
INS_revision_server_bundle/
  YX/
    CIM/
      revision_experiments/
        run_suite.py
        shared_experiment_core.py
        s2v_adapter.py
        s2v_network.py
        summarize_suite.py
        C04_candidate_pool_sensitivity.json
        C05_component_ablation.json
        C07_fair_budget.json
        experiments.ipynb
        requirements.txt
        setup.ps1
        setup.sh
        test_shared_experiment_core.py
        test_portable_suite.py
        integration_smoke.py
        plan.json
        MIGRATION.md
    HyperGraphMatrix/
      synthetic_HCL_v1000_e1000_exp2.5_maxdeg10.npy
      synthetic_HCL_v1000_e1000_exp2.5_maxdeg10_nodelist.npy  # 可选，解释节点名称时需要
      HyperGraphMatrix/
        Algebra.npy
        Algebra_nodelist.npy  # 可选
    Hyper_Distance/
      synthetic_HCL_v1000_e1000_exp2.5_maxdeg10_distance_1.npy
      ...直到_distance_10.npy
      Hyper_Distance/
        Algebra_distance_1.npy
        ...直到_distance_10.npy
```

压缩包仅含代码和文档。必须另传两个矩阵和20个缓存；无需复制其他数据、旧.venv、.git或历史图件。
保留大小写，Linux文件名区分大小写。也可用 --data-root 和 --cache-root 指向服务器已有目录，无需重复复制。
本机已有 D:/YX/HyperGraphMatrix 和 D:/YX/Hyper_Distance，可直接传这两个根路径。

## 3. 环境安装

推荐两端相同的64位Python3.12。无需GPU，代码固定CPU、torch单线程，传播仿真主要在CPU。
本次工程测试实际使用Python3.13.5、numpy2.3.2、scipy1.16.1、torch2.9.0+cpu；不冒充已经验证3.12或Linux。
requirements固定核心依赖版本；不同平台torch构建后缀可能不同，实际版本写入manifest。

Windows，在revision_experiments目录：
```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Python C:/Users/10152/AppData/Local/Programs/Python/Python312/python.exe
.venv/Scripts/python.exe -m unittest test_shared_experiment_core test_portable_suite -v
.venv/Scripts/python.exe integration_smoke.py
```
Linux：
```bash
bash setup.sh
.venv/bin/python -m unittest test_shared_experiment_core test_portable_suite -v
.venv/bin/python integration_smoke.py
```
安装失败应保留日志，不要直接更改版本后把两端结果当作完全相同环境。
可用 `python -m pip freeze > environment_actual.txt` 保存实际环境。
Notebook选择 INS Revision (Python 3.12) 内核；若没有Jupyter界面，可另装jupyterlab或用服务器已有界面。

## 4. 两端统一计划，先检查再正式启动

包内plan.json为固定140任务计划。**两端复制同一份代码和plan.json**，不能分别修改配置后继续合并。
修改实验配置时，使用新计划路径和新输出目录：`python run_suite.py plan --plan plan_new.json`。
计划锁定Python源码哈希，运行时拒绝代码不同。每次运行check/run均核对矩阵形状与20个缓存并记录SHA256。
check仅核对文件、形状与哈希，不证明距离数学正确；run还会检验实际读取的竞争种子列。

本机（在revision_experiments目录执行，下文python替换为所选环境解释器）：
```powershell
python run_suite.py check --data-root D:/YX/HyperGraphMatrix --cache-root D:/YX/Hyper_Distance
python run_suite.py run --data-root D:/YX/HyperGraphMatrix --cache-root D:/YX/Hyper_Distance --output results/local --shards 2 --shard-index 0
```
服务器，按上面的文件夹布局放好数据：
```bash
python run_suite.py check
python -u run_suite.py run --output results/server --shards 2 --shard-index 1
```
服务器可用tmux保持会话，或：
```bash
mkdir -p logs
nohup .venv/bin/python -u run_suite.py run --output results/server --shards 2 --shard-index 1 > logs/server.log 2>&1 &
```
Notebook提供相同功能，不含复制粘贴的第二套算法。**命令行和Notebook同一端二选一启动**。
两端必须使用相同 --shards 2、不同 --shard-index 0/1；独立文件系统无法替你阻止误配同一编号。
任务按排序后的task_id取模分配，固定且互不重叠；每项独立训练，跨电脑不共享在线优化状态。
先在两端运行integration_smoke；可以用 --limit 1 试运行首个正式任务，它仍是600回合的完整任务，不是smoke。

## 5. 续跑、失败和日志

重跑原命令自动验证并跳过complete.json标记的已完成任务；只有完成仿真、模型和结果文件后才写完成标记。
任务失败保留attempt目录。DQN/S2V每完成一个回合原子保存完整训练检查点，重跑相同命令自动恢复最后完成回合；中断回合重做。Greedy失败任务仍从头选择。
RUNNING.lock防止同一输出目录同时启动同一任务。正常异常退出会移除；断电或强制kill可能残留。
只有确认对应主机PID已停止后，才手动删除该任务RUNNING.lock再重跑；不要删除完成结果。
tqdm显示任务、训练回合、Greedy选择和最终评估进度；单回合本身也可能较慢，哈希和预处理期间也有等待。
运行输出失败立即停止，不自动重算缓存或改参数。results目录不同attempt不覆盖。

## 6. 两端结果如何搬回和合并

服务器结束后，把完整results/server文件夹复制到本机revision_experiments/results/server。
必须保留每个task_id子目录、complete.json、attempt目录及全部原始文件，不能只传summary。

```powershell
python run_suite.py merge --inputs results/local results/server --output merged_final
```

缺任务、重复任务、数据或缓存哈希不同、代码不同、损坏的结果文件均拒绝正式合并。
中途查看可加 --allow-partial 并使用新的输出目录；merge_manifest.json会标记complete=false。
汇总目录必须不存在，避免覆盖。

输出：
- quality_summary.csv：独立重复均值、样本SD、t分布95%区间半宽、种子Jaccard。
- matched_quality_differences.csv：同重复编号的DQN-S2V、DQN-Greedy、筛选-全池差值及区间。
- timing_by_host.csv：按主机分别统计阶段时间，不能跨机器计算公平加速比。
- results.jsonl与merge_manifest.json：任务归属、结果目录、完成性与来源记录。

原始样本npy、随机种子npz、候选池、种子行号、模型pt、配置和manifest保留在results目录，不复制进汇总目录。
如再次搬走汇总文件，应连同results归档；results.jsonl中的result_directory是本次合并机器的绝对定位。
传播样本SD不能代替跨训练重复SD；固定同一随机种子不代表事件级耦合。
少于5重复的进度汇总不能冒充正式结果。小样本t区间不保证正态性或显著差异，不自动宣称提升。

## 7. 公平比较版本必须如何描述

S2V保留本机Baseline_S2V.py的Structure2Vec与QNetwork定义：32维、32轮消息传播，含LayerNorm。
这是**修正后的本地S2V适配版**，不声称逐字复现原论文或历史图表。
采用完整超图的加权投影：边权为共享超边数，邻接为非零权指示、对角线为0；全图嵌入、同HCDP候选池选动作。
输入为已选目标种子标签和固定竞争种子标签，不使用xi/phi。embedding和Q一起训练，目标网络覆盖二者，掩码排除已选动作，终止不bootstrap。
与HCDP-DQN统一600回合、奖励50次、lr=.03、gamma=.9、batch64、replay30000、每100决策步同步，epsilon从.5线性降到.01。
这仅保证这些预算一致，**没有执行超参数搜索，不能称调参最优或已投入等量调参努力**。
Greedy没有训练回合，公平性指同候选池、同传播后端、同50次选择评估和1000次最终评估。
固定训练种子库在训练内复用；最终评估种子库独立保留。

S2V使用稠密n×n投影和批量消息传播，内存/时间可能较高；此包默认代表数据，不承诺大规模可扩展。
模型初始化与图构建计入training_seconds，DQN/Greedy最终评估另计；缓存生成成本不在任何本轮时间中。
跨机器质量统计可合并，但公平速度比较必须从同硬件同环境的对照中得出。
需要速度结论时，可在同一机器另用 --shards 1 --shard-index 0 完整运行，输出新目录；不要和拆分结果重复合并。
不自动复用跨图策略，不自动补充gamma、alpha或增强状态等额外实验。

## 8. 验证记录

见VALIDATION.md。工程smoke不产生论文结论；正式实验尚未运行，服务器环境仍需按本文验收。

## 9. 实时保存与训练恢复（新版）

- 每个训练回合：*_training.jsonl保存影响力、epsilon和步数并flush/fsync；*_training_latest.pt通过临时文件、fsync与原子替换保存最近完整回合。包含在线/目标网络、优化器、经验回放、随机数状态和训练步数。
- 训练完成即保存*_selection.json和最终模型，再执行最终评估。
- 每条最终评估样本立即保存到*_evaluation.jsonl并flush/fsync，包括仿真编号、随机种子和影响力。
- 每个方法/比例/重复完成后立即写results.jsonl及独立*_result.json；任务全部文件写完并校验后再写complete.json。
- 续跑创建新attempt并自动读取上次训练检查点，旧attempt保留。已训完但评估中断时跳过训练，重新评估；部分旧评估样本保留，不自动拼接，避免重复计数。
- 每回合检查点含经验回放，保存会增加I/O成本；训练和评估计时包含实时保存开销。使用各机器本地磁盘。
- 原子替换与fsync不能保证磁盘硬件故障时零损失。只加载本程序自己产生的可信检查点，torch恢复使用weights_only=False。
- 不要跨机器接续未完成训练后将累积时间用于公平硬件耗时比较。
- 两端请更新整个新版包，使用相同新版plan.json及新输出目录；不要混用旧版计划和结果。
