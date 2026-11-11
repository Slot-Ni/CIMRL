# C05补充：一阶与多阶的候选筛选对照

## 设计
两个网络：Algebra、synthetic_HCL_v1000_e1000_exp2.5_maxdeg10（当前HCL）。
只新增一阶筛选的5次重复/网络，共10任务；复用已有多阶筛选的10个完整结果。
候选比例.2、预算10，乘积筛选。对照：xi1*phi vs xi10*phi。
两组DQN的输入始终为[phi,xi10]，所以只考察高阶信息在候选筛选阶段的增量作用，不是整个模型去掉高阶信息。
保持600回合、训练奖励50次、最终评估1000次、5独立重复、lr=.03、gamma=.9及原网络。
训练与评估随机种子银行按原重复编号匹配。只读取旧缓存，不修复或重算。

## 本机运行
代码位于D:/YX/CIM/revision_C05_order_ablation_20261030，不覆盖旧实验及结果。
```powershell
powershell -ExecutionPolicy Bypass -File D:/YX/CIM/revision_C05_order_ablation_20261030/run_order_ablation.ps1
```
默认3个任务并行。保留每回合检查点、每条评估样本即时保存与Windows检查点替换重试。
若已后台运行，不要重复启动；中断后同一命令自动跳过已完成任务并从完整回合恢复。
总进度在stderr.log，任务启动在stdout.log；每个任务的训练tqdm位于results/first_order/任务编号/console.log。
全部10项完成后启动器自动调用summarize_order_ablation.py，报告在comparison/结果分析.md和summary.json。

## 可迁移内容
迁移全部根目录py、plan_order_ablation.json、baseline_reference目录、.ops目录及Notebook，不搬.venv或现有results。
baseline_reference仅保存比较所需的10组原始评估样本、训练/评估种子和来源信息，不需要迁移模型大文件。
Linux使用已有可用环境：
```bash
python run_suite.py check --plan plan_order_ablation.json --data-root /path/to/HyperGraphMatrix --cache-root /path/to/Hyper_Distance
python -u run_suite.py run --plan plan_order_ablation.json --output results/first_order --data-root /path/to/HyperGraphMatrix --cache-root /path/to/Hyper_Distance --workers 3
python summarize_order_ablation.py
```
若把新目录放在服务器YX下、两个矩阵在YX根目录，可把data-root设..、cache-root设../Hyper_Distance。
Notebook使用同一Python实现，命令行与Notebook二选一。

## 验证与解释限制
工程测试验证了：一阶只使用首阶缓存；多阶返回原xi不变；两组DQN输入数组逐项一致，而候选池不同；低预算完整训练与保存流程通过。
实际候选池：Algebra各84节点，重合70；HCL各200节点，重合56。
汇总会核对原始样本均值、矩阵/缓存哈希、DQN特征数组哈希，以及每次重复训练/评估种子银行一致后才输出最终比较。
统计基于5个独立运行均值，给出均值、SD、配对差值95% t区间；区间为描述性、未校正两网络的多重比较。
两组输入都需要xi10，因此一阶组仍读取全部10阶缓存。不能将这次时间比较解释成一阶相对十阶距离生成成本。
HCL旧缓存已有漏算缺陷，本实验只能说明旧缓存实现下筛选的观察，不证明理想高阶定义的普遍贡献。真实缓存也未做全部阶最短路验收。
600回合等预算未降档；新增运行预计约1—3小时，取决于本机资源与训练轨迹，不保证硬截止。
