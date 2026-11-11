# 第二轮固定三网络的一阶/多阶筛选对照

固定新增Music-Rev(1106,694)、NDC-classes(1161,1088)、high-school_2012(1257,2078)。
原候选house-committees矩阵含2，不满足二值输入，在训练前排除；未修改矩阵或重算缓存，记录见excluded_input.json。
每网络一阶/十阶各5重复，共30项。保持600回合、奖励50次、最终评估1000次、p=.2、预算10。DQN输入始终[phi,xi10]，只改变候选筛选。
没有新增S2V。使用既有距离缓存，全部结果（包括无改善和负向结果）与此前五网络一起保留。

本机：
```powershell
powershell -ExecutionPolicy Bypass -File D:/YX/CIM/revision_C05_order_extension_round2_20261031/run_extension.ps1
```
按Music、NDC、high-school顺序，每网络3任务并行；每完成一个网络自动更新comparison/扩展结果.md。
若后台已启动，不要重复运行。中断后同一命令跳过完成任务并恢复最近完整回合。
总进度stderr.log，细分训练results/extension/任务编号/console.log。每回合检查点、每条最终评估样本即时保存，含Windows临时占用重试。
这三个网络比上一轮规模更大，而且两组均需训练，不保证每网络1小时。

移植：复制根目录代码/JSON/Notebook、previous_evidence、.ops及三个网络矩阵/各10阶缓存；不复制旧虚拟环境。
Notebook默认本机D:/YX路径，服务器需改DATA_ROOT/CACHE_ROOT并选择已安装torch的CNN环境；CLI也支持指定根路径。
使用summarize_extension.py汇总，不用旧通用分组脚本混合两个screening_orders。
代码继承上一轮已通过的底层实现；启动前检查三矩阵二值性、缓存形状/哈希，且核对一阶共享超边距离。这不代表全部阶最短路正确。

本轮为看到前五网络后的探索性扩展；校正区间只覆盖本轮三个比较，不代表持续扩展搜索已获得全局错误率保证。不得删去前轮无改善结果或只发表本轮正结果。

额外的设计诊断：high-school_2012的一阶/多阶候选池成员完全相同（251个），只有排序不同。该网络若出现差异，不能归因为保留了不同的候选节点；还可能来自候选动作编号/随机初始化和有限重复。此网络应作为集合不变的对照完整报告，不用作筛选集合改善的证据。Music和NDC候选池成员确实不同，分别重合178/221和132/232。
