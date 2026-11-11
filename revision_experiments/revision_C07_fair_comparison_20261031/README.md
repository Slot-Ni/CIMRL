# C07公平预算比较：先收集结果，只补缺项

## 实验矩阵
Bars-Rev；HCDP-DQN、HCDP-Greedy、修正同池S2V-DQN各5重复，共15项。
所有方法共享HCDP乘积筛选候选池p=.2、目标预算10、竞争种子、传播后端和各重复的种子银行；beta=.03、T=30、目标方平局优先。
DQN与S2V均600回合、每次奖励50仿真、最终评估1000次、lr=.03、gamma=.9、batch64、replay30000、每100决策步目标同步。
S2V保持32维/32轮，以microbatch8累计到有效batch64；全图嵌入、同池选动作。没有降低训练预算，也没有新开特征消融。
Greedy没有训练回合，其选种每次评估50仿真，最终1000次。

该S2V是修正后的本地适配版，输入为图拓扑及种子标签，不使用xi/phi。它同时改变架构和表示，因此不能称为只删除HCDP特征的严格消融；也不是原文全空间S2V。

## 使用顺序（当前未启动训练）

优先读取服务器最新目录，不要把旧ZIP中的234回合当作当前进度。
把本包放到服务器YX/revision_C07_fair_comparison_20261031下，使用之前通过测试的CNN环境。

1. 查看最新完成状态（只读，不停止任何进程）：
```bash
python fair_comparison.py inspect --source ../results/server_BarsRev
```
2. 将已完成结果验真复制到独立工作区，不动原结果，不启动训练：
```bash
python fair_comparison.py prepare --source ../results/server_BarsRev --data-root .. --cache-root ../Hyper_Distance
```
3. 汇总已有证据（未完成S2V不计为0，不生成完整S2V结论）：
```bash
python fair_comparison.py analyze
```
重复prepare可补入后来完成的S2V结果；源码/预算/矩阵缓存哈希不一致则拒绝导入。

## 只有计划迁移/接续训练时才使用

如果原来的S2V还在跑，就继续让其运行；本工具只收集和分析已完成结果。原文件夹有实际活跃调度器时，补跑命令会拒绝启动。
只有原调度器已停止、使用最新原目录时才导入其可信检查点：
```bash
python fair_comparison.py prepare --source ../results/server_BarsRev --data-root .. --cache-root ../Hyper_Distance --import-checkpoints
python fair_comparison.py run --live-source ../results/server_BarsRev --data-root .. --cache-root ../Hyper_Distance --workers 1 --s2v-workers 1
python fair_comparison.py analyze
```
不从旧ZIP直接启动补跑；ZIP只作为快照导入/分析。不能据此推断远端进程状态。跨机器源目录复制也无法证明远端已停止，必须实际确认不要重复运行。
本地Windows补跑也有检查点I/O重试；不要复制旧虚拟环境到服务器。

## 文件与输出
fair_comparison.py：inspect/prepare/run/analyze总入口；fair_report.py：配对分析；其余训练代码原样复用已验证加速实现。
experiments_C07_fair.ipynb：Notebook入口，训练单元默认注释，避免无意重复启动。
默认工作区results/fair_BarsRev。
- fair_plan.json：15项固定计划，含源码哈希。
- pending_tasks.json：尚无完整收据的任务编号，只补缺失项。
- import_receipt.json：结果来源、原计划标识、快照提醒及导入检查点状态。
- 每任务complete.json/attempt目录：保留原模型、原始样本、manifest和校验信息。
- analysis/C07分析.md、summary.json：完整性、各方法均值/跨重复SD/t区间、配对差值及主机/后端分层耗时。
导入完成收据同时保留execution_plan_id、origin_task和原始manifest；当前plan_id仅表示新调度计划，不冒充原结果由新代码训练。

## 统计/成本口径
仅完整评估1000次并成功写收据的任务进入分析，3方法各5项齐后才标记complete=true。
同重复编号比较DQN−Greedy、DQN−S2V，提供普通95% t区间与预先设定两个比较的Bonferroni近似区间。
只有一个完整S2V重复时展示单点但不提供完整五重复推断。没有搜索调参，不能称等量调参已完成或各方法最优。
相同训练回合与MC预算不等于FLOPs或墙钟一致。时间按主机和执行后端分层，不混算跨硬件加速比；旧距离生成成本不在本轮预处理中。
当前代码复用原训练算法，主要节省重复运行，不承诺32轮S2V能快速完成。不要再从头训练已完成的DQN和Greedy。

## 已完成的软件验证
3项单元测试（任务身份/15单元、路径安全、缺失与单重复区间）；旧快照真实导入10项完成结果；第234回合可信检查点迁移校验；运行中原调度器拦截测试。
所有导入记录核对样本、候选池、参数和算法源码。这里只是软件/导入验收，未启动新训练，服务器最新S2V状态仍需inspect读取。
