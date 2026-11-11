# C05探索性扩展：固定3个新增真实网络

原检验只在Algebra与synthetic_HCL_v1000_e1000_exp2.5_maxdeg10上完成。Bars-Rev尚未做一阶/多阶筛选对照。
新增网络在训练结果产生前固定为：senate-committees(282,315)、Restaurants-Rev(565,601)、Geometry(580,1193)。选择依据为规模小、十阶旧缓存齐全；不是按提升幅度挑选。
所有结果连同原两个网络一起保留，正、负及无清晰差异均报告；本扩展属探索性证据。

每个新网络一阶/多阶各5重复，共10项，3网络30项；没有可复用的现成多阶结果，所以比前次同网络仅补一阶所需训练更多。
固定600回合、奖励50次、最终评估1000次，p=.2，预算10，DQN输入[phi,xi10]，只切换screening_orders=1/10。
两个条件用相同重复编号匹配随机种子银行，不把同标量种子误称为传播事件级耦合。
继续使用旧缓存；新增3网络一阶同超边距离检查未见异常，但不是全阶/全最短路验证。

## 本机运行
```powershell
powershell -ExecutionPolicy Bypass -File D:/YX/CIM/revision_C05_order_extension_20261030/run_extension.ps1
```
按网络顺序执行：senate、Restaurants、Geometry；每个网络3任务并行，完成一个即自动更新comparison/扩展结果.md。
若已后台启动，不要再次启动。中断后同一命令跳过完成项并从训练检查点恢复。
总进度stderr.log，任务/阶段信息stdout.log，单任务训练进度results/extension/任务编号/console.log。
尚未实测完整新增网络耗时，不保证每网络恰好1小时。先看senate完成后的实际时间，不改变预算来制造速度。

## 移植
复制代码、plan_order_extension.json、tasks_*.json、cohort_declaration.json、previous_evidence、.ops及Notebook，勿复制虚拟环境。
数据矩阵和对应各10阶旧缓存仍需提供。使用run_suite.py的--data-root/--cache-root指定路径。
Notebook和命令行二选一。不得修改已开始实验的Python文件或计划；参数变化使用新目录/新计划。
只用summarize_extension.py汇总，本实验含两种screening_orders，旧通用summary按mode分组不适合直接合并本对照。

新增网络来源和固定名单见cohort_declaration.json，含候选池重合和缓存文件SHA256；完成收据记录原始文件哈希。
最终汇总检查候选筛选外的输入特征数组、训练/评估种子银行、矩阵和缓存来源是否匹配。
每网络给出配对95% t区间及针对新增3网络的Bonferroni近似区间，原先两网络结果也保留；校正不等于消除探索性选择或缓存问题。
