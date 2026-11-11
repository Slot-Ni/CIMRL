# 需要搬走的文件

1. 本目录全部代码、JSON配置、plan.json、Notebook、MD、requirements与安装脚本；压缩包已包含。
2. 两个矩阵：Algebra.npy、synthetic_HCL_v1000_e1000_exp2.5_maxdeg10.npy；另行复制。
3. 上述两个数据集各1—10阶distance缓存，共20个npy；另行复制。
4. 可选的两份nodelist.npy，用于把种子行号转换为节点名。
5. 搬回结果时复制完整results/server，不只复制汇总表。

不搬旧.venv、__pycache__或其他项目的CIM副本。完整层级与命令见MIGRATION.md。

历史兼容文件C04/C05单独入口、launch.ps1和旧summary仍保留；新版双端流程统一用run_suite.py。
