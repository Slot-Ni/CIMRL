# 加速版返修实验（2026-10-29）

请首先阅读FAST_START.md。当前实际分配：3项旧完整结果已复用，本机99项，服务器38项。使用tasks_local.json/tasks_server.json，不能再使用旧版140项全量启动命令。
S2V保留32轮，训练600回合，训练奖励50次、最终评估1000次、5重复。
新版Notebook默认仅启动服务器38项，数据路径为上一级YX及其Hyper_Distance。
每项细分进度见results/对应输出/任务编号/console.log。
原MIGRATION.md和本机运行说明为历史布局参考，当前启动方式以FAST_START.md为准。
