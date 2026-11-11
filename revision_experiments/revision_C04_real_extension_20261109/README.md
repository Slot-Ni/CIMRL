候选池扩展：Bars-Rev补.1/.4各5次，NDC-classes五个比例各5次。
沿用此前配置：目标方优先，双方beta=.03，T30，600回合，奖励50，最终1000，预算10，product评分。
不重新计算距离。完成任务保存complete.json；训练每回合保存检查点；最终评估逐样本保存。重新运行run_local.ps1可跳过完成项并续接训练。
总体日志stdout.log/stderr.log；细进度results/extension/任务编号/console.log。
完成后summary.json仅含新增35项；Bars已有15项仍在server_BarsRev_results_34.zip中，须按主机/后端区分耗时，不能声称统一硬件。