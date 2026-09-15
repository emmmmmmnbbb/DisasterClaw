# Agent-VQA 在线结果失效记录

日期：2026-09-12。

此前 Agent-VQA 在线运行及固定视图重问结果不得用于论文数值、策略比较或模型能力结论。原因如下：

1. damage 题的可见十字固定绘制在整幅图中心，没有使用题目目标的 `ref_id` 和经纬度；
2. damage 题解析时目标损伤亚类为空，旧证据过滤据此排除了全部正常损伤类别检测；
3. 控制器计算出的目标标签、预测亚类、匹配数量和位置没有传入 VLM；
4. 最终题集没有逐题完成人工审核，自动几何检查不能替代人工审核。

为避免误用，下列内容已从工作区清除（2026-09-12）：

- 在线运行目录：`runs/benchmarks/cja_agent_vqa/paper_cja_mech_final_{shard,rerun_shard,reports,rerun_reports,contaminated_*}` 及其启动日志，共 13 项、约 117 MB（已删除）。
- 可被再次引用的汇总入口：`cja_en/review/manuscript_analysis.json` 的 `online` 分块已改为失效标记，不再含任何在线数值；`cja_en/scripts/build_assets.py` 不再读回旧在线 episode。
- 执行记录 `cja_en/review/experiment_run_plan.md` 顶部已加失效声明，仅保留接口约定。
- 固定视图重问（不动重问）运行目录：`cja_en/runs/no_move_reask_{smoke,shard0,shard1,perception_shard0,perception_shard1}`，共 5 项、约 2.6 GB（已删除）。生成脚本 `cja_en/scripts/no_move_reask.py` 保留，可在修复后的接口上重跑，但重跑必须写入新的 out-dir。

Git 跟踪文件可从版本历史恢复；未跟踪运行目录如需取证，应从原始服务器归档恢复，但恢复内容仍保持“失效”状态。

新实验必须使用修复后的目标投影、ROI 内检测关联、结构化证据传递和 hybrid/deterministic 回答模式，并满足以下条件：

- 题集自动检查与人工审核分栏记录，只有两者均通过才进入最终测试；
- 冻结题集、源码、模型 revision、有效环境、提示词和随机种子 hash；
- 先报告 chance、按题型 majority 和确定性规则基线，再报告 VLM 与闭环策略；
- 保存逐题预测、目标关联方式、匹配距离、证据字段、动作、预算和终止原因；
- replacement run 使用新目录名，禁止复用旧结果目录或旧汇总文件名。
