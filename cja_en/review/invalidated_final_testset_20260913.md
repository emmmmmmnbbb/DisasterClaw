# 2026-09-13 二分类 final 题集失效记录

`backend/data/benchmarks/agent_vqa_final_v2_binary.json` 不再具有未触碰 final 测试集资格。

原因：该文件的 160 题与 `agent_vqa_v2_binary.json` 使用完全相同的 43 个 tile。后者已经用于 2026-09-13 ChangeOS 闭环结果分析、题型弱点定位和下一版任务条件化策略设计。两份题集虽然起点等配置可能不同，但不能再把相同地理样本用于新方法的确认性测试。

处置：

- 保留原文件及 SHA-256，不覆盖或伪装成新题集；
- 本轮 160 题和同 tile 的历史 `final` 均只作为 development/diagnostic 数据；
- 将全部 43 个 tile 纳入消费登记的排除集合；
- 新 final 必须从登记表未出现的 tile/ROI 生成，并在任何新策略运行前完成审核和冻结。

这项失效只涉及确认性测试资格，不否定已经完成的诊断运行及其“复观测同时产生纠正和破坏”的观察结果。
