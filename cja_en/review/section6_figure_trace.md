# Section 6 figure trace

日期：2026-09-12。两幅图均由 `scripts/build_section6_figures.py` 从本地已审计离线记录确定性生成；具体输入、输出 SHA-256 与派生计数见 `section6_figure_manifest.json`。

## Figure: view-change outcomes

- Source data: `runs/benchmarks/paper_cja_mech_v1/final_fov/fov_ladder_eval_items.jsonl`.
- Transformation: 对每个固定建筑比较 cruise 与 floor 的 argmax 类别，并按相对真值的变化分为错误到正确、正确到错误、错误到另一错误类；不纳入预测类别未变化的 3,370 栋建筑。
- Supported claim: 观测范围收窄在当前影像与感知管线中同时产生纠正和伤害，不能被视为一致有益的处理。
- Limits: 主底座曾接触评估事件；类别变化来自既有正射影像裁剪与重采样，不是实际 UAV 再成像的因果效应。

## Figure: budget curve

- Source data: `runs/benchmarks/paper_cja_mech_v1/budget_allocation.json`.
- Transformation: 原文件中七种规则、五个预算点的 macro-F1 直接绘图；未插值、未平滑、未选择新的预算点。
- Supported claim: 不同选择规则的相对表现随预算变化，且可部署规则未显示 calibrated entropy 或 expected entropy reduction 对 raw entropy 的一致优势。
- Limits: random 是一次冻结随机分配，不包含随机策略 seed uncertainty；label-informed diagnostic 使用评估标签；一个被选建筑计作一次动作，不表示访问这些建筑的飞行路线成本；纵轴从 0.60 开始并在图注中明示。
