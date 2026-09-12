# CJA 英文论文重构工作区

本目录用于独立审阅与重构 DisasterClaw 论文。原 `paper_cja/` 保持只读。

当前状态（2026-09-11）：**独立英文审阅稿已按“救灾智能体＋DisasterClaw 仿真平台＋预算重观测案例”重构并编译为 [main.pdf](main.pdf)**，共 30 页，含八章正文、附录、4 幅正文图、4 张正文表、3 张附录表及 18 条参考文献。VLM 已确认为 Qwen2.5-VL-7B-Instruct。Figure 4 已保留两个控制台截图位置，作者放入指定 PNG 后可自动替换占位框。平台定位为地理配准影像上的仿真与评测框架，不称为高保真飞行模拟器。投稿前仍需归档新实验逐题记录、运行源码和作者信息，**不是可直接提交的最终稿**。

任务逐项状态见 [完成情况](review/task_completion_status.md)。

建议先读 [英文 PDF](main.pdf)，再看 [平台主线方案](review/platform_reframing_plan.md)、[补实验纳入审计](review/supplemental_result_audit.md)、[本轮修订](review/change_log.md)、[内部复审](review/final_reviewer_simulation.md) 与 [投稿清单](review/submission_checklist.md)。旧机制主线见 [历史证据蓝图](review/draft_blueprint.md)，原始数值来源见 [下载结果复核](review/downloaded_results_findings.md)。

`review/source_manifest.json` 保存原论文及相关源码的 SHA-256；不是实验真实性或全量阅读证书。所有新表和定量结论必须在原始结果复核后生成。

## 编译

在 `cja_en/` 内执行：

```sh
make pdf
```

需要 TeX Live 的 `elsarticle`、BibTeX、TikZ/PGFPlots、latexmk。已生成的表和绘图坐标随工程保存，所以仅编译 PDF 不需要下载 `runs/` 或调用模型。当前采用用户允许的 Elsevier 通用 A4 12pt preprint 方案；CJA 专用最新要求仍需终核。

## 从记录重建分析和图表

```sh
make assets
make pdf
python3 scripts/validate_manuscript.py
```

Python 3 和 NumPy 足够。`make assets` 先运行审计，再生成图表；读取项目根目录 `runs/` 下的固定路径。没有自动调用 VLM、训练权重、GPU 或付费 API，也不写入原始运行目录。

- `scripts/audit_downloaded_results.py`：重建 7×5 策略曲线、三档 FOV 和 1600 episode 汇总，输出输入 hash 与算术核对。
- `scripts/build_assets.py`：同一数据上的 ROI bootstrap、共同匹配敏感性、按题型统计和 risk–coverage；输出 `review/manuscript_analysis.json` 及分析资产。当前正文只纳入与新平台主线直接相关的表图。
- `figures/overview.tex`、`figures/observation_ladder.tex`、`figures/agent_loop.tex`：总体架构、三档观测构造和重观测状态机的矢量示意图。
- `figures/console_composite.tex`：Figure 4 的控制台组合图；截图方法和固定文件名见 `review/console_screenshot_guide.md`。
- `figures/budget_curve.tex` 仍为可重建资产，但已从正文撤下，以避免弱化平台与智能体主线。
- `scripts/validate_manuscript.py`：引用/标签/路径、编译日志、PDF文本、输入及原工程指纹核验。
- `review/validation_report.md`：最终技术检查与逐页视觉 QA 范围。

`references.bib` 使用 `sourceurl` 字段保存核验来源，不在印刷参考中重复输出长 URL。没有未核实的 DOI 占位值；RSS/NeurIPS 三条无页字段可产生非致命 BibTeX 提示，详见参考审计。

关键未决项为 Qwen checkpoint revision、有效生成配置、提示词与运行源码快照，以及新实验逐题结果、题集/权重实体的本地归档。匹配预算和不动重问实验已完成，但执行动作仍相差 1 次，且当前工作区只有结果摘要。作者选择不为当前稿新增工程型平台实验；相应项目保留为发布与复现建议，不作为本轮交付前置条件。本文未宣称严格未见事件泛化、在线策略优势、飞行动力学保真度或干净的动作因果效应。归档的 200 指令 VLN B0--B3 消融已作为组件诊断纳入 Appendix A.3，不作为性能增益证据。
