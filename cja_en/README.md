# CJA 英文论文重构工作区

本目录用于独立审阅与重构 DisasterClaw 论文。原 `paper_cja/` 保持只读。

当前状态（2026-09-12）：**独立英文审阅稿已按“救灾智能体＋DisasterClaw 仿真平台＋预算重观测案例”重构并编译为 [main.pdf](main.pdf)**。正文新增两幅由已审计离线记录重建的结果图：预测变化结果分解与完整预算曲线。VLM 已确认为 Qwen2.5-VL-7B-Instruct。控制台组合图仍保留两个截图位置，作者放入指定 PNG 后可自动替换占位框。平台定位为地理配准影像上的仿真与评测框架，不称为高保真飞行模拟器。投稿前仍需归档新实验逐题记录、运行源码和作者信息，**不是可直接提交的最终稿**。

任务逐项状态见 [完成情况](review/task_completion_status.md)。

建议先读 [英文 PDF](main.pdf)，再看 [在线结果失效记录](review/invalidated_online_results.md)、[平台主线方案](review/platform_reframing_plan.md)、[本轮修订](review/change_log.md)、[内部复审](review/final_reviewer_simulation.md) 与 [投稿清单](review/submission_checklist.md)。旧机制主线见 [历史证据蓝图](review/draft_blueprint.md)。

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

Python 3 和 NumPy 足够。`make assets` 只重建当前保留的离线图表；读取项目根目录 `runs/` 下带冻结哈希的固定路径。没有自动调用 VLM、训练权重、GPU 或付费 API，也不写入原始运行目录。

- `scripts/audit_downloaded_results.py` 与 `scripts/build_assets.py` 属历史流程，包含已失效在线实验入口，不再由 Makefile 调用。
- `figures/overview.tex`、`figures/observation_ladder.tex`、`figures/agent_loop.tex`：总体架构、三档观测构造和重观测状态机的矢量示意图。
- `figures/console_composite.tex`：Figure 4 的控制台组合图；截图方法和固定文件名见 `review/console_screenshot_guide.md`。
- `scripts/build_section6_figures.py`：只读取已审计离线记录，重建正文中的预测变化结果分解图和完整预算曲线，并写出输入/输出哈希清单；不调用模型。
- `scripts/validate_manuscript.py`：引用/标签/路径、编译日志、PDF文本、输入及原工程指纹核验。
- `review/validation_report.md`：最终技术检查与逐页视觉 QA 范围。

`references.bib` 使用 `sourceurl` 字段保存核验来源，不在印刷参考中重复输出长 URL。没有未核实的 DOI 占位值；RSS/NeurIPS 三条无页字段可产生非致命 BibTeX 提示，详见参考审计。

关键未决项为修复后 Agent-VQA 的 Qwen checkpoint revision、有效生成配置、提示词与运行源码快照，以及人工审核题集和逐题结果归档。旧在线与不动重问实验已经失效并删除。本文未宣称严格未见事件泛化、在线策略优势、飞行动力学保真度或干净的动作因果效应。归档的 200 指令 VLN B0--B3 消融已作为组件诊断纳入附录，不作为性能增益证据。
