# CJA 英文论文重构工作区

本目录用于独立审阅与重构 DisasterClaw 论文。原 `paper_cja/` 保持只读。

当前状态（2026-09-18）：[main.pdf](main.pdf) 已按“地图锚定灾害世界＋程序化任务 Benchmark＋可追踪闭环评测”修订。新增独立的第 4 节 Benchmark 构建，接入 `figure-paper/` 四张用户提供的图片，保留 camera ladder 与 agent state machine。原有 160 题、44 ROI、三事件、七策略与 3,360 episode 的实验数值及统计边界不变；没有增加物理仿真实验。图像来源映射和双人审核原始记录仍需补充，**不是可直接提交的最终稿**。

本轮改动、代码依据和待确认事项见 [平台与 Benchmark 修订说明](review/platform_benchmark_revision_20260918.md)。

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
- `figure-paper/`：本轮提供的总流程、地图嵌入、任务类型与平台闭环四张 PNG/SVG；正文引用 PNG，原图未改写。
- `figures/observation_ladder.tex`、`figures/agent_loop.tex`：正文保留的三档观测与状态机矢量图；旧 `overview.tex`、`console_composite.tex` 保留作历史素材，不再由正文引用。
- `sections/04_benchmark_construction.tex`：独立 Benchmark 章节；原方法和实验章节自动顺延编号。
- `scripts/build_section6_figures.py`：只读取已审计离线记录，重建正文中的预测变化结果分解图和完整预算曲线，并写出输入/输出哈希清单；不调用模型。
- `scripts/validate_manuscript.py`：引用/标签/路径、编译日志、PDF文本、输入及原工程指纹核验。
- `review/validation_report.md`：最终技术检查与逐页视觉 QA 范围。

`references.bib` 使用 `sourceurl` 字段保存核验来源，不在印刷参考中重复输出长 URL。没有未核实的 DOI 占位值；RSS/NeurIPS 三条无页字段可产生非致命 BibTeX 提示，详见参考审计。

当前冻结协议见 `review/changeos_p6_final_protocol_20260915.json`。本地协议列出一个审核报告的路径和哈希，未单独列出两份独立审核表及分歧仲裁日志；完整发布前还需补充这些材料、图中 tile/题目映射及可获取的影像/权重/运行环境。正文保留 T1 的观察样本数值优势，但不宣称其相对 HOLD 的稳定优势、严格未见事件泛化、飞行动力学保真度或纯动作因果效应。归档 VLN 与四分类结果仅作为历史诊断。
