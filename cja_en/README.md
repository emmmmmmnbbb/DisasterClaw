# CJA 英文论文重构工作区

本目录用于独立审阅与重构 DisasterClaw 论文。原 `paper_cja/` 保持只读。

当前状态（2026-09-06）：**独立英文审阅稿已完成并编译为 [main.pdf](main.pdf)**，含七章正文、附录、三幅矢量图、六张生成表及 18 条参考文献。已完成内部复审并落实一轮修改。投稿前科学依赖和作者信息尚未齐备，**不是可直接提交的最终稿**。

任务逐项状态见 [完成情况](review/task_completion_status.md)。

建议先读 [英文 PDF](main.pdf)，再看 [本轮修订](review/change_log.md)、[内部复审](review/final_reviewer_simulation.md) 与 [投稿清单](review/submission_checklist.md)。实际重构方向见 [证据蓝图](review/draft_blueprint.md)，数值来源见 [下载结果复核](review/downloaded_results_findings.md)。

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
- `scripts/build_assets.py`：同一数据上的 ROI bootstrap、共同匹配敏感性、按题型统计和 risk–coverage；输出 `review/manuscript_analysis.json`、六张表与两幅数值图。置信阈值是事后诊断，不是部署策略。
- `figures/overview.tex`：观察模型与两层评测的矢量示意图。
- `scripts/validate_manuscript.py`：引用/标签/路径、编译日志、PDF文本、输入及原工程指纹核验。
- `review/validation_report.md`：最终技术检查与逐页视觉 QA 范围。

`references.bib` 使用 `sourceurl` 字段保存核验来源，不在印刷参考中重复输出长 URL。没有未核实的 DOI 占位值；RSS/NeurIPS 三条无页字段可产生非致命 BibTeX 提示，详见参考审计。

关键未决项为实际 VLM/提示词与运行源码快照、题集/权重实体和预算匹配复验。本文未宣称严格未见事件泛化，也未将现存在线数据解释为干净的动作因果消融。
