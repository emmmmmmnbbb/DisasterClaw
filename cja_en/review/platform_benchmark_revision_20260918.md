# 平台与 Benchmark 修订说明（2026-09-18）

## 本轮范围

依据根目录 `DisasterClaw_平台包装与Benchmark构建修改方案.md` 与 `cja_en/figure-paper/` 四组图片修改英文 LaTeX 和 PDF。按学术修订流程区分实现事实、作者提供的流程说明和待补证据；按 PDF 工作流编译与检查版式。未运行模型、改动实验代码、重算结果或添加物理仿真。

## 已实施

- 标题、摘要、引言四项贡献及结论统一到“真实灾害数据 → 地理世界 → 任务 Benchmark → 闭环评测”。
- 第 3 节重组为设计目标、地图锚定世界、可控观测、交互接口与执行记录。
- 新增第 4 节：数据与候选 ROI、事件划分/ROI、程序化生成、自动质检、双人复核与冻结；原方法、协议、结果章节顺延为第 5–7 节。
- 增加 Benchmark 摘要与四类任务证据需求表；区分 160 道题、44 个 ROI 和 3,360 条策略运行。
- 接入四张新图，保留原 camera ladder 与 state machine，原始 PNG/SVG 不修改。旧架构图及占位控制台源文件保留，但不再引用。
- 保留所有原有定量结果、主要比较的 98.75% 区间和 T1 相对 HOLD 优势未确立的结论。
- 核对并修正 ThinkGeo 的占位作者，依据 [arXiv v1 官方作者页](https://arxiv.org/abs/2505.23752v1)。

## 代码依据与关键澄清

- `scripts/benchmarks/build_roi_index.py`：0.80 是巡航视野的 post-disaster 影像覆盖率门槛，不是建筑覆盖率或 ROI 损伤比例；正文明确这是默认门槛，不擅自断言缺失生成记录的最终参数。
- `scripts/benchmarks/gen_agent_vqa_testset_v2.py`：合格事件/地理信息/标签过滤，consumed registry 排除、种子洗牌和事件轮转；每个 tile 按前提生成至多四类题。
- 同一生成器：binary 模式合并 minor/major/destroyed；建筑归属按质心在 ROI 的闭区间判断；presence 为受损存在性，damage 为指定建筑二分类，count 为 0/1/2/3+，spatial 为相对 ROI 中心最近受损建筑的八方向。
- 不声称每类恰好 40 题、不声称方向均衡，也不将程序确定的答案描述为 LLM 生成。
- `scripts/benchmarks/review_agent_vqa_testset.py` 与冻结协议支持字段/答案/标签模式/划分/重复等检查的描述；自动检查不等于视觉可回答性保证。
- `review/changeos_p6_final_protocol_20260915.json` 提供 160/44/三事件/七策略/三重复/种子/哈希定义。协议 SHA-256 为 `41ccbb5fed5e8ec95be36b20e7dd43693cc237fb2d1e63246815638a4e83ed2c`，原文件未修改。

## 图片使用边界与作者待确认

1. 总流程、地图嵌入和任务面板缺少可核验的 tile ID / 最终 qid 对照。正文作为流程与构造示例使用，不编造 question、reference answer 或图中颜色含义。建议后续补充每张图对应的 tile、任务记录和 marker legend。
2. 平台组合图显示 30 m、YOLO/SegFormer、VLN 等历史界面；图注与 ChangeOS headless / camera ladder 定量协议明确分开。组合图不描述为同一 episode 的连续轨迹。其模块来源请作者确认。
3. 方案和原稿陈述“两名标注者独立复核、仲裁后冻结”，本轮保留。现有协议只列一个 `review_report`，当前工作区未找到分别对应两名审核者及仲裁的原始记录，故删除“这些记录已逐一哈希核验/本地包已包含”的无据表述。请提供文件路径并链接最终 qid；本轮不将双审流程认证为已独立核验。
4. 本轮使用原图，不作生成式重绘，也未将旧四分类样例冒充最终二分类样例。若要把任务构造图升级为真实 final-set QA 图，仍需明确可追溯的题目数据。

## 验证

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` 成功，31 页。LaTeX 无 warning、overfull/underfull、未解析引用；BibTeX 保留两项原有缺页码提示（`curtis2024tampura`、`ovadia2019can`）。
- `git diff --check` 通过；主结果章节仅新增证据需求表和解释段落，原结果数字与区间没有改写。
- 31 页完成缩略联系表版式检查，8、9、10、12、13、14、15 页用 Poppler 120 dpi 单页查看；存在性公式的字体问题已修正并重新渲染。原图中的细小标签、控制台文字及 Fig. 2 箭头旁原始碎片仍是后续图片优化项，不将本轮称作最终制图验收。
- `scripts/validate_manuscript.py` 的引用、标签、输入路径和保留的两份离线结果哈希检查通过；完整脚本因历史 `source_manifest.json` 与现有后端源码的指纹差异返回 1。这些后端文件不是本轮改动，不更新旧基线或更改实验源码来使检查通过。详见 `technical_validation.json`。
- 最终 PDF SHA-256：`e465a620f32e62444307751c46b8c766539c506feea4d984b34fbff9cb46092f`；视觉记录见 `visual_qa.json`。未独立重跑模型、全面核验所有文献或认证人工审核。
