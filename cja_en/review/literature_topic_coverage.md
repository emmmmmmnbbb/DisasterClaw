# 22 个主题的定向检索覆盖记录

更新：2026-09-06。沿用 ARS 的来源核验与论断对齐流程。以下记录本会话实际执行的扩展检索及其用途；不是系统综述筛选流程，不报告虚构命中数、纳入数或新颖性认证。检索覆盖不等于每篇候选全文已读。

## 扩展查询

查询在 2026-09-05 至 09-06 的连续工作中执行，优先 arXiv、ACL Anthology、CVF、PMLR、RSS、OpenReview 及出版社。

- Q1：`"embodied intelligence" "embodied AI" review`（Nature、Science、arXiv）。
- Q2：`"UAV embodied intelligence" "autonomous aerial vehicles"`。
- Q3：`"UAV agents" "LLM" "VLM" aerial`（arXiv、CVF、IEEE）。
- Q4：`"LLM-based agents" "tool" ReAct SayCan`（arXiv、OpenReview、PMLR）。
- Q5：`vision-language navigation aerial VLN OpenFly`（arXiv、CVF）。
- Q6：`vision-language-action models autonomous navigation UAV AutoFly`（arXiv、OpenReview）。
- Q7：`task planning hierarchical planning UAV CityNavAgent`（ACL、arXiv）。
- Q8：`multimodal reasoning spatial reasoning aerial STMR`（arXiv、CVF）。
- Q9：`UAV disaster response remote sensing agents RescueADI`（arXiv、IEEE、Nature）。
- Q10：`closed-loop perception planning action online replanning UAV LLM`（arXiv、ScienceDirect、RSS）。
- Q11：`foundation models aerospace autonomy UAV LLM VLM "Chinese Journal of Aeronautics"`（ScienceDirect、航空学报网站）。
- Q12：`"CityEQA" benchmark embodied question answering`（ACL、arXiv）。

## 对照用户指定的主题

1. Embodied intelligence：Q1；限定观测—动作反馈定义，不把框架搭建视为原创。
2. Embodied AI：Q1、Q12；城市主动问答为直接反例。
3. UAV embodied intelligence：Q2、Q3；AerialClaw、城市具身任务。
4. Autonomous aerial vehicles：Q2、Q6；区分飞行控制与理想化传感器动作。
5. UAV agents：Q3；模块化航空智能体。
6. LLM-based agents：Q4；工具接口不自动构成新算法。
7. LLM for UAV：Q3、Q7、Q11；CityNavAgent、AerialClaw。
8. VLM for UAV：Q3、Q6、Q8；视觉输入与动作生成。
9. Vision-language navigation：Q5；经典 VLN 及其航空扩展。
10. Aerial VLN：Q5；AerialVLN、CityNav、OpenFly。
11. Vision-language-action models：Q6；AutoFly 等候选，未直接排名。
12. Autonomous navigation：Q2、Q5、Q6；导航成功不等于灾情回答正确。
13. Task planning：Q7、Q10；CityNavAgent、TAMPURA。
14. Hierarchical planning：Q7、Q12；CityNavAgent、CityEQA。
15. Tool-using agents：Q4、Q9；RescueADI、AerialClaw。
16. Multimodal reasoning：Q8；输入、结构化证据与预测能力分开。
17. Spatial reasoning：Q8、Q12；STMR、CityEQA。
18. UAV disaster response：Q9；不把单幅影像分析等同实际救援执行。
19. Remote sensing agents：Q9；RescueADI 直接限定灾情工具分析的新颖性。
20. Closed-loop perception–planning–action：Q10、Q12；AerialClaw、CityEQA。
21. Online replanning：Q10；RePLan 等发现候选，不据摘要宣称全面比较。
22. Foundation models for aerospace autonomy：Q11、Q3、Q6；航空语言/VLA 工作与 CJA 专项检索。

## 决定性近邻及阅读边界

**CityEQA**：正式 EMNLP 2025 题名、十作者、页码与 DOI 已按 [ACL 官方条目](https://aclanthology.org/2025.emnlp-main.630/) 核实。方法与实验比较使用 [作者 arXiv v1 正文](https://arxiv.org/html/2502.12532v1) 的 §2–4、§6–7 及附录相关部分；正式出版 PDF 抓取失败，未冒称逐页读完出版版。Collector 已研究视角调整与回答变化，因此本稿只保留固定建筑预算的纠正/破坏分析及灾情任务诊断这一受限区别。不将不同评分体系的结果放进同一排名。

**AerialClaw**：已阅读 [作者正文](https://arxiv.org/html/2606.12142v1) 的主要架构、技能接口、运行适配与示例部分。它限制“通用闭环框架”的原创声明；代码继承关系仍须本项目来源证据。

**RescueADI**：已阅读 [作者正文](https://arxiv.org/html/2410.13384v1) 的任务定义、数据、方法与实验部分。输入图像上的工具分析与改变相机输入的动作需要区分；这一差别是任务比较，不是宣称后者首次出现。

**AutoFly**：已检索 [ICLR 2026 官方 PDF](https://openreview.net/pdf?id=88RKxlFUNY) 与 [arXiv](https://arxiv.org/abs/2602.09657)。用于 VLA/连续导航的范围筛选；未成为本文性能基线，也不为凑数进入 bib。

其他保留引用的来源、版本与支持范围见 [reference_audit.md](reference_audit.md)。近期候选包括 AerialVLA、WorldFly、LookasideVLN 与 RePLan；发现候选不代表其方法已充分评审。

## CJA 专项检索与限制

已核验 CJA 的 LLM 多智能体鲁棒性样例，见 [出版社记录](https://www.sciencedirect.com/science/article/pii/S1000936125003851)。其多智能体安全论证不能转移到本稿。航空软件测试、固定翼建模等检索候选与本研究端点不同，不强加引用。当前未穷尽用户列出的所有期刊/会议。

[SciOpen 的 CJA 页面](https://www.sciopen.com/journal/join_journal/copyright_agreement?id=1808700965752745985&issn=1000-9361) 仅指回 ScienceDirect Guide for Authors，没有提供可独立读取的详细规范。因此最新 CJA 格式终核仍未完成。
