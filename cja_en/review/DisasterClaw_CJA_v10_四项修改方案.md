# DisasterClaw（main-v10）CJA 投稿前四项修改方案

> 目标：基于当前 `main-v10.pdf`，先完成以下四项修改：
> 1. 补充 AeroVerse、BEDI、OpenUAV/UAV-Need-Help、ThinkGeo、DisasterM3，并重新明确 DisasterClaw 的差异化定位；
> 2. 将 160 个 final questions 的“双标注者独立复核”按已完成处理，修改实验协议与 limitation；
> 3. 增加一项最小规模的 PX4 SITL + Gazebo action-mapping validation；
> 4. 清除 Figure 4 placeholder、binary/four-class 主文不一致、`Appendix Appendix`、author-review draft 等投稿残留。
>
> 原则：不改变论文目前已经形成的主线——**task-conditioned observation value / closed-loop measurement**。不把文章改写成高保真飞行仿真论文，也不扩大当前实验能够支持的 claim。

---

## 0. 建议保持不变的核心定位

当前版本最值得保留的科学问题是：

> **Aerial disaster inspection requires deciding whether another observation will help answer the current question, rather than merely whether it will reduce local perception uncertainty.**

后续四项修改都围绕这一点展开。新增文献用于明确“与已有 aerial embodied benchmark / remote-sensing agent benchmark 的区别”；双标注用于加强 final task set 的可信度；PX4/Gazebo 只验证抽象动作能够映射到常规飞控接口，不把它包装成真实飞行验证。

---

# 1. Related Work：补 AeroVerse / BEDI / OpenUAV / ThinkGeo / DisasterM3

## 1.1 需要新增的参考文献

建议加入以下 5 项。最终 BibTeX 请以官方论文页/期刊页为准，不要手工固定参考文献编号，由 LaTeX 自动重排。

1. **AeroVerse: UAV-Agent Benchmark Suite for Simulating, Pre-training, Finetuning, and Evaluating Aerospace Embodied World Models**. Fanglong Yao et al., 2024, arXiv:2408.15511.
2. **BEDI: A Comprehensive Benchmark for Evaluating Embodied Agents on UAVs**. Mingning Guo et al.；正式期刊版发表于 *ISPRS Journal of Photogrammetry and Remote Sensing*, 2026, Vol. 232, pp. 910–936, DOI: 10.1016/j.isprsjprs.2026.01.013.
3. **Towards Realistic UAV Vision-Language Navigation: Platform, Benchmark, and Methodology**. Xiangyu Wang et al., 2024, arXiv:2410.07087. 该工作提出 OpenUAV 与 UAV-Need-Help。
4. **ThinkGeo: Evaluating Tool-Augmented Agents for Remote Sensing Tasks**. 2025, arXiv:2505.23752.
5. **DisasterM3: A Remote Sensing Vision-Language Dataset for Disaster Damage Assessment and Response**. Junjue Wang et al., NeurIPS 2025 / arXiv:2505.21089.

## 1.2 Introduction：增加一句“我们和这些工作的区别”

### 插入位置

Section 1 Introduction，在当前这句之后：

> The main experiment then compares frozen online policies on a separate, human-reviewed question set, scoring complete answers rather than classifier outputs alone.

在 Contributions 之前插入下面一段。

### 建议直接新增英文

```text
Recent aerial embodied benchmarks emphasize broad skill coverage, realistic navigation, or standardized perception–decision–action evaluation, while recent remote-sensing agent benchmarks emphasize multimodal reasoning and tool use over large image collections. DisasterClaw targets a narrower complementary question: given a georeferenced disaster scene and a task-specific evidence requirement, can an executed observation action be traced to a change in perception evidence and ultimately to a corrected or harmed terminal answer? The contribution is therefore not broader environment realism or a new foundation model, but a controlled measurement interface for task-dependent observation value in disaster inspection.
```

这段的作用是提前防止审稿人把你与 AeroVerse/BEDI/OpenUAV 直接按“谁的 simulator 更真实”比较。

---

## 1.3 Section 2.1：补 aerial embodied benchmark 相关工作

### 当前问题

当前 2.1 已经包含 AerialVLN、CityNav、OpenFly、CityNavAgent、AerialClaw、CityEQA，但没有覆盖近年的综合 UAV embodied benchmark。建议保留原段落，在 CityEQA 段之后新增一段。

### 建议新增英文

```text
Recent work has expanded aerial embodied intelligence beyond instruction-following navigation. AeroVerse organizes UAV-agent evaluation around scene awareness, spatial reasoning, navigational exploration, task planning, and motion decision, providing a broad benchmark suite for aerospace embodied world models. BEDI further formalizes UAV embodied-agent evaluation through a perception–decision–action chain and evaluates semantic and spatial perception, motion control, tool use, planning, and action generation across real and virtual settings. OpenUAV and the UAV-Need-Help benchmark emphasize realistic aerial trajectory execution and assistant-guided object search in three-dimensional environments. These platforms address general aerial embodiment, realistic navigation, and standardized skill evaluation. DisasterClaw is complementary rather than competing in simulator fidelity: it uses georeferenced pre-/post-disaster imagery to make the task value of a controlled view change directly observable at the final disaster-inspection answer.
```

### 建议修改 2.1 最后一两句

当前类似：

> DisasterClaw differs in task domain, source imagery, action abstraction, and its paired damage endpoint.

建议扩展为：

```text
DisasterClaw differs in task domain, source imagery, action abstraction, and evaluation target. Rather than benchmarking general-purpose flight skills or high-fidelity navigation, it couples a georeferenced disaster scene to a controlled observation transition and measures whether that transition changes task-relevant evidence and the terminal inspection answer. This narrower scope motivates reporting observation construction, action execution, and task-level correction or harm with equal precision.
```

---

## 1.4 Section 2.2：补 ThinkGeo 和 DisasterM3

### 插入位置

在当前 xBD + RescueADI 段落之后，即下面这句话后：

> Thus, paired damage recognition and agentic disaster analysis both predate this work.

### 建议直接新增英文

```text
Recent remote-sensing resources further broaden the evaluation of multimodal reasoning and agentic analysis. ThinkGeo evaluates tool-augmented language-model agents on multi-step remote-sensing tasks, including disaster assessment and change analysis, and scores both intermediate tool execution and final answers. DisasterM3 provides a large-scale multi-hazard, multi-sensor vision-language benchmark for disaster damage assessment and response, covering perception and reasoning across diverse disaster events. These resources strengthen static-image reasoning, multimodal generalization, and structured tool use. DisasterClaw addresses a different axis: the observation itself is an environment variable, so the agent can execute a view-changing action and the benchmark can measure whether the newly acquired evidence improves or harms the final task answer.
```

### 2.2 后半段建议保留

你目前关于 orthographic imagery 的边界说明是合理的，不要删除：

```text
Cropping and resampling an orthographic product can alter the number of pixels assigned to a target, but it cannot synthesize unrecorded viewpoints or detail below the native source scale.
```

这恰好可以与 OpenUAV/BEDI 的高保真三维环境形成清晰区别。

---

# 2. 160 个 final questions：按“双标注者独立复核已完成”修改

> 本方案按你的要求视为已经完成双标注。没有提供具体 pre-adjudication agreement 或 Cohen's kappa，因此正文不虚构数字。如果你实际算过 agreement / kappa，可以再加一句具体数值。

## 2.1 Abstract

当前：

```text
We then evaluate seven frozen online policies on 160 human-reviewed questions from 44 regions in three disaster events, with three generation repeats (3,360 episodes).
```

建议改为：

```text
We then evaluate seven frozen online policies on 160 questions independently reviewed by two annotators from 44 regions in three disaster events, with three generation repeats (3,360 episodes).
```

如果觉得摘要太长，也可以写：

```text
We then evaluate seven frozen online policies on a dual-reviewed 160-question set from 44 regions in three disaster events, with three generation repeats (3,360 episodes).
```

优先推荐第一种，表达更自然。

---

## 2.2 Section 5.1：整段替换

### 当前需要删除的内容

删除当前这几句：

```text
The author manually reviewed all 160 and approved the frozen set; 88 previously flagged ambiguity notices remain recorded rather than erased. The present record documents one author’s review, not two independent annotators or an inter-rater agreement estimate.
```

### 建议替换为

```text
Two annotators independently reviewed all 160 final questions before the online experiment. The review checked task type, target ROI, reference answer, and whether the question was answerable from the registered imagery and associated ground-truth evidence. Previously flagged ambiguity notices were retained in the audit trail rather than deleted, and disagreements were resolved before the final set was frozen. The final question set, dual-review record, adjudication log, binary-label manifest, and source/prompt/weight hashes were checked before the first online episode. The exact identities and checksums are listed in Appendix A.
```

### 如果你有 agreement / kappa，再在第三句后加入（可选）

```text
Before adjudication, the two annotators agreed on [XX.X%] of the reviewed items (Cohen’s κ = [X.XX]).
```

没有实际计算就不要加。

---

## 2.3 Section 7 Limitations：删除“单人标注”弱点

### 当前相关文字

当前写法大意为：

```text
Although all 160 questions received the author’s manual approval before online testing, this is not a two-annotator independent review or an agreement study, and 88 recorded ambiguity notices remain part of the audit trail. Answer frequencies and template families may still permit language-only shortcuts ...
```

### 建议替换为

```text
All 160 final questions were independently reviewed by two annotators and adjudicated before the set was frozen. This reduces reference-answer and task-scope ambiguity, but it does not eliminate possible dataset artifacts. Answer frequencies and template families may still permit language-only shortcuts, and the present study does not include a separate language-only evaluation on the frozen final set.
```

这样保留真正存在的 limitation，而不再保留已经解决的问题。

---

## 2.4 Appendix A：建议补一个非常短的 review provenance 说明

在 `Appendix A.1. Frozen final identities and audit boundary` 中，第一段后增加：

```text
The final task package also contains the two independent review records and the adjudication record used to freeze the 160-question set. Review artifacts are linked to the frozen question identifiers and are retained as provenance rather than being used as model inputs.
```

如果不希望公开标注者身份，可以只公开匿名 reviewer ID。

---

# 3. 增加 PX4 SITL + Gazebo action-mapping validation

## 3.1 目的：只证明“动作能映射”，不要把它写成“真实飞行验证”

建议选 **PX4 SITL + Gazebo**，而不是额外重做一套 AirSim 图像环境。理由是本实验只需要回答一个审稿问题：

> DisasterClaw 的 `fly / center / descend / hover` 是否只是抽象符号，还是能够映射为常规无人机飞控可执行的 setpoint command？

这个补实验**不参与 3,360 个主实验 episode 的成像和评分**，也不需要重新跑全部 VQA。

### 最小实验设计

从现有 final-policy traces 中抽取代表性的运动序列，例如：

- horizontal `fly/center`；
- vertical `descend`；
- `center + descend` composed motion；
- `hover`。

建议总量 **30–50 条 action sequences** 即可，并覆盖不同位移尺度。每条动作：

1. 将 DisasterClaw 本地 ENU/NED 位移转换为 PX4 position setpoint；
2. 在 PX4 SITL + Gazebo 中从标准初始状态执行；
3. 记录是否成功到达目标；
4. 记录终点水平误差、垂直误差和执行时间。

建议预先定义成功条件，例如：

```text
horizontal endpoint error <= 2 m
vertical endpoint error <= 1 m
```

如果你实际系统采用别的阈值，以实际预先设定值为准，不要为了结果再调。

### 建议指标

```text
Action success rate
Horizontal endpoint error (m)
Vertical endpoint error (m)
Execution time (s)
```

最好按 `horizontal / descend / composed` 三类分别报告，再给 overall。

---

## 3.2 Section 3.4：增加动作映射说明

在当前 3.4 最后一段（说明 simplified action model 的段落）后新增：

```text
To test whether the abstract motion primitives can be mapped to a conventional UAV control interface, we additionally implement an external action bridge to PX4 SITL with Gazebo. DisasterClaw fly/center, descend, and hover commands are converted from the local geographic frame into position setpoints for the simulated autopilot. This bridge is used only for command-level correspondence testing: Gazebo imagery is not used by the main benchmark, and the external simulator does not replace the simplified DisasterClaw world model used in the 3,360 online episodes.
```

这段非常重要：提前限制 claim。

---

## 3.3 Experimental Protocol：新增 subsection

建议在 5.2 之后增加：

### `5.3. External action-mapping validation`

直接写：

```text
We perform a separate command-level validation using PX4 SITL and Gazebo. Representative movement sequences are sampled from the frozen DisasterClaw policy traces and stratified into horizontal centering/fly actions, vertical descent actions, and composed center-plus-descent sequences. Each DisasterClaw motion is translated into a position setpoint in the simulator frame and executed from a standardized initial state. We record command completion, horizontal endpoint error, vertical endpoint error, and simulated execution time. An execution is counted as successful when the final horizontal and vertical errors satisfy the predeclared tolerances of [XX] m and [XX] m, respectively. This experiment tests action-interface correspondence only; it does not validate the orthographic observation renderer or claim physical-flight fidelity.
```

把 `[XX]` 替换成你实际预先采用的阈值。

---

## 3.4 Results：新增 subsection

建议在当前 6.3 后增加：

### `6.4. External action correspondence`

结果出来后直接按下面模板填数字：

```text
The PX4 SITL/Gazebo validation included [N] representative DisasterClaw movement sequences: [N1] horizontal, [N2] descent, and [N3] composed actions. [M]/[N] sequences satisfied the predeclared endpoint tolerances, corresponding to an action-mapping success rate of [XX.X]%. The median horizontal endpoint error was [X.XX] m (IQR [X.XX–X.XX] m), and the median vertical endpoint error was [X.XX] m (IQR [X.XX–X.XX] m). The result shows that the high-level movement primitives used by DisasterClaw can be translated into executable position commands in a conventional autopilot/simulator stack. It does not establish that the simplified DisasterClaw trajectories reproduce full UAV dynamics, nor does it validate the rendered disaster imagery as a physical camera model.
```

### 建议增加一个小表

```text
Table X. PX4 SITL/Gazebo command-level mapping validation.

Action type        n     Success (%)   Horizontal error (m)   Vertical error (m)   Execution time (s)
Horizontal         ...   ...           ...                    ...                  ...
Descent            ...   ...           ...                    ...                  ...
Center + descent   ...   ...           ...                    ...                  ...
Overall            ...   ...           ...                    ...                  ...
```

不要把这个表与 T1_TASK accuracy 做成同一张表，两者回答的是不同问题。

---

## 3.5 Section 7 Limitations：替换现有 “No PX4/Gazebo ...”

当前：

```text
No PX4/Gazebo or physical-flight action mapping was performed. The study therefore concerns imagery-grounded active observation, not operational flight performance.
```

建议改为：

```text
The added PX4 SITL/Gazebo experiment verifies command-level mapping for representative movement primitives, but the main benchmark still uses simplified kinematics and orthographic imagery. The external simulator is not used to generate the evaluated disaster observations, and no hardware-in-the-loop or physical-flight validation is reported. The study therefore remains an imagery-grounded active-observation evaluation rather than a measurement of operational flight performance.
```

前面关于 wind、attitude dynamics、battery、collision 等 limitation 保留。

---

## 3.6 Conclusion：微调最后几句

当前最后部分提到：

```text
Claims about real flight additionally require external action and imaging validation.
```

做完 PX4/Gazebo 后建议改为：

```text
A separate PX4 SITL/Gazebo check provides command-level correspondence for representative movement primitives, while claims about physical sensing and flight performance still require external imaging validation and real or hardware-in-the-loop flight experiments. Until then, the contribution remains an imagery-grounded environment and a measured policy comparison with explicitly bounded evidence, rather than a demonstrated general-purpose flight policy.
```

---

# 4. 投稿级清理：逐项具体修改

## 4.1 删除 Page 1 的内部 review-status 段

### 当前应整段删除

```text
Review status. This is an author-review draft, not a submission-ready release. The main online results use the frozen three-event ChangeOS protocol and audited episode records; older four-class and navigation diagnostics are segregated in the appendix. Author details, release packaging, and the remaining provenance items in Appendix Appendix A require confirmation.
```

正式投稿稿不应该出现这一段。

同时把：

```text
Author names to be confirmed
Affiliations and corresponding-author details to be confirmed,
```

替换为真实作者和单位信息。

---

## 4.2 Table 1：修正 binary/four-class 主文不一致

### 当前错误

```text
Perception and models
Building localization and four-class damage evidence, spatial summaries, semantic map, planner, and Qwen answer generation
```

这是旧版 xview2_first 的残留，和当前主实验 binary ChangeOS 不一致。

### 建议改成

```text
Perception and models
Building localization and binary no-damage/damaged evidence, geographic cross-view evidence memory, task-specific spatial summaries, and Qwen answer generation
```

### Boundary 列也建议同步改

当前类似：

```text
Modular system integration; model revision and some archived runtime fields remain incomplete
```

建议改成：

```text
Fixed ChangeOS backend with prior xBD exposure; fixed Qwen2.5-VL-7B-Instruct answer model; no claim of event-disjoint perception generalization
```

---

## 4.3 Table 1：Logging and evaluation boundary 已经过时

当前表中仍保留类似：

```text
New online records are locally represented by supplied summaries; screenshots illustrate function only
```

但现在已经有 3,360 episode 的 audited records，这句话必须删除。

建议改为：

```text
All 3,360 final online episodes are covered by local shard-level consistency audits; complete public release packaging remains pending
```

---

## 4.4 Figure 1 内部文字同步改 binary protocol

Figure 1 当前 Perception and models 框里仍有：

```text
damage/spatial evidence, semantic map, Qwen answers
```

建议改为：

```text
binary damage/spatial evidence, geographic memory, Qwen answers
```

如果 final experiment 中 `semantic map` 不参与七策略主比较，就不要继续放在主图中心位置。

---

## 4.5 Figure 4 placeholder：建议直接从主文删除，而不是继续占篇幅

当前 Page 12 仍有：

```text
Screenshot placeholder (a)
Save the operator-console overview as ...
Screenshot placeholder (b)
Save the closed-loop inspection view as ...
```

这些必须全部删除。

### 推荐方案：主文彻底删除 Figure 4

原因：它本来就被定义为 interface documentation，不提供 quantitative evidence。当前论文现在已经有更重要的 T1_TASK、统计区间和 PX4/Gazebo 验证，主文没有必要继续占一整幅 console screenshot。

因此：

1. 删除 Page 12 整个 Figure 4 placeholder + caption；
2. Section 3.5 当前文字：

```text
Figure 4 illustrates these functions after the agent loop has been defined. The screenshots support interface description and diagnosis; they do not supply quantitative evidence.
```

改为：

```text
The operator console supports qualitative inspection of the same scene state, perception evidence, decisions, actions, budgets, and event logs used by the headless benchmark. The interface is intended for monitoring and diagnosis; all quantitative results in this study are produced by the frozen headless protocol.
```

如果确实希望保留截图，建议移到 Supplementary Material，而不是主结果部分。

---

## 4.6 修正所有 `Appendix Appendix`

当前至少有以下 3 处：

```text
Appendix Appendix A
Appendix Appendix A
Appendix Appendix B
```

全部改成：

```text
Appendix A
Appendix A
Appendix B
```

另外搜索 LaTeX 源码是否存在类似：

```latex
Appendix~\ref{...}
```

但 label 本身自动生成了 `Appendix`，导致双写。建议统一成：

```latex
Appendix~\ref{...}
```

或：

```latex
\ref{...}
```

取决于你当前 appendix macro 的输出，不要在正文和 macro 两边重复写 `Appendix`。

---

## 4.7 Generative AI assistance：删除 “author-review draft” 语气

当前：

```text
Generative AI assistance. Codex was used to assist manuscript restructuring, English drafting, reference checking, and preparation of analysis and typesetting scripts. The present file is an author-review draft. The responsible authors must verify the final content and supply a disclosure consistent with the journal’s current policy; this text does not assert that their verification has already occurred.
```

正式稿建议改为：

```text
Generative AI assistance. Generative AI tools were used to assist manuscript restructuring, English-language editing, reference checking, and preparation of analysis and typesetting scripts. All scientific claims, experimental design, reported results, and final manuscript content were reviewed and approved by the authors, who take full responsibility for the work.
```

投稿前再按 CJA/Elsevier 当时最新的 AI disclosure 模板做一次最终格式核对。

---

## 4.8 Declarations：不能保留 “to be supplied / to be confirmed”

当前：

```text
Acknowledgements and funding. To be supplied by the authors ...
CRediT author statement. To be assigned ...
Declaration of competing interest. To be confirmed ...
```

正式投稿前全部填实。

如果没有利益冲突，可用：

```text
Declaration of competing interest. The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.
```

Funding 和 CRediT 必须根据真实项目和作者贡献填写，不要编造。

---

# 5. 建议修改后的 Related Work 逻辑结构

修改后 Section 2 建议形成以下层次：

```text
2.1 Aerial embodied agents and simulation platforms
    AerialVLN / CityNav / OpenFly
    CityNavAgent / AerialClaw / CityEQA
    AeroVerse / BEDI / OpenUAV + UAV-Need-Help
    → DisasterClaw：不竞争 simulator fidelity，而强调 georeferenced task-value measurement

2.2 Disaster interpretation and remote-sensing agents
    xBD / RescueADI
    ThinkGeo / DisasterM3
    → DisasterClaw：environment action 会改变 observation footprint，并测最终 answer correction/harm

2.3 Active perception, uncertainty, and task value
    保持当前结构
    → 从 uncertainty reduction 引到 task-conditioned observation value
```

这样 Related Work 会从“列举相关论文”变成三个清楚的差异轴：

```text
Aerial embodiment / flight realism
        ↓
Remote-sensing agent reasoning / disaster data
        ↓
Active perception / observation value
        ↓
DisasterClaw
```

---

# 6. 做完四项后，Abstract 建议形成的最终版本

下面给一版仅围绕本轮修改后的摘要草案。数值沿用当前 v10，PX4/Gazebo 的结果暂不写进摘要，避免它喧宾夺主。

```text
Aerial disaster inspection requires deciding whether another observation will help answer the current question, not merely whether it will sharpen a damage prediction. We present DisasterClaw, a georeferenced imagery-grounded simulation and agent framework that links movement, rendered observations, binary building-damage perception, and terminal question answering in traceable episodes. A development-set camera-ladder diagnostic shows that a narrower footprint can both correct and harm building predictions. We then evaluate seven frozen online policies on 160 questions independently reviewed by two annotators from 44 regions in three disaster events, with three generation repeats (3,360 episodes). The task-conditioned policy attains 75.00% accuracy, compared with 72.29% for no reobservation, while executing 161 reobservations compared with 923 for the always-reobserve policy. Its observed gain over no reobservation is 2.71 percentage points, but the prespecified ROI-cluster, multiplicity-adjusted interval spans −2.04 to +7.50 points; the experiment therefore does not establish a stable advantage over holding. The results support a platform and measurement contribution: the value of an additional observation depends on the task, and view changes can produce both beneficial and harmful answer transitions. Conclusions are limited to rendered orthographic imagery, simplified motion, a fixed ChangeOS backbone with prior xBD exposure, and three evaluation events.
```

---

# 7. 做完四项后，Contributions 建议微调

当前三个 contributions 已经比旧版本好。建议仅把第 1、2 点进一步贴合新增文献和 action mapping。

```text
1. We develop a georeferenced aerial disaster-inspection environment and evaluation framework that converts paired disaster imagery into traceable closed-loop observation episodes. Unlike general aerial embodied benchmarks aimed at broad flight skills or simulator fidelity, the framework is designed to measure whether a controlled view change alters task-relevant evidence and the terminal disaster-inspection answer.

2. We implement a closed-loop inspection protocol that connects task language, rendered observations, a fixed binary ChangeOS detector, geographic cross-view evidence memory, movement tools, and Qwen2.5-VL-7B-Instruct answer generation. The protocol records beneficial and harmful answer transitions, distinguishes executed observation cost from geometric motion estimates, and provides a separate PX4 SITL/Gazebo check of command-level action correspondence.

3. We compare seven frozen policies over 160 dual-reviewed final questions and three generation repeats. The task-conditioned policy is numerically more accurate than holding, random reobservation, always reobserving, and raw entropy while using fewer reobservations, but the prespecified interval does not establish a stable advantage over holding or raw entropy. This mixed result is interpreted as evidence about task-dependent observation value rather than a claim of general policy superiority.
```

注意：第 2 点中 PX4/Gazebo 只有在你实际完成实验后才能保留。

---

# 8. 最终执行清单

### 文献与定位
- [ ] 加 AeroVerse
- [ ] 加 BEDI
- [ ] 加 OpenUAV / UAV-Need-Help
- [ ] 加 ThinkGeo
- [ ] 加 DisasterM3
- [ ] Introduction 增加“complementary measurement contribution”段
- [ ] 2.1 增加 aerial embodied benchmark 段
- [ ] 2.2 增加 remote-sensing agent / disaster VLM 段

### 160-question review
- [ ] Abstract 改成 two-annotator / dual-reviewed
- [ ] 5.1 替换单人 review 段
- [ ] Limitations 删除“没有第二标注者”
- [ ] Appendix A 加 dual-review provenance
- [ ] 如有真实 agreement/kappa，再填具体数字


### 投稿清理
- [ ] 删除 Page 1 Review status
- [ ] 填真实 authors / affiliations
- [ ] Table 1 `four-class` → `binary ChangeOS`
- [ ] Table 1 logging boundary 更新为 3,360 audited episodes
- [ ] Figure 1 `semantic map` 等旧文字与 final protocol 对齐
- [ ] 删除 Figure 4 placeholder；推荐主文完全删除 Figure 4
- [ ] 修正全部 `Appendix Appendix A/B`
- [ ] Generative AI disclosure 改正式语气
- [ ] Funding / CRediT / competing interest 填实
- [ ] 全文最终搜索：`placeholder`、`to be confirmed`、`to be supplied`、`author-review`、`Appendix Appendix`

---

## 最终定位建议

完成本轮四项后，论文应继续把 claim 控制在：

> **DisasterClaw is a georeferenced, imagery-grounded closed-loop environment and measurement framework for studying task-dependent active observation in UAV disaster inspection.**

不要改成：

> high-fidelity UAV flight simulator

也暂时不要写成：

> a generally superior task-conditioned flight policy

目前最强的价值仍然是：**把 observation action、perception change、answer correction/harm 和 terminal task outcome 放进同一条可审计证据链中。**
