# DisasterClaw：相关工作能力对照与当前局限性矩阵

> 调研日期：2026-09-21  
> 用途：回应模拟审稿意见中“增加一张紧凑的功能对照表”的要求，并同步梳理 DisasterClaw 当前仍存在的局限性、已经完成的缓解措施和论文主张边界。  
> 建议用途：本文档用于返修决策；其中“正文推荐表”可进一步压缩后放入 Related Work 或 Introduction 末尾。

---

## 1. 审稿意见实际要求的“矩阵”是什么

模拟审稿意见第 1 条并不是要求简单列出“别人有什么、我们没有什么”，而是要求把 **DisasterClaw 的新增评测能力** 与已有工作明确区分。审稿意见的核心意思是：

- 不需要证明“已有工作绝对做不到地理裁剪或缩放”；
- 需要说明哪些过去容易混淆的问题，被 DisasterClaw 变成了标准化、可记录、可复核的评测对象；
- 特别包括：
  1. 固定任务范围与变化视野的分离；
  2. 跨视角证据关联；
  3. 请求动作与实际执行动作的区分；
  4. 生成随机性与观测变化的区分；
  5. 一个真实冻结回合的可重放 action → observation/evidence → answer 链。

因此，最适合的矩阵不是“综合能力排行榜”，而是一个 **benchmark capability matrix**。它应该同时承认其他平台在 3D 场景、真实飞行、规模和通用任务上更强，只突出 DisasterClaw 所标准化的“任务相关观测价值测量”这一窄而明确的能力。

---

## 2. 符号说明

| 符号 | 含义 |
|---|---|
| ✓ | 论文/官方页面明确将该能力作为任务或评测的一部分 |
| △ | 存在相邻能力，但不是同一评测定义，或主要体现在具体 agent 方法而非 benchmark contract |
| — | 该工作主要不以此为评测目标；这里表示“not designed/reported for this axis”，不等同于“技术上绝对无法实现” |
| ? | 根据当前公开的一手资料无法可靠判断 |

**重要：**正文中最好避免写成“Only DisasterClaw can ...”。更稳妥的表达是：  
**“Among the compared benchmarks, DisasterClaw explicitly standardizes ... as part of the evaluation contract.”**

---

# 3. 相关工作完整能力矩阵

## 3.1 Reviewer-oriented capability matrix

| Work | Primary setting | Agent action changes observation | 3D / real-flight viewpoint realism | Disaster-specific | Fixed geographic ROI + annotation-derived answer independent of current view | Cross-view evidence association as benchmark/system state | Requested vs. executed action explicitly separated | Final answer evaluated after active sensing | Matched fixed-view control isolating observation change |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **AerialVLN** (ICCV 2023) | UAV vision-language navigation in 3D simulator | ✓ | ✓ | — | — | △ history is relevant to navigation, but object-level evidence association is not the benchmark target | — | — navigation success is primary | — |
| **CityNav** (ICCV 2025) | Real-world aerial VLN with geographic semantic maps | ✓ | △ real-city data / navigation benchmark, not DisasterClaw-style replayed disaster sensing | — | — | △ geographic semantic map assists navigation | — | — navigation success is primary | — |
| **OpenFly** (2025–2026) | Large-scale aerial VLN platform and benchmark | ✓ | ✓ multiple rendering engines, including UE/GTA/Google Earth/3DGS | — | — | △ OpenFly-Agent explicitly uses historical keyframes, but benchmark does not standardize geographic evidence identity in the DisasterClaw sense | — | — navigation metrics are primary | — |
| **AeroVerse** (2024–) | Broad UAV-agent / aerospace embodied benchmark suite | ✓ / △ across navigation, planning and motion tasks | ✓ / △ includes simulator and ego datasets | — | — | △ supports embodied/spatial reasoning, but not a fixed ROI answer contract | — | △ some downstream tasks produce semantic/planning outputs, not a fixed post-observation disaster answer | — |
| **BEDI** (ISPRS JPRS 2026) | Standardized UAV embodied-agent evaluation with perception–decision–action chains | ✓ | ✓ hybrid real/virtual, static/dynamic scenarios | △ includes fire/flood and other scenarios, but is not disaster-specific | — | △ Dynamic Chain-of-Embodied-Task evaluates multi-stage loops, but not a fixed geographic disaster-answer identity across views | △ step/loop/action chains are explicit, but requested-vs-executed action is not the central distinction reported in the public description | △ task-level evaluation exists, but not the same fixed-answer active-sensing contract | — |
| **CityEQA** (EMNLP 2025) | Embodied QA through active exploration in realistic 3D city space | ✓ | ✓ | — | △ question/reference answer stays fixed during exploration, but it is not defined as a fixed geographic ROI with annotation-derived disaster answer independent of observation footprint | △ PMA uses an object-centric cognitive map; this is an agent design rather than a benchmark-wide cross-view evidence identity contract | — | ✓ | — |
| **RescueADI** (2024) | Disaster remote-sensing agent for sequential interpretation/tool use | — viewpoint is based on given RS imagery; actions are interpretation/tool operations | — | ✓ | — | △ sequential planning/perception/recognition, but not cross-view sensing memory | — | △ final interpretation request is evaluated, but not after embodied view change | — |
| **ThinkGeo** (2025) | Tool-augmented remote-sensing agent benchmark | — / △ tool calls change derived evidence, not UAV viewpoint | — | △ includes disaster assessment among many applications | — | △ ReAct/tool history is evaluated step-wise, but not cross-view object association | △ tool execution traces are evaluated, but not UAV requested-vs-executed motion | △ final answer correctness is evaluated | — |
| **DisasterM3** (NeurIPS 2025 dataset) | Multi-hazard, multi-sensor disaster VLM benchmark | — | — | ✓ | — | — | — | — static/bitemporal instruction following, not active sensing | — |
| **DisasterClaw** (current) | Georeferenced disaster imagery inspection benchmark | ✓ controlled search/recheck changes rendered observation | △ **2D orthographic resampling / crop-scale change, not true new 3D view** | ✓ | **✓** | **✓** geographic evidence tracks / memory | **✓** requested action, executed action, observation ID and answer call are logged separately | **✓** | **✓** matched-call fixed-view branch at executed rechecks |

### 3.2 这个矩阵真正支持的结论

该矩阵**不支持**“DisasterClaw 比所有相关平台更全面”。

它支持的是更窄、更稳妥的结论：

> AerialVLN、CityNav 和 OpenFly 主要回答“无人机能否按照语言目标完成导航”；AeroVerse 和 BEDI 主要回答“UAV embodied agent 的多类感知—决策—动作能力如何评测”；CityEQA 回答“主动探索后能否回答城市空间问题”；RescueADI、ThinkGeo 和 DisasterM3 则重点覆盖遥感/灾害理解、工具调用或多模态推理。  
> DisasterClaw 的差异化能力在于：**把一个固定地理任务及其参考答案，与可变化的观测窗口分离，并把动作、跨视角预测证据、回答调用和最终答案绑定在同一个可审计 episode 中；随后通过 fixed-view matched-call control 直接检查“新观测”与“只是再回答一次”的差异。**

这与审稿意见要求的“新增评测能力，而不是系统集成”一致。

---

# 4. 建议放进论文正文的紧凑版表格

正文不建议放上面的 10×9 大表。可压缩为以下版本。

## Table X. Comparison of evaluation capabilities most relevant to task-dependent observation value

| Benchmark | Primary task | Active observation change | Disaster-specific | Fixed geo task/answer under view change | Answer after active sensing | Explicit observation-value isolation |
|---|---|---:|---:|---:|---:|---:|
| AerialVLN | Aerial VLN | ✓ | — | — | — | — |
| OpenFly | Aerial VLN | ✓ | — | — | — | — |
| BEDI | UAV embodied evaluation | ✓ | △ | — | △ | — |
| CityEQA | Embodied QA | ✓ | — | △ | ✓ | — |
| RescueADI | Disaster RS agent | — | ✓ | — | △ | — |
| ThinkGeo | RS tool-use agent | △ | △ | — | △ | — |
| DisasterM3 | Disaster VLM | — | ✓ | — | — | — |
| **DisasterClaw** | Disaster inspection / observation value | **✓** | **✓** | **✓** | **✓** | **✓** |

**表注建议：**

> “—” means that the capability is not the primary evaluation contract reported by the work, rather than that it is technically impossible to implement. “△” denotes a related but non-equivalent capability. DisasterClaw’s active-view setting uses controlled orthographic imagery rather than physical 3D flight.

这一句非常重要，可以防止审稿人认为作者在“人为做表突出自己”。

---

# 5. 可直接用于 Related Work 的英文解释段落

> Existing aerial embodied benchmarks cover complementary evaluation regimes. AerialVLN, CityNav, and OpenFly emphasize language-conditioned aerial navigation, while AeroVerse and BEDI broaden evaluation toward perception, spatial reasoning, planning, motion, and perception–decision–action loops. CityEQA evaluates question answering after active exploration in a realistic urban simulator. In remote sensing, RescueADI and ThinkGeo evaluate sequential interpretation or structured tool use, and DisasterM3 provides broad disaster-specific multimodal perception and reasoning tasks. These benchmarks do not primarily standardize the narrower measurement studied here: keeping an annotation-derived geographic task and reference answer fixed while the observation footprint changes, linking each executed observation to geographic evidence and answer calls, and explicitly controlling for an additional answer opportunity. DisasterClaw targets this evaluation contract rather than simulator realism or general UAV-agent capability.

建议在此段后直接接 capability table。

---

# 6. DisasterClaw 当前工作局限性矩阵

这是和“相关工作能力矩阵”不同的第二张矩阵。它回答：**当前稿件还有哪些未解决问题，哪些已经缓解，哪些必须通过主张边界处理。**

| ID | 当前局限性 | 当前证据/状态 | 对结论的影响 | 已有缓解 | 建议处理方式 | 优先级 |
|---|---|---|---|---|---|---|
| L1 | 最终集仅 44 ROIs、3 个事件 | 当前 final set = 160 questions / 44 ROIs / 3 events | 不能推断未知灾害或更大地域泛化 | ROI-cluster bootstrap；event breakdown 仅作描述 | 保留 limitation；不要宣称 unseen-disaster generalization | 高，但本轮无需扩数据 |
| L2 | Count 答案分布明显不均衡 | 0=14, 1=1, 2=0, 3+=28 | Count accuracy 可能部分来自类别先验 | 已加入 development-fitted task-majority baseline | 保留分布表和 S0；不要用 Count 单项结果证明四档辨识能力 | 已较好处理 |
| L3 | Shortcut controls 仍不可能穷尽所有非视觉捷径 | S1 language-only、S2 metadata-only 已明显低于 image-aware path | 能证明当前已测捷径不足以解释全部性能，但不能证明每题都必须看图 | 已增加 S0/S1/S2 | 用 “beyond the measured text/metadata controls”，避免 “all tasks require vision” | 已较好处理 |
| L4 | 固定 ChangeOS checkpoint 具有 prior xBD exposure | 当前 manuscript 已承认 | 可能高估同域灾害图像上的感知表现；不支持完全独立的 unseen-xBD 泛化 | 主张已收缩 | 明确这是 benchmark instantiation，不是 backbone 泛化研究 | 中 |
| L5 | Camera ladder 混合了 footprint/context、重采样和 detector-scale effects | pre/post preprocessing 非对称 | 不能把变化因果归为“真实下降带来更多物理细节” | 已增加 preprocessing construction table；正文明确不归因于纯 GSD | 把 camera ladder 定位为 “changed rendered evidence diagnostic” | 已基本处理 |
| L6 | 不是真实 3D/6-DoF 飞行或真实新视角 | 正射影像 crop/scale；无新角度、遮挡、采集时间、运动动力学 | 不能声称真实 UAV descent/viewpoint improvement | 标题/摘要已强调 imagery benchmark / controlled observation | 在 capability table 中主动承认该项弱于 OpenFly/BEDI/CityEQA | 关键主张边界 |
| L7 | Matched-call 是 transition-level，而非完整 terminal counterfactual | 161 executed rechecks 被 fork；不是从分支继续跑完整回合 | 能隔离局部新观测 vs 再回答一次，但不能给出完整 episode causal effect | 已明确 candidate-answer transition diagnostic | 不能把 +9.32 pp 写成完整策略因果效应 | 已较好处理 |
| L8 | Fixed-view rerun 仍可能有 perception nondeterminism | 118/119 episode schedule 完全复现，1 个 mismatch | 小部分差异可能来自感知路径重跑而非图像变化 | byte-identical fixed image、shared seed、严格 call gate | 建议补 “original pre-recheck vs fixed rerun” stability/sensitivity；至少排除 mismatch episode 再算一次 | 建议补 |
| L9 | Spatial uncertainty 没有完整建模 target identity uncertainty | 最近受损目标可能在 recheck 后切换；当前 harm case 即发生 target switch | 解释 Spatial harm 时必须承认目标身份排序是关键来源 | 当前正文已写 “does not fully represent uncertainty about which damaged building is nearest” | 把 Figure 3 的 spatial case 明确标注为 target-identity switch | 高价值解释 |
| L10 | Count uncertainty 不建模漏检建筑 | 巡航状态 detector recall 较低；Count entropy 只基于已检测对象 | 可能对错误的 count 产生虚假高置信度 | 正文已明确 | 不把 T1 gate 宣称为 calibrated expected task gain；未来工作可做 missed-object uncertainty | 中 |
| L11 | Policy cost 不完整 | recheck count 不含 search/render/perception/Qwen；search detail 不全 | 无法得出“总成本最低/最高效率” | 已明确 A0_NO_RECHECK 仍允许 search | 删除/避免 efficiency claims；recheck 仅称 “dedicated recheck count” | 已处理 |
| L12 | 42/3360 episodes 缺 terminal fly 后的 recorded state | terminal-answer denominator 仍完整；complete-motion summary 受影响 | 影响完整运动成本/轨迹重放，不影响终答准确率分母 | 已分开报告 3318 vs 3360 | 审计表继续分开写 execution success 与 trajectory completeness | 中 |
| L13 | 最小公开复现包仍未最终落实 | 本地已有 protocol/hash/scripts/report；public archive/license 未确认 | 平台论文第三方独立复核仍受限 | 内部 provenance 明显增强 | 投稿前提供匿名 minimal reproduction package：合法场景构建 + one episode replay + scoring check | 投稿前必须 |
| L14 | Human review 的独立双审链未完全可核验 | 当前文本已降为 “reviewed by people before freezing” | 若没有独立记录，不能继续声称 independent double review | 已降低主张 | 有原始双审记录则整理；没有则保持当前弱表述 | 中 |
| L15 | Dev/final spatial overlap 尚缺显式 audit | 不同 tile/ROI 不自动保证视野内容独立 | 可能增加相关性，不等于已经发生泄漏 | ROI cluster inference 部分缓解统计独立性 | 低成本报告 building-ID / ROI / observation-footprint overlap | 建议补 |
| L16 | Camera-ladder detection 的 false-positive side 仍不足 | 主要报告 localization recall 和 fixed-denominator building accuracy | Count/Presence/Spatial 也可能受误检影响 | 暂无充分补充 | 如现有日志可直接计算，补 precision / FP per ROI 到 supplement | 建议补 |
| L17 | Policy comparison不是 matched total-cost / matched-call comparison | 各策略后续行为与调用数不同 | T1 vs A0/Random/Always 是 whole-policy operating points | 已把 policy comparison 放到 secondary evidence | 保持 “operating-point comparison”，不把 terminal difference解释为单次新观测因果效应 | 已处理 |

---

# 7. 当前最值得保留的“限制—贡献”对应关系

DisasterClaw 的局限性并不要求全部消除。更重要的是把每个局限性与允许的主张一一对应。

| 局限 | 因此不能声称 | 仍然可以声称 |
|---|---|---|
| 正射影像重采样，不是真实新视角 | “真实下降必然改善感知” | “受控观测窗口变化会改变可用预测证据” |
| 只有 3 个灾害事件 | “可泛化到未知灾害” | “在当前冻结集上可测量 task-dependent help/harm” |
| +9.32 pp CI 跨零 | “changed view 稳定提高平均准确率” | “changed view 产生可观测的 corrections 和 harms，并改变 answer path” |
| matched-call 不是 terminal counterfactual | “单次 recheck 对最终 episode 的因果效应为 +9.32 pp” | “在匹配 answer opportunity 的 transition 上，changed view 与 fixed view 可直接比较” |
| cost 不完整 | “T1 最省资源” | “T1 执行的 dedicated recheck 数少于某些 active policies” |
| Spatial target identity 可切换 | “更近视野一定提供更正确空间信息” | “空间任务揭示局部细节与目标身份/区域覆盖之间的冲突” |
| 公开复现包未完成 | “第三方已经可以完整重建环境” | “本地 protocol/hash/log/script 已建立可审计 provenance 链” |

这张表适合用于作者内部检查，不一定要进入正文。

---

# 8. 建议正文如何利用这项调研

## 8.1 Introduction

保留当前的核心问题：

> when does a changed view improve the requested answer, and when does it discard necessary context?

随后用 1–2 句指出现有工作覆盖 navigation、embodied QA、tool-use RS reasoning，但 **“fixed geographic task + variable observation + traceable evidence/answer + matched-call diagnostic”** 并没有被作为统一 benchmark contract 来测量。

## 8.2 Related Work

建议结构：

### 2.1 Aerial interaction and embodied evaluation
AerialVLN → CityNav → OpenFly → AeroVerse → BEDI → CityEQA

强调：
- 它们证明“行动改变观测”和“UAV embodied evaluation”本身已有大量工作；
- 不要宣称 DisasterClaw 首次闭环、首次 UAV agent、首次 active perception；
- DisasterClaw 的差异是 **measurement target**。

### 2.2 Disaster / remote-sensing agent benchmarks
RescueADI → ThinkGeo → DisasterM3

强调：
- 它们证明 disaster/RS reasoning、工具调用和多任务理解已有强基准；
- DisasterClaw 不应争夺“灾害遥感数据集规模”或“通用 RS agent”；
- 差异在于 agent 是否可以改变 observation，并评估 observation change 对同一固定 geographic answer 的帮助/伤害。

### 2.3 Task value of an observation
引出：
- action–observation–answer；
- matched fixed-view control；
- help/harm；
- task dependence。

---

# 9. 不建议使用的表述

为了避免 capability matrix 被审稿人反驳，建议不要写：

- “Existing UAV benchmarks are static.”
  - BEDI 明确包含 dynamic real/virtual settings；AerialVLN/OpenFly/CityEQA 也有 active interaction。
- “Existing benchmarks do not support closed-loop interaction.”
  - AerialVLN、OpenFly、BEDI、CityEQA 都涉及闭环或主动交互。
- “Existing work cannot link actions and observations.”
  - 多种 embodied benchmarks 本身就记录动作与观测。
- “DisasterClaw is the first UAV embodied benchmark.”
  - 明显不成立，BEDI/AeroVerse 等已覆盖。
- “DisasterClaw provides more realistic UAV simulation.”
  - 当前正射图像渲染远弱于 UE/AirSim/3DGS/真实视频等设置。

更稳妥的是：

> Existing benchmarks standardize navigation, embodied capabilities, active EQA, or tool-augmented remote-sensing reasoning. DisasterClaw instead standardizes a narrower evaluation contract for task-dependent observation value: the geographic task and reference answer remain fixed while the observation changes, and the episode record exposes the corresponding action, predicted evidence, answer call, and matched fixed-view control.

---

# 10. 推荐插入论文的最终版本

如果版面只允许一张表，我建议使用第 4 节的紧凑表，并在表后加下面这段：

> The comparison is capability-oriented rather than a ranking of benchmark generality. Several prior platforms provide substantially richer 3D simulation, real-world flight data, or broader embodied tasks than DisasterClaw. Our narrower contribution is to standardize the measurement of task-dependent observation value under a fixed geographic question and answer, including geographic evidence traces and a fixed-view matched-call diagnostic.

这段话能主动承认 DisasterClaw 在仿真真实性和任务广度上的局限，同时把新增价值锁定在 reviewer 最关心的 evaluation capability 上。

---

# 11. 一手资料与参考来源

以下优先使用论文官方页面、会议页面、出版社或 arXiv：

1. **AerialVLN: Vision-and-Language Navigation for UAVs.** ICCV 2023.  
   https://openaccess.thecvf.com/content/ICCV2023/html/Liu_AerialVLN_Vision-and-Language_Navigation_for_UAVs_ICCV_2023_paper.html

2. **CityNav: A Large-Scale Dataset for Real-World Aerial Navigation.** ICCV 2025.  
   https://openaccess.thecvf.com/content/ICCV2025/html/Lee_CityNav_A_Large-Scale_Dataset_for_Real-World_Aerial_Navigation_ICCV_2025_paper.html

3. **OpenFly: A Comprehensive / Versatile Platform for Aerial Vision-Language Navigation.** arXiv:2502.18041.  
   https://arxiv.org/abs/2502.18041

4. **AeroVerse: UAV-Agent Benchmark Suite for Simulating, Pre-training, Finetuning, and Evaluating Aerospace Embodied World Models.** arXiv:2408.15511.  
   https://arxiv.org/abs/2408.15511

5. **BEDI: a comprehensive benchmark for evaluating embodied agents on UAVs.** ISPRS Journal of Photogrammetry and Remote Sensing, 2026.  
   https://doi.org/10.1016/j.isprsjprs.2026.01.013

6. **CityEQA: A Hierarchical LLM Agent on Embodied Question Answering Benchmark in City Space.** EMNLP 2025.  
   https://aclanthology.org/2025.emnlp-main.630/

7. **RescueADI: Adaptive Disaster Interpretation in Remote Sensing Images with Autonomous Agents.** arXiv:2410.13384.  
   https://arxiv.org/abs/2410.13384

8. **ThinkGeo: Evaluating Tool-Augmented Agents for Remote Sensing Tasks.** arXiv:2505.23752.  
   https://arxiv.org/abs/2505.23752

9. **DisasterM3: A Remote Sensing Vision-Language Dataset for Disaster Damage Assessment and Response.** arXiv:2505.21089 / NeurIPS 2025.  
   https://arxiv.org/abs/2505.21089

10. **DisasterClaw current manuscript and simulated reviewer report.**  
    本文档中的 DisasterClaw 能力与局限性来自当前返修稿、冻结实验和审稿意见，不额外扩张其主张。

---

# 12. 最终建议

当前返修阶段最合适的做法不是再增加一个“我们什么都比别人强”的大矩阵，而是：

1. 在 Related Work 增加一张 **capability-oriented compact table**；
2. 明确承认 OpenFly/BEDI/CityEQA 在 3D、动态场景或通用 embodied evaluation 上更强；
3. 只把 DisasterClaw 的独特定位放在：
   - fixed geographic task / invariant answer；
   - variable observation；
   - cross-view geographic evidence；
   - action–evidence–answer trace；
   - matched fixed-view control；
   - task-dependent help/harm；
4. 在 Limitations 中保留当前主张边界，不需要为了“填满矩阵”再开展大规模新实验。

这样最符合审稿报告最后强调的方向：论文最值得保留的是 **traceable observation-value measurement framework**，而不是策略准确率排序。
